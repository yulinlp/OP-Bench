"""End-to-end OPBench search, generation, scoring, and metric commands."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .clients import AgentRouter, MemosAPIClient, chat_completion, memory_text
from .config import EvaluationConfig, create_openai_client
from .io import read_json, slug, write_json
from .postprocessing import ContextProcessor
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_WITH_MEMORY,
    ANSWER_USER_PROMPT_WITHOUT_MEMORY,
)
from .scoring import Scorer, aggregate_metrics, score_results
from .types import PERSON_TO_CONV_IDX, TestCase, extract_test_cases


def _user_id(person_name: str, version: str) -> str:
    if person_name not in PERSON_TO_CONV_IDX:
        raise KeyError(f"No LoCoMo index is registered for {person_name!r}")
    return f"locomo_exp_user_{PERSON_TO_CONV_IDX[person_name]}_speaker_a_{version}"


def _context(person_name: str, value: Any) -> str:
    return f"Memories for user {person_name}:\n\n{memory_text(value)}"


def _run_parallel(items: list[Any], fn: Callable[[Any], Any], workers: int, label: str) -> list[Any]:
    if workers <= 1:
        return [fn(item) for item in tqdm(items, desc=label)]
    output: list[Any] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=label):
            output.append(future.result())
    return output


def _cases(benchmark_path: str | Path, config: EvaluationConfig) -> tuple[list[dict[str, Any]], list[TestCase]]:
    benchmark = read_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise TypeError("The benchmark JSON must contain a list")
    return benchmark, extract_test_cases(
        benchmark,
        config.task_types,
        config.use_both_personas,
    )


def run_search(
    benchmark_path: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig,
    *,
    frame: str = "memos-api-online",
    version: str = "default",
    online: bool = True,
) -> Path:
    """Retrieve top-k memories from a MemOS service for every OPBench question."""

    _, cases = _cases(benchmark_path, config)
    client = MemosAPIClient(online=online, timeout=config.agent.timeout)
    output: dict[str, Any] = {}

    def one(case: TestCase) -> dict[str, Any]:
        results = []
        user_id = _user_id(case.person_name, version)
        for question in case.questions:
            started = time.perf_counter()
            try:
                raw = client.search(question, user_id, config.top_k)
                context = _context(case.person_name, raw)
                error = None
            except Exception as exc:  # noqa: BLE001 - record one failed question and continue
                context = ""
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "question": question,
                    "context": context,
                    "search_duration_ms": (time.perf_counter() - started) * 1000,
                    "error": error,
                }
            )
        return {
            "test_id": case.test_id,
            "task_type": case.task_type,
            "person_name": case.person_name,
            "results": results,
        }

    for value in _run_parallel(cases, one, config.workers, "Searching memories"):
        output[value["test_id"]] = value
    target = Path(output_dir) / "search" / f"{slug(frame)}_{slug(version)}.json"
    write_json(target, output)
    return target


def _answer_messages(question: str, context: str, use_memory: bool) -> list[dict[str, str]]:
    if use_memory and context.strip():
        prompt = ANSWER_USER_PROMPT_WITH_MEMORY.format(memory=context, question=question)
    else:
        prompt = ANSWER_USER_PROMPT_WITHOUT_MEMORY.format(question=question)
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _answer_with_retry(client: Any, config: EvaluationConfig, messages: list[dict[str, str]]) -> Any:
    last = None
    for attempt in range(config.agent.retries):
        result = chat_completion(client, config.agent, messages)
        if not result.error:
            return result
        last = result
        if attempt + 1 < config.agent.retries:
            time.sleep(min(2**attempt, 8))
    return last


def run_generate(
    benchmark_path: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig,
    *,
    frame: str,
    version: str = "default",
    search_path: str | Path | None = None,
    use_memory: bool = True,
    postprocess: str = "none",
) -> Path:
    """Generate answers either through a per-person agent or from MemOS context."""

    _, cases = _cases(benchmark_path, config)
    output: dict[str, Any] = {}
    agent_frame = frame.lower() in {"agent", "ldagent", "simplerag"}

    if agent_frame:
        router = AgentRouter(config)

        def generate_case(case: TestCase) -> dict[str, Any]:
            values = []
            for question in case.questions:
                result = router.chat(case.person_name, question)
                values.append(
                    {
                        "question": question,
                        "answer": result.text,
                        "context": "",
                        "search_duration_ms": 0.0,
                        "response_duration_ms": result.duration_ms,
                        "error": result.error,
                    }
                )
            return {
                "test_id": case.test_id,
                "task_type": case.task_type,
                "person_name": case.person_name,
                "responses": values,
            }

        values = _run_parallel(cases, generate_case, config.workers, "Generating agent answers")
    else:
        if search_path is None:
            search_path = Path(output_dir) / "search" / f"{slug(frame)}_{slug(version)}.json"
        search_data = read_json(search_path)
        client = create_openai_client(config.agent)
        processor = ContextProcessor(client, config.agent)

        def generate_case(case: TestCase) -> dict[str, Any]:
            search_result = search_data.get(case.test_id, {})
            responses = []
            for item in search_result.get("results", []):
                question = str(item.get("question", ""))
                original_context = str(item.get("context", ""))
                context = original_context
                postprocess_ms = 0.0
                postprocess_error = None
                if use_memory and postprocess not in {"", "none", "self_critic"}:
                    context, postprocess_ms, postprocess_error = processor.filter(
                        postprocess, question, context
                    )
                    if postprocess_error:
                        context = original_context

                if postprocess == "self_critic":
                    initial = _answer_with_retry(client, config, _answer_messages(question, context, use_memory))
                    revised, critic_ms, critic_error = processor.revise(question, context, initial.text)
                    answer = revised if not critic_error and revised else initial.text
                    result_error = critic_error or initial.error
                    duration_ms = initial.duration_ms + critic_ms
                    postprocess_ms = critic_ms
                else:
                    result = _answer_with_retry(
                        client, config, _answer_messages(question, context, use_memory)
                    )
                    answer = result.text
                    result_error = result.error
                    duration_ms = result.duration_ms
                responses.append(
                    {
                        "question": question,
                        "answer": answer,
                        "context": context if use_memory else "",
                        "search_duration_ms": item.get("search_duration_ms", 0.0),
                        "response_duration_ms": duration_ms,
                        "postprocess_duration_ms": postprocess_ms,
                        "error": result_error or postprocess_error,
                    }
                )
            return {
                "test_id": case.test_id,
                "task_type": case.task_type,
                "person_name": case.person_name,
                "responses": responses,
            }

        values = _run_parallel(cases, generate_case, config.workers, "Generating answers")

    for value in values:
        output[value["test_id"]] = value
    target = Path(output_dir) / "responses" / (
        f"{slug(frame)}_{slug(config.agent.model)}_{slug(postprocess or 'none')}.json"
    )
    write_json(target, output)
    return target


def run_score(
    benchmark_path: str | Path,
    responses_path: str | Path,
    output_dir: str | Path,
    config: EvaluationConfig,
    *,
    frame: str,
) -> Path:
    benchmark = read_json(benchmark_path)
    responses = read_json(responses_path)
    if not isinstance(responses, dict):
        raise TypeError("Responses JSON must be an object keyed by test_id")
    judged = score_results(responses, benchmark, Scorer(config), config.workers)
    target = Path(output_dir) / "judged" / f"{slug(frame)}_{slug(config.scorer.model)}.json"
    write_json(
        target,
        {
            "frame": frame,
            "benchmark": str(benchmark_path),
            "responses": str(responses_path),
            "results": judged,
        },
    )
    return target


def run_metrics(judged_path: str | Path, output_dir: str | Path, *, frame: str) -> Path:
    data = read_json(judged_path)
    results = data.get("results", []) if isinstance(data, dict) else []
    report = {"frame": frame, "metrics": aggregate_metrics(results)}
    target = Path(output_dir) / "metrics" / f"{slug(frame)}.json"
    write_json(target, report)
    return target
