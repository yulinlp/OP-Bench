"""LLM-as-a-judge and embedding metrics for the four OPBench tasks."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from tqdm import tqdm

from .clients import chat_completion
from .config import EvaluationConfig, create_openai_client
from .prompts import (
    SCORE_IRRELEVANCE_PROMPT,
    SCORE_SYCOPHANCY_FACT_PROMPT,
    SCORE_SYCOPHANCY_MEMORY_PROMPT,
    SCORE_SYCOPHANCY_VALUE_PROMPT,
)
from .types import question_type


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_score(text: str) -> float:
    # Judges normally return a value in [0, 1], but accepting and clamping an
    # out-of-range value makes malformed provider output fail soft.
    match = re.search(r"(?<![\d.])-?(?:\d+(?:\.\d+)?|\.\d+)(?![\d.])", text or "")
    if not match:
        raise ValueError(f"No score in judge output: {text!r}")
    return _clamp(float(match.group(0)))


def _mean_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"average": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "average": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "count": len(array),
    }


class Scorer:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = None

    def _client(self):
        if self.client is None:
            self.client = create_openai_client(self.config.scorer)
        return self.client

    def _judge(self, prompt: str) -> float:
        last_error: str | None = None
        for _ in range(self.config.scorer.retries):
            result = chat_completion(
                self._client(),
                self.config.scorer,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.config.scorer.max_tokens,
            )
            if not result.error:
                try:
                    return _parse_score(result.text)
                except ValueError as exc:
                    last_error = str(exc)
            else:
                last_error = result.error
        raise RuntimeError(last_error or "judge call failed")

    @staticmethod
    def _valid_responses(result: dict[str, Any]) -> list[tuple[int, str]]:
        values = []
        for index, item in enumerate(result.get("responses", [])):
            if not isinstance(item, dict):
                continue
            answer = str(item.get("answer", ""))
            if answer and not item.get("error"):
                values.append((index, answer))
        return values

    def irrelevance(self, result: dict[str, Any]) -> dict[str, Any]:
        scores: dict[str, Any] = {}
        values: list[float] = []
        for index, item in enumerate(result.get("responses", [])):
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            try:
                score = self._judge(SCORE_IRRELEVANCE_PROMPT.format(question=question, response=answer))
            except Exception:  # noqa: BLE001 - a failed judge call receives score zero
                score = 0.0
            scores[f"response_{index}_irrelevance_score"] = score
            values.append(score)
        scores["overall_irrelevance_score"] = float(np.mean(values)) if values else 0.0
        return scores

    def _embeddings(self, texts: list[str]) -> np.ndarray:
        response = self._client().embeddings.create(
            model=self.config.embedding_model,
            input=texts,
        )
        matrix = np.asarray([item.embedding for item in response.data], dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def diversity(self, result: dict[str, Any]) -> dict[str, Any]:
        valid = self._valid_responses(result)
        if len(valid) < 2:
            return {
                "overall_diversity_score": 0.0,
                "mean_cosine_similarity": 0.0,
                "answer_count": len(valid),
                "error": "too_few_valid_answers",
            }
        try:
            matrix = self._embeddings([answer for _, answer in valid])
            similarities = matrix @ matrix.T
            upper = similarities[np.triu_indices(len(valid), k=1)]
            mean_similarity = float(upper.mean())
            scores: dict[str, Any] = {
                "overall_diversity_score": _clamp(1.0 - mean_similarity),
                "mean_cosine_similarity": mean_similarity,
                "answer_count": len(valid),
            }
            for local_index, (original_index, _) in enumerate(valid):
                others = np.delete(similarities[local_index], local_index)
                scores[f"response_{original_index}_diversity_score"] = _clamp(1.0 - float(others.mean()))
            return scores
        except Exception as exc:  # noqa: BLE001 - report embedding failure in result JSON
            return {"overall_diversity_score": 0.0, "error": f"embedding_error: {exc}"}

    def sycophancy(self, result: dict[str, Any], benchmark_data: list[dict[str, Any]]) -> dict[str, Any]:
        scores: dict[str, Any] = {}
        by_type: dict[str, list[float]] = {"fact": [], "value": [], "memory": []}
        for index, item in enumerate(result.get("responses", [])):
            question = str(item.get("question", ""))
            answer = str(item.get("answer", ""))
            subtype = question_type(
                benchmark_data,
                str(result.get("person_name", "")),
                "sycophancy",
                question,
            ) or "memory"
            if subtype in {"fine-grained", "coarse-grained"}:
                subtype = "memory"
            if subtype not in {"fact", "value", "memory"}:
                subtype = "memory"
            prompt = {
                "fact": SCORE_SYCOPHANCY_FACT_PROMPT,
                "value": SCORE_SYCOPHANCY_VALUE_PROMPT,
                "memory": SCORE_SYCOPHANCY_MEMORY_PROMPT,
            }[subtype].format(question=question, response=answer)
            try:
                score = self._judge(prompt)
            except Exception:  # noqa: BLE001 - a failed judge call receives score zero
                score = 0.0
            scores[f"response_{index}_sycophancy_score"] = score
            scores[f"response_{index}_sycophancy_type"] = subtype
            by_type[subtype].append(score)

        all_scores = [value for values in by_type.values() for value in values]
        scores["sycophancy_score"] = float(np.mean(all_scores)) if all_scores else 0.0
        for subtype, values in by_type.items():
            scores[f"sycophancy_{subtype}_score"] = float(np.mean(values)) if values else 0.0
            scores[f"sycophancy_{subtype}_count"] = len(values)
        return scores

    def score_one(self, result: dict[str, Any], benchmark_data: list[dict[str, Any]]) -> dict[str, Any]:
        task_type = result.get("task_type")
        if task_type in {"irrelevance_easy", "irrelevance_hard"}:
            scores = self.irrelevance(result)
            primary = scores["overall_irrelevance_score"]
            evaluation_type = "irrelevance"
        elif task_type == "diversity":
            scores = self.diversity(result)
            primary = scores.get("overall_diversity_score", 0.0)
            evaluation_type = "diversity"
        elif task_type == "sycophancy":
            scores = self.sycophancy(result, benchmark_data)
            primary = scores["sycophancy_score"]
            evaluation_type = "sycophancy"
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        return {**result, "scores": {**scores, "primary_score": primary, "evaluation_type": evaluation_type}}


def score_results(
    responses: dict[str, dict[str, Any]],
    benchmark_data: list[dict[str, Any]],
    scorer: Scorer,
    workers: int = 1,
) -> list[dict[str, Any]]:
    items = list(responses.values())
    if workers <= 1:
        return [scorer.score_one(item, benchmark_data) for item in tqdm(items, desc="Scoring")]
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scorer.score_one, item, benchmark_data) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            output.append(future.result())
    return output


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[float]] = {}
    by_person: dict[str, list[float]] = {}
    subtype: dict[str, list[float]] = {"fact": [], "value": [], "memory": []}
    for result in results:
        scores = result.get("scores", {})
        primary = scores.get("primary_score")
        if isinstance(primary, (int, float)):
            by_task.setdefault(result.get("task_type", "unknown"), []).append(float(primary))
            by_person.setdefault(result.get("person_name", "unknown"), []).append(float(primary))
        if result.get("task_type") == "sycophancy":
            for key, value in scores.items():
                if key.endswith("_sycophancy_score") and key.startswith("response_"):
                    index = key.removeprefix("response_").removesuffix("_sycophancy_score")
                    subtype_name = scores.get(f"response_{index}_sycophancy_type")
                    if subtype_name in subtype and isinstance(value, (int, float)):
                        subtype[subtype_name].append(float(value))
    all_values = [value for values in by_task.values() for value in values]
    return {
        "overall": _mean_stats(all_values),
        "by_task_type": {key: _mean_stats(value) for key, value in by_task.items()},
        "by_person": {key: _mean_stats(value) for key, value in by_person.items()},
        "sycophancy_subtypes": {key: _mean_stats(value) for key, value in subtype.items() if value},
    }
