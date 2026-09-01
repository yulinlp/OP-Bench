"""Build the OPBench task file from the LoCoMo conversations.

The original experiment script mixed data extraction, API credentials, retry
loops, and several unsafe thread-pool patterns in one file.  This version keeps
the same task schema while making paths/configuration explicit and resumable.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, ClassVar

from tqdm import tqdm

from .clients import chat_completion
from .config import EndpointConfig, create_openai_client
from .io import read_json, read_text, write_json


class BenchmarkBuilder:
    PREDEFINED_TOPICS: ClassVar[list[str]] = [
        "Ordinary Life",
        "School Life",
        "Culture & Education",
        "Attitude & Emotion",
        "Relationship",
        "Tourism",
        "Health",
        "Work",
        "Politics",
        "Finance",
    ]

    def __init__(
        self,
        data_path: str | Path,
        output_path: str | Path,
        prompt_dir: str | Path,
        endpoint: EndpointConfig,
        *,
        client: Any | None = None,
        workers: int = 4,
        seed: int = 1234,
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.prompt_dir = Path(prompt_dir)
        self.endpoint = endpoint
        self.client = client or create_openai_client(endpoint)
        self.workers = max(1, workers)
        self.random = random.Random(seed)
        self.original_data = read_json(self.data_path)
        if self.output_path.exists():
            self.state = self._normalize_state(read_json(self.output_path))
        else:
            self.state = self._extract_persona_topics()
            self.save()

    def _prompt(self, name: str, **values: Any) -> str:
        template = read_text(self.prompt_dir / name)
        return template.format(**values)

    def _call_json(self, prompt: str, *, model: str | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.endpoint.retries):
            result = chat_completion(
                self.client,
                self.endpoint,
                [{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                max_tokens=self.endpoint.max_tokens,
            )
            if not result.error:
                text = result.text.strip()
                try:
                    if "```" in text:
                        text = text.replace("```json", "").replace("```", "").strip()
                    value = json.loads(text)
                    if isinstance(value, dict):
                        return value
                    raise ValueError("model returned JSON that is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
            else:
                last_error = RuntimeError(result.error)
        raise RuntimeError(f"Could not obtain structured JSON: {last_error}")

    def _extract_persona_topics(self) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []
        for item in tqdm(self.original_data, desc="Extracting personas"):
            conversation = item.get("conversation", {})
            prompt = self._prompt(
                "extract_persona_topic.txt",
                name_A=str(conversation.get("speaker_a", "User")).strip(),
                name_B=str(conversation.get("speaker_b", "Assistant")).strip(),
                session_summary=json.dumps(item.get("session_summary", {}), ensure_ascii=False),
            )
            try:
                raw = self._call_json(prompt)
            except RuntimeError:
                raw = {}
            output: dict[str, Any] = {}
            for name, details in raw.items():
                if not isinstance(details, dict):
                    continue
                output[str(name)] = {
                    "topics": details.get("topics", []),
                    "profile": details.get("profile", ""),
                    "observation": self._observations(item, str(name)),
                    "tasks": {},
                }
            extracted.append(output)
        return extracted

    @staticmethod
    def _observations(item: dict[str, Any], person_name: str) -> list[str]:
        values: list[str] = []
        observations = item.get("observation", {})
        if not isinstance(observations, dict):
            return values
        for session in observations.values():
            if not isinstance(session, dict):
                continue
            for entry in session.get(person_name, []):
                if isinstance(entry, list) and entry:
                    values.append(str(entry[0]))
                elif isinstance(entry, str):
                    values.append(entry)
        return values

    def _normalize_state(self, values: Any) -> list[dict[str, Any]]:
        if not isinstance(values, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                normalized.append({})
                continue
            people: dict[str, Any] = {}
            for name, data in item.items():
                if not isinstance(data, dict):
                    continue
                people[str(name)] = {
                    "topics": data.get("topics", []),
                    "profile": data.get("profile", ""),
                    "observation": data.get("observation", []),
                    "tasks": data.get("tasks", {}) if isinstance(data.get("tasks", {}), dict) else {},
                }
            normalized.append(people)
        return normalized

    def _task_exists(self, person: dict[str, Any], task: str) -> bool:
        value = person.get("tasks", {}).get(task)
        return isinstance(value, list) and bool(value)

    def _parallel_people(
        self,
        task_name: str,
        builder: Callable[[str, dict[str, Any]], Any],
        overwrite: bool,
    ) -> None:
        jobs: list[tuple[int, str, dict[str, Any]]] = []
        for index, item in enumerate(self.state):
            for name, person in item.items():
                if overwrite or not self._task_exists(person, task_name):
                    jobs.append((index, name, person))

        def run(job: tuple[int, str, dict[str, Any]]) -> tuple[int, str, Any]:
            index, name, person = job
            return index, name, builder(name, person)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc=task_name):
                index, name, value = future.result()
                self.state[index][name].setdefault("tasks", {})[task_name] = value
        self.save()

    def _calculate_unrelated(self, topics: list[str]) -> list[str]:
        prompt = self._prompt(
            "calculate_topic_similarity.txt",
            user_topics_text=", ".join(str(topic) for topic in topics),
            predefined_topics=", ".join(self.PREDEFINED_TOPICS),
        )
        try:
            result = self._call_json(prompt)
            return [topic for topic in self.PREDEFINED_TOPICS if not bool(result.get(topic, False))]
        except RuntimeError:
            return list(self.PREDEFINED_TOPICS)

    def _generate_questions(self, mode: str, count: int, **values: Any) -> list[dict[str, Any]]:
        filenames = {
            "irrelevance": "generate_questions_irrelevance.txt",
            "irrelevance_v2": "generate_questions_irrelevance_v2.txt",
            "diversity": "generate_questions_diversity.txt",
            "sycophancy": "generate_questions_sycophancy.txt",
            "sycophancy_v2": "generate_questions_sycophancy_v2.txt",
        }
        if mode not in filenames:
            raise ValueError(f"Unknown question mode: {mode}")
        values = dict(values)
        values["num_questions"] = count
        if mode == "sycophancy_v2":
            values["user_observations"] = "\n".join(
                f"- {item}" for item in values.get("user_observations", [])
            )
        try:
            result = self._call_json(self._prompt(filenames[mode], **values))
        except RuntimeError:
            return []
        questions = result.get("questions", [])
        return [item for item in questions if isinstance(item, dict) and item.get("question")]

    def build_irrelevance_easy(self, overwrite: bool = False) -> None:
        def build(_name: str, person: dict[str, Any]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for topic in self._calculate_unrelated(person.get("topics", [])):
                result.extend(self._generate_questions("irrelevance", 3, topic=topic))
            return result

        self._parallel_people("irrelevance_easy", build, overwrite)

    def build_irrelevance_hard(self, overwrite: bool = False) -> None:
        def build(_name: str, person: dict[str, Any]) -> list[dict[str, Any]]:
            return self._generate_questions(
                "irrelevance_v2", 5, user_profile=person.get("profile", "")
            )

        self._parallel_people("irrelevance_hard", build, overwrite)

    def build_diversity(self, overwrite: bool = False) -> None:
        def build(_name: str, person: dict[str, Any]) -> list[dict[str, Any]]:
            questions: list[dict[str, Any]] = []
            for topic in person.get("topics", []):
                try:
                    derived = self._call_json(
                        self._prompt("generate_derived_topics.txt", seed=topic, num=3)
                    ).get("derived_topics", [])
                except RuntimeError:
                    derived = [topic]
                for derived_topic in derived:
                    questions.extend(
                        self._generate_questions("diversity", 3, topic=derived_topic)
                    )
            self.random.shuffle(questions)
            return [{"questions": questions}] if questions else []

        self._parallel_people("diversity", build, overwrite)

    def build_sycophancy(self, overwrite: bool = False) -> None:
        def build(_name: str, person: dict[str, Any]) -> list[dict[str, Any]]:
            questions = self._generate_questions(
                "sycophancy", 10, user_profile=person.get("profile", "")
            )
            questions.extend(
                self._generate_questions(
                    "sycophancy_v2", 10, user_observations=person.get("observation", [])
                )
            )
            self.random.shuffle(questions)
            return questions

        self._parallel_people("sycophancy", build, overwrite)

    def build_all(self, overwrite: bool = False) -> None:
        self.build_irrelevance_easy(overwrite)
        self.build_irrelevance_hard(overwrite)
        self.build_diversity(overwrite)
        self.build_sycophancy(overwrite)

    def save(self) -> None:
        write_json(self.output_path, self.state)
