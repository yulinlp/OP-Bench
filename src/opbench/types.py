"""Canonical data structures shared by benchmark generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TASK_TYPES = ("irrelevance_easy", "irrelevance_hard", "sycophancy", "diversity")


@dataclass
class TestCase:
    test_id: str
    task_type: str
    person_name: str
    questions: list[str]
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    """One benchmark group and its per-question generations."""

    test_id: str
    task_type: str
    person_name: str
    responses: list[dict[str, Any]]
    scores: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.scores is None:
            value.pop("scores", None)
        return value


PERSON_TO_CONV_IDX = {
    "Caroline": 0,
    "Jon": 1,
    "John": 2,
    "Joanna": 3,
    "Tim": 4,
    "Audrey": 5,
    "James": 6,
    "Deborah": 7,
    "Evan": 8,
    "Calvin": 9,
}


def _question_text(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        value = item.get("question")
        return str(value).strip() if value else None
    return None


def _questions_for_task(task_type: str, task_data: Any) -> list[str]:
    if not isinstance(task_data, list):
        return []
    if task_type == "diversity":
        items: list[Any] = []
        for group in task_data:
            if isinstance(group, dict):
                items.extend(group.get("questions", []))
        task_data = items
    return [question for item in task_data if (question := _question_text(item))]


def extract_test_cases(
    benchmark_data: list[dict[str, Any]],
    task_types: list[str] | tuple[str, ...] | None = None,
    use_both_personas: bool = False,
) -> list[TestCase]:
    """Normalize the generated benchmark JSON into deterministic test groups."""

    requested = tuple(task_types or TASK_TYPES)
    cases: list[TestCase] = []
    for conversation_item in benchmark_data:
        if not isinstance(conversation_item, dict):
            continue
        people = list(conversation_item.items())
        if not use_both_personas:
            people = people[:1]
        for person_name, person_data in people:
            if not isinstance(person_data, dict):
                continue
            tasks = person_data.get("tasks", {})
            if not isinstance(tasks, dict):
                continue
            for task_type in requested:
                questions = _questions_for_task(task_type, tasks.get(task_type))
                if not questions:
                    continue
                cases.append(
                    TestCase(
                        test_id=f"{person_name}_{task_type}",
                        task_type=task_type,
                        person_name=person_name,
                        questions=questions,
                        metadata={
                            "profile": person_data.get("profile", ""),
                            "topics": person_data.get("topics", []),
                            "observations": person_data.get("observation", []),
                        },
                    )
                )
    return cases


def question_type(
    benchmark_data: list[dict[str, Any]], person_name: str, task_type: str, question: str
) -> str | None:
    """Find the subtype annotation used by the sycophancy judge."""

    for item in benchmark_data:
        if not isinstance(item, dict) or person_name not in item:
            continue
        person = item[person_name]
        tasks = person.get("tasks", {}) if isinstance(person, dict) else {}
        values = tasks.get(task_type, []) if isinstance(tasks, dict) else []
        if task_type == "diversity" and values and isinstance(values[0], dict):
            values = values[0].get("questions", [])
        for value in values:
            if isinstance(value, dict) and value.get("question") == question:
                return value.get("type")
    return None

