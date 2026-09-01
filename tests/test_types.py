import json
from pathlib import Path

from opbench.types import extract_test_cases


def test_checked_in_benchmark_has_four_task_groups():
    data = json.loads(
        (Path(__file__).parents[1] / "data" / "locomo10_overpersonalized.json").read_text()
    )
    cases = extract_test_cases(data)
    assert len(cases) == 40
    assert {case.task_type for case in cases} == {
        "irrelevance_easy",
        "irrelevance_hard",
        "sycophancy",
        "diversity",
    }

