from opbench.scoring import _parse_score


def test_parse_score_clamps_and_ignores_surrounding_text():
    assert _parse_score("Output: 0.85") == 0.85
    assert _parse_score("score = 1.7") == 1.0

