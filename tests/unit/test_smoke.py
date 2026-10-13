"""Tests of the smoke run, without the real model."""

from babykev.smoke import verdict


def test_verdict_agrees_when_the_answer_is_the_label() -> None:
    """A verdict says what the model said, what the label says, and whether they agree."""
    question = {"type": "choice", "criteria": {"cow": None, "goat": None}, "label": "goat"}
    given = {"type": "choice", "choice": "goat", "probabilities": {"cow": 0.2, "goat": 0.8}}
    assert verdict(question, given) == ("goat (0.80)", "goat", True)
