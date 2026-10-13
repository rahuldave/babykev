"""Tests of the smoke run, without the real model."""

from babykev.smoke import verdict


def test_verdict_agrees_when_the_answer_is_the_label() -> None:
    """A verdict says what the model said, what the label says, and whether they agree."""
    question = {"type": "choice", "criteria": {"cow": None, "goat": None}, "label": "goat"}
    given = {"type": "choice", "choice": "goat", "probabilities": {"cow": 0.2, "goat": 0.8}}
    assert verdict(question, given) == ("goat (0.80)", "goat", True)


def test_verdict_reads_a_noul_answer_as_yes_from_one_half() -> None:
    """A probability of yes at or above 0.5 is a yes."""
    said, label, agree = verdict({"type": "noul", "label": False}, {"type": "noul", "noul": 0.5})
    assert (said, label, agree) == ("yes (0.50)", "no", False)
