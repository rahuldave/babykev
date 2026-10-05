"""The smoke run, on the tiny model: it must report on every question of the sample data."""

from transformers import PreTrainedTokenizerFast

from babykev.model import DecisionModel
from babykev.smoke import read_lines, smoke, verdict


def test_smoke_reports_on_every_question_of_the_sample(
    model: DecisionModel,  # The tiny model
    tok: PreTrainedTokenizerFast,  # Its tokenizer
) -> None:
    """Five requests of three questions: fifteen verdicts and a closing count."""
    lines = read_lines()
    report = list(smoke(model, tok, lines))
    verdicts = [text for text in report if text.startswith(("  ok", "  --"))]
    assert len(verdicts) == sum(len(line["questions"]) for line in lines) == 15
    assert any(
        text.strip().startswith(f"{sum('  ok' in v for v in verdicts)} of 15 right")
        for text in report
    )


def test_verdict_agrees_when_the_answer_is_the_label() -> None:
    """A verdict says what the model said, what the label says, and whether they agree."""
    question = {"type": "choice", "criteria": {"cow": None, "goat": None}, "label": "goat"}
    given = {"type": "choice", "choice": "goat", "probabilities": {"cow": 0.2, "goat": 0.8}}
    assert verdict(question, given) == ("goat (0.80)", "goat", True)


def test_verdict_reads_a_noul_answer_as_yes_from_one_half() -> None:
    """A probability of yes at or above 0.5 is a yes."""
    said, label, agree = verdict({"type": "noul", "label": False}, {"type": "noul", "noul": 0.5})
    assert (said, label, agree) == ("yes (0.50)", "no", False)


def test_verdict_reads_a_score_answer_as_its_most_likely_level() -> None:
    """A score is judged by the level with the highest probability."""
    given = {"type": "score", "score": 1.1, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}
    assert verdict({"type": "score", "label": 1}, given) == (
        "level 1 (score 1.10)",
        "level 1",
        True,
    )
