"""Unit tests of the contract in `babykev.api`: validation, rendering, mappings, confidences."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from babykev.api import (
    MAX_OPTIONS,
    Choice,
    Noul,
    Score,
    SystemOneRequest,
    to_record,
)

SAMPLE = Path(__file__).parents[2] / "data" / "cheese" / "sample.jsonl"
SAMPLE_LINES = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]

MILKS = {"cow": None, "goat": None, "sheep": None, "mixed": "More than one kind of milk"}
FIRMNESS = ["Soft", "Semi-soft", "Semi-hard", "Hard"]


def cheese_request():
    """One request that asks a question of each type."""
    return SystemOneRequest.model_validate(
        {
            "state": {"cheese": "Roquefort", "origin": "France"},
            "questions": {
                "blue": {"type": "noul", "instructions": "Is this a blue cheese?"},
                "milk": {"type": "choice", "instructions": "Which milk?", "criteria": MILKS},
                "firmness": {"type": "score", "instructions": "How firm?", "criteria": FIRMNESS},
            },
        }
    )


# The sample data


@pytest.mark.parametrize("line", SAMPLE_LINES)
def test_sample_line_fits_the_contract(line):
    """Every line of the sample data is a valid request, with one record question per question."""
    request = SystemOneRequest.model_validate(line)
    record, meta = to_record(request)
    assert record["state"]
    assert [m["id"] for m in meta] == list(line["questions"])
    assert all(len(q["options"]) >= 2 for q in record["questions"])


@pytest.mark.parametrize("line", SAMPLE_LINES)
def test_sample_line_has_a_usable_label_on_every_question(line):
    """A label is a boolean for noul, an option name for choice, and a level index for score."""
    for question in line["questions"].values():
        label = question["label"]
        if question["type"] == "noul":
            assert isinstance(label, bool)
        elif question["type"] == "choice":
            assert label in question["criteria"]
        else:
            assert label in range(len(question["criteria"]))


# Validation


def test_request_holds_one_model_per_question_type():
    """The `type` field decides which model a question becomes."""
    questions = cheese_request().questions
    assert isinstance(questions["blue"], Noul)
    assert isinstance(questions["milk"], Choice)
    assert isinstance(questions["firmness"], Score)


def test_request_without_questions_is_rejected():
    """A request must ask at least one question."""
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "Brie", "questions": {}})


def test_unknown_question_type_is_rejected():
    """Only noul, choice and score exist."""
    question = {"type": "essay", "instructions": "Describe this cheese."}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "Brie", "questions": {"q": question}})


def test_noul_needs_no_criteria():
    """A yes-or-no question may leave out what yes and no mean."""
    assert Noul(type="noul", instructions="Is it blue?").criteria is None


def test_choice_without_options_is_rejected():
    """A choice needs at least one option."""
    with pytest.raises(ValidationError):
        Choice(type="choice", instructions="Which milk?", criteria={})


def test_choice_accepts_the_largest_number_of_options_and_no_more():
    """The limit on options is exact."""
    at_limit = {f"option{i}": None for i in range(MAX_OPTIONS)}
    assert (
        len(Choice(type="choice", instructions="Which?", criteria=at_limit).criteria) == MAX_OPTIONS
    )
    with pytest.raises(ValidationError):
        Choice(type="choice", instructions="Which?", criteria={**at_limit, "one_more": None})
