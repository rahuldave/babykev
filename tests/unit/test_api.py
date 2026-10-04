"""Unit tests for the contract in `babykev.api`: validation, rendering, mappings, confidences."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from babykev.api import (
    MAX_OPTIONS,
    Choice,
    JSONContent,
    Noul,
    Score,
    SystemOneRequest,
    choice_confidence,
    option_text,
    render,
    score_confidence,
    to_answers,
    to_record,
)

SAMPLE = Path(__file__).parents[2] / "data" / "cheese" / "sample.jsonl"
SAMPLE_LINES = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]

MILKS = {"cow": None, "goat": None, "sheep": None, "mixed": "More than one kind of milk"}
FIRMNESS = ["Soft", "Semi-soft", "Semi-hard", "Hard"]


def cheese_request() -> SystemOneRequest:
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
def test_sample_line_fits_the_contract(line: dict[str, Any]) -> None:
    """Every sample line is a valid request and gives one record question per question."""
    request = SystemOneRequest.model_validate(line)
    record, meta = to_record(request)
    assert record["state"]
    assert [m["id"] for m in meta] == list(line["questions"])
    assert all(len(q["options"]) >= 2 for q in record["questions"])


@pytest.mark.parametrize("line", SAMPLE_LINES)
def test_sample_line_has_a_usable_label_on_every_question(line: dict[str, Any]) -> None:
    """A label is a boolean for noul, an option name for choice, a level index for score."""
    for question in line["questions"].values():
        label = question["label"]
        if question["type"] == "noul":
            assert isinstance(label, bool)
        elif question["type"] == "choice":
            assert label in question["criteria"]
        else:
            assert label in range(len(question["criteria"]))


# Validation


def test_request_holds_one_model_per_question_type() -> None:
    """The `type` field decides which model a question becomes."""
    questions = cheese_request().questions
    assert isinstance(questions["blue"], Noul)
    assert isinstance(questions["milk"], Choice)
    assert isinstance(questions["firmness"], Score)


def test_request_without_questions_is_rejected() -> None:
    """A request must ask at least one question."""
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "Brie", "questions": {}})


def test_unknown_question_type_is_rejected() -> None:
    """Only noul, choice and score exist."""
    question = {"type": "essay", "instructions": "Describe this cheese."}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "Brie", "questions": {"q": question}})


def test_noul_needs_no_criteria() -> None:
    """A yes-or-no question may leave out what yes and no mean."""
    assert Noul(type="noul", instructions="Is it blue?").criteria is None


def test_choice_without_options_is_rejected() -> None:
    """A choice needs at least one option."""
    with pytest.raises(ValidationError):
        Choice(type="choice", instructions="Which milk?", criteria={})


def test_choice_accepts_the_largest_number_of_options_and_no_more() -> None:
    """The limit on options is exact."""
    at_limit = {f"option{i}": None for i in range(MAX_OPTIONS)}
    assert (
        len(Choice(type="choice", instructions="Which?", criteria=at_limit).criteria) == MAX_OPTIONS
    )
    with pytest.raises(ValidationError):
        Choice(type="choice", instructions="Which?", criteria={**at_limit, "one_more": None})


def test_score_with_one_level_is_rejected() -> None:
    """A scale needs at least two levels."""
    with pytest.raises(ValidationError):
        Score(type="score", instructions="How firm?", criteria=["Soft"])


# Rendering


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, ""),
        ("Brie", "Brie"),
        (3, "3"),
        (2.5, "2.5"),
        (True, "True"),
        (["creamy", "soft"], "- creamy\n- soft"),
        ({"cheese": "Brie", "age": 4}, "cheese: Brie\nage: 4"),
        (
            {"cheese": "Brie", "notes": {"texture": ["creamy", "soft"]}},
            "cheese: Brie\nnotes:\n  texture:\n    - creamy\n    - soft",
        ),
        ([{"role": "customer", "content": "hello"}], "- role: customer\n  content: hello"),
    ],
)
def test_render_flattens_json_into_text(value: JSONContent, text: str) -> None:
    """Strings, numbers, lists and objects all become the text the model reads."""
    assert render(value) == text


@pytest.mark.parametrize(
    ("name", "description", "text"),
    [
        ("cow", None, "cow"),
        ("cow", "", "cow"),
        ("mixed", "More than one kind of milk", "mixed: More than one kind of milk"),
    ],
)
def test_option_text_adds_a_description_only_when_there_is_one(
    name: str, description: str | None, text: str
) -> None:
    """An option is its name, followed by its description when it has one."""
    assert option_text(name, description) == text


# From a request to a record


def test_record_keeps_the_state_and_the_order_of_questions() -> None:
    """The state is rendered once and the questions stay in request order."""
    record, meta = to_record(cheese_request())
    assert record["state"] == "cheese: Roquefort\norigin: France"
    assert [m["id"] for m in meta] == ["blue", "milk", "firmness"]
    assert [q["instr"] for q in record["questions"]] == [
        "Is this a blue cheese?",
        "Which milk?",
        "How firm?",
    ]


def test_noul_becomes_no_then_yes() -> None:
    """Option 0 is no and option 1 is yes, so the answer is the probability of option 1."""
    record, meta = to_record(cheese_request())
    assert record["questions"][0]["options"] == ["no", "yes"]
    assert meta[0] == {"id": "blue", "type": "noul"}


def test_noul_criteria_describe_no_and_yes() -> None:
    """The criteria under `false` and `true` become the descriptions of no and yes."""
    criteria = {"true": "It has blue veins", "false": "It has none"}
    question = {"type": "noul", "instructions": "Is it blue?", "criteria": criteria}
    request = SystemOneRequest.model_validate({"state": "Brie", "questions": {"blue": question}})
    record, _ = to_record(request)
    assert record["questions"][0]["options"] == ["no: It has none", "yes: It has blue veins"]


def test_choice_becomes_one_option_per_criterion_in_order() -> None:
    """Options keep the order of the criteria, and the metadata remembers their names."""
    record, meta = to_record(cheese_request())
    assert record["questions"][1]["options"] == [
        "cow",
        "goat",
        "sheep",
        "mixed: More than one kind of milk",
    ]
    assert meta[1] == {"id": "milk", "type": "choice", "keys": ["cow", "goat", "sheep", "mixed"]}


def test_score_becomes_one_option_per_level_with_a_legend() -> None:
    """Options are the levels in order, and the legend maps each index to its level."""
    record, meta = to_record(cheese_request())
    assert record["questions"][2]["options"] == FIRMNESS
    assert meta[2]["legend"] == {"0": "Soft", "1": "Semi-soft", "2": "Semi-hard", "3": "Hard"}


# From probabilities to answers


def test_noul_answer_is_the_probability_of_yes() -> None:
    """The answer to a yes-or-no question is one number."""
    answers = to_answers([[0.25, 0.75]], [{"id": "blue", "type": "noul"}])
    assert answers == {"blue": {"type": "noul", "noul": 0.75}}


def test_choice_answer_names_the_most_likely_option() -> None:
    """A choice answer has the winner, a probability for each option, and a confidence."""
    meta = [{"id": "milk", "type": "choice", "keys": ["cow", "goat", "sheep"]}]
    answer = to_answers([[0.1, 0.7, 0.2]], meta)["milk"]
    assert answer["choice"] == "goat"
    assert answer["probabilities"] == {"cow": 0.1, "goat": 0.7, "sheep": 0.2}
    assert answer["confidence"] == pytest.approx(0.55)


def test_score_answer_is_the_expected_level() -> None:
    """A score answer is the mean level under the probabilities, with the legend to read it by."""
    legend = {"0": "Soft", "1": "Semi-soft", "2": "Hard"}
    answer = to_answers([[0.1, 0.2, 0.7]], [{"id": "firmness", "type": "score", "legend": legend}])[
        "firmness"
    ]
    assert answer["score"] == pytest.approx(1.6)
    assert answer["legend"] == legend
    assert answer["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.7}


def test_every_question_gets_an_answer_of_its_own_type() -> None:
    """Records and answers line up: one answer per question, under the question's id."""
    record, meta = to_record(cheese_request())
    uniform = [[1 / len(q["options"])] * len(q["options"]) for q in record["questions"]]
    answers = to_answers(uniform, meta)
    assert {qid: a["type"] for qid, a in answers.items()} == {
        "blue": "noul",
        "milk": "choice",
        "firmness": "score",
    }


def test_rounded_probabilities_still_sum_to_one() -> None:
    """Rounding for output must not lose probability, even with many options."""
    keys = [f"intent{i}" for i in range(77)]
    answer = to_answers([[1 / 77] * 77], [{"id": "intent", "type": "choice", "keys": keys}])[
        "intent"
    ]
    assert sum(answer["probabilities"].values()) == pytest.approx(1, abs=0.02)


# Confidence


def test_choice_confidence_is_zero_for_a_guess() -> None:
    """Equal probabilities are a guess, whatever the number of options."""
    assert choice_confidence([0.25] * 4) == pytest.approx(0)


def test_choice_confidence_is_one_when_certain() -> None:
    """All the probability on one option is certainty."""
    assert choice_confidence([0, 1, 0]) == pytest.approx(1)
    assert choice_confidence([1.0]) == 1.0


def test_choice_confidence_grows_with_the_winner() -> None:
    """Between a guess and certainty, confidence follows the winning probability."""
    assert choice_confidence([0.5, 0.25, 0.25]) == pytest.approx(0.25)


def test_score_confidence_of_a_single_level_is_one() -> None:
    """One level is the whole scale; there is nothing to be unsure about."""
    assert score_confidence([1.0]) == 1.0


@pytest.mark.parametrize("level", [0, 1, 2])
def test_score_confidence_is_one_when_certain(level: int) -> None:
    """All the probability on one level is certainty, wherever the level is."""
    p = [0.0, 0.0, 0.0]
    p[level] = 1.0
    assert score_confidence(p) == pytest.approx(1)


def test_score_confidence_is_zero_for_a_guess() -> None:
    """Equal probabilities over the levels are a guess, as they are for a choice."""
    assert score_confidence([1 / 3] * 3) == pytest.approx(0)


def test_score_confidence_prefers_neighbouring_levels() -> None:
    """Doubt between adjacent levels is more confident than doubt between the two ends."""
    assert score_confidence([0.5, 0.5, 0, 0]) > score_confidence([0.5, 0, 0, 0.5])
