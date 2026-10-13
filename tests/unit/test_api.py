"""Unit tests of the contract in `babykev.api`: validation, rendering, mappings, confidences."""

import json
from pathlib import Path

import pytest

from babykev.api import SystemOneRequest, to_record

SAMPLE = Path(__file__).parents[2] / "data" / "cheese" / "sample.jsonl"
SAMPLE_LINES = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines()]

MILKS = {"cow": None, "goat": None, "sheep": None, "mixed": "More than one kind of milk"}
FIRMNESS = ["Soft", "Semi-soft", "Semi-hard", "Hard"]


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
