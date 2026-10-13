"""Tests of the model, on the tiny model."""

from typing import Any

import pytest
from transformers import PreTrainedTokenizerFast

from babykev.model import DecisionModel, delimiter_ids, encode


def test_probabilities_sum_to_one_for_every_question(
    model: DecisionModel,  # The tiny model
    tok: PreTrainedTokenizerFast,  # Its tokenizer
    record: dict[str, Any],  # A record with three questions
) -> None:
    """Each question gets one probability per option, and they sum to 1."""
    probs = model.probs(encode(tok, record))
    assert [len(p) for p in probs] == [2, 3, 3]
    for p in probs:
        assert p.sum().item() == pytest.approx(1)


def test_packing_marks_every_part_with_its_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
) -> None:
    """A packed record is the state, then each question with its options and its decide token."""
    s, q, o, c, d = delimiter_ids(tok)
    record = {"state": "ab", "questions": [{"instr": "x?", "options": ["n", "y"], "label": 1}]}
    enc = encode(tok, record)
    a, b, x, mark, n, y = (tok.convert_tokens_to_ids(ch) for ch in "abx?ny")
    assert enc["ids"] == [s, a, b, q, x, mark, o, n, c, o, y, c, d]
    assert enc["seg"] == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert enc["decide_idx"] == [12]
    assert enc["opt_idx"] == [[8, 11]]
    assert enc["labels"] == [1]
