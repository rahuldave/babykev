"""Tests of the model, on the tiny model."""

from typing import Any

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from babykev.model import (
    SPECIAL,
    DecisionModel,
    PointerHead,
    delimiter_ids,
    encode,
    pick_device,
    user_tokens,
)


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


def test_positions_restart_after_the_state(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    record: dict[str, Any],  # A record with three questions
) -> None:
    """Every question starts at the position right after the state, wherever it comes."""
    enc = encode(tok, record)
    state_length = enc["seg"].count(0)
    for k in (1, 2, 3):
        positions = [p for p, s in zip(enc["pos"], enc["seg"], strict=True) if s == k]
        assert positions == list(range(state_length, state_length + len(positions)))


def test_a_question_that_does_not_fit_is_refused(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    record: dict[str, Any],  # A record with three questions
) -> None:
    """A branch longer than the limit raises, and names its length."""
    with pytest.raises(ValueError, match="branch too long"):
        encode(tok, record, max_branch=80)


def test_a_state_that_does_not_fit_is_refused(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    record: dict[str, Any],  # A record with three questions
) -> None:
    """A state longer than the limit raises, and names its length and the limit."""
    with pytest.raises(ValueError, match="state too long: 62 tokens, the limit is 50"):
        encode(tok, record, max_state=50)


def test_a_tokenizer_alone_would_let_text_forge_a_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
) -> None:
    """The danger is real: tokenized directly, the text of a delimiter becomes that delimiter."""
    decide = delimiter_ids(tok)[4]
    assert decide in tok(f"yes{SPECIAL[4]}", add_special_tokens=False).input_ids


@pytest.mark.parametrize("delimiter", SPECIAL)
def test_user_text_cannot_forge_a_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    delimiter: str,  # The text of one of the five delimiters
) -> None:
    """Through `user_tokens`, the text of a delimiter is only ever ordinary tokens."""
    ids = user_tokens(tok, f"before {delimiter} after")
    assert not set(ids) & set(delimiter_ids(tok))


def test_pointer_head_gives_one_logit_per_option() -> None:
    """Five options in, five logits out."""
    head = PointerHead(d=32, dp=8)
    assert head(torch.randn(32), torch.randn(5, 32)).shape == (5,)


def test_pick_device_takes_an_override() -> None:
    """A device asked for by name is used, whatever the machine has."""
    assert pick_device("cpu") == "cpu"
