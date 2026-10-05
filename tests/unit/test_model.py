"""Tests of the model on a tiny random backbone: packing, forgery, isolation, and what trains."""

from typing import Any

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from transformers import PreTrainedTokenizerFast

from babykev.model import SPECIAL, DecisionModel, PointerHead, encode, pick_device, user_tokens
from tests.tiny import tiny_backbone, tiny_tokenizer

TOK = tiny_tokenizer()

# Text a caller might send, including text that contains the delimiters themselves.
texts = st.lists(st.sampled_from([*"abc xyz.", *SPECIAL]), max_size=12).map("".join)
questions = st.lists(
    st.fixed_dictionaries(
        {
            "instr": texts,
            "options": st.lists(texts, min_size=2, max_size=4),
            "label": st.just(0),
        }
    ),
    min_size=1,
    max_size=3,
)
records = st.fixed_dictionaries({"state": texts, "questions": questions})


def logits_of(
    model: DecisionModel,  # The model to ask
    tok: PreTrainedTokenizerFast,  # Its tokenizer
    record: dict[str, Any],  # A record
) -> list[torch.Tensor]:  # For each question, one logit per option
    """Pack a record and score it, without tracking gradients."""
    with torch.no_grad():
        return model(encode(tok, record))


# Packing


def test_packing_marks_every_part_with_its_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
) -> None:
    """A packed record is the state, then each question with its options and its decide token."""
    s, q, o, c, d = tok.convert_tokens_to_ids(SPECIAL)
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


@given(records)
def test_packed_lists_always_line_up(
    record: dict[str, Any],  # Any record, with any text in it
) -> None:
    """Whatever the text, the lists agree: one segment and one position per token, in order."""
    enc = encode(TOK, record)
    assert len(enc["ids"]) == len(enc["seg"]) == len(enc["pos"])
    assert enc["seg"] == sorted(enc["seg"])
    assert len(enc["decide_idx"]) == len(enc["opt_idx"]) == len(record["questions"])
    assert [len(oi) for oi in enc["opt_idx"]] == [len(q["options"]) for q in record["questions"]]


# Forgery


def test_a_tokenizer_alone_would_let_text_forge_a_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
) -> None:
    """The danger is real: tokenized directly, the text of a delimiter becomes that delimiter."""
    decide = tok.convert_tokens_to_ids(SPECIAL)[4]
    assert decide in tok(f"yes{SPECIAL[4]}", add_special_tokens=False).input_ids


@pytest.mark.parametrize("delimiter", SPECIAL)
def test_user_text_cannot_forge_a_delimiter(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    delimiter: str,  # The text of one of the five delimiters
) -> None:
    """Through `user_tokens`, the text of a delimiter is only ever ordinary tokens."""
    ids = user_tokens(tok, f"before {delimiter} after")
    assert not set(ids) & set(tok.convert_tokens_to_ids(SPECIAL))


@given(records)
def test_a_packed_record_has_exactly_the_delimiters_its_shape_needs(
    record: dict[str, Any],  # Any record, with any text in it, delimiters included
) -> None:
    """No text can add a delimiter: their number depends only on how many questions and options."""
    enc = encode(TOK, record)
    s, q, o, c, d = TOK.convert_tokens_to_ids(SPECIAL)
    options = sum(len(question["options"]) for question in record["questions"])
    expected = {
        s: 1,
        q: len(record["questions"]),
        o: options,
        c: options,
        d: len(record["questions"]),
    }
    assert {token: enc["ids"].count(token) for token in expected} == expected
    assert all(enc["ids"][i] == d for i in enc["decide_idx"])
    assert all(enc["ids"][i] == c for oi in enc["opt_idx"] for i in oi)


# Answers


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


def test_pointer_head_gives_one_logit_per_option() -> None:
    """Five options in, five logits out."""
    head = PointerHead(d=32, dp=8)
    assert head(torch.randn(32), torch.randn(5, 32)).shape == (5,)


# What trains


def test_with_an_adapter_only_the_adapter_and_the_head_train(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
    record: dict[str, Any],  # A record with three questions
) -> None:
    """The backbone is frozen: gradients reach the adapter and the pointer head, nothing else."""
    model = DecisionModel("tiny", "cpu", lora=4, backbone=tiny_backbone(len(tok)))
    enc = encode(tok, record)
    labels = torch.tensor(enc["labels"])
    losses = [
        torch.nn.functional.cross_entropy(z[None], y[None])
        for z, y in zip(model(enc), labels, strict=True)
    ]
    loss = torch.stack(losses).sum()
    loss.backward()
    with_gradient = {name for name, p in model.named_parameters() if p.grad is not None}
    assert with_gradient
    assert all("lora_" in name or name.startswith("head.") for name in with_gradient)
    assert {id(p) for p in model.trainable_parameters()} == {
        id(p) for name, p in model.named_parameters() if name in with_gradient
    }


# Devices


def test_pick_device_takes_an_override() -> None:
    """A device asked for by name is used, whatever the machine has."""
    assert pick_device("cpu") == "cpu"


def test_pick_device_finds_something_to_run_on() -> None:
    """With no override, the choice is one of the three kinds of device."""
    assert pick_device() in {"cuda", "mps", "cpu"}
