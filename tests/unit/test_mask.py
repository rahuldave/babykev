"""Tests of the block-causal mask. They state its rule as properties that hold for any request."""

import torch
from hypothesis import given
from hypothesis import strategies as st

from babykev.model import branch_mask

# A request of any shape: a state of 1 to 6 tokens, then up to 4 questions of 1 to 5 tokens each.
shapes = st.tuples(st.integers(1, 6), st.lists(st.integers(1, 5), max_size=4))


def segments(
    state: int,  # How many tokens the state has
    questions: list[int],  # How many tokens each question has
) -> list[int]:  # The segment of each token: 0 for the state, k for question k
    """Lay out a request as `encode` does: the state, then each question in turn."""
    return [0] * state + [k for k, n in enumerate(questions, start=1) for _ in range(n)]


def allowed(
    seg: list[int],  # The segment of each token
) -> torch.Tensor:  # Shape [n, n]: True where token i may attend to token j
    """Read the mask back as yes or no."""
    return branch_mask(seg, "cpu")[0, 0] == 0


def test_mask_of_a_small_request() -> None:
    """Two state tokens and two questions of two tokens: the picture to keep in mind."""
    picture = [
        "#.....",
        "##....",
        "###...",
        "####..",
        "##..#.",
        "##..##",
    ]
    rows = ["".join("#" if ok else "." for ok in row) for row in allowed([0, 0, 1, 1, 2, 2])]
    assert rows == picture


def test_mask_has_a_shape_the_model_can_add_to_its_attention() -> None:
    """One batch, one head, a square of the sequence length; 0 allows, very negative blocks."""
    mask = branch_mask([0, 0, 1, 2], "cpu")
    assert mask.shape == (1, 1, 4, 4)
    assert set(mask.unique().tolist()) == {0.0, torch.finfo(torch.float32).min}


@given(shapes)
def test_no_token_sees_a_later_token(
    shape: tuple[int, list[int]],  # The lengths of a state and of its questions
) -> None:
    """The mask is causal: nothing above the diagonal is allowed."""
    allow = allowed(segments(*shape))
    assert not torch.triu(allow, diagonal=1).any()


@given(shapes)
def test_no_question_sees_another_question(
    shape: tuple[int, list[int]],  # The lengths of a state and of its questions
) -> None:
    """A token of one question never attends to a token of another."""
    seg = torch.tensor(segments(*shape))
    siblings = (seg[:, None] != seg[None, :]) & (seg[:, None] > 0) & (seg[None, :] > 0)
    assert not (allowed(seg.tolist()) & siblings).any()


@given(shapes)
def test_every_token_sees_the_state_before_it_and_its_own_question(
    shape: tuple[int, list[int]],  # The lengths of a state and of its questions
) -> None:
    """What is allowed is exactly: not later, and in the state or in the same question."""
    seg = segments(*shape)
    rule = [
        [j <= i and (seg[j] == 0 or seg[j] == seg[i]) for j in range(len(seg))]
        for i in range(len(seg))
    ]
    assert allowed(seg).tolist() == rule
