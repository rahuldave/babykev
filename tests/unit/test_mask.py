"""Tests of the mask: which token of a packed record may attend to which."""

import torch

from babykev.model import branch_mask


def allowed(
    seg: list[int],  # The segment of each token: 0 for the state, k for question k
) -> torch.Tensor:  # Shape [n, n]: True where token i may attend to token j
    """Read the mask back as yes or no."""
    return branch_mask(seg, "cpu")[0, 0] == 0


def test_mask_of_a_small_request() -> None:
    """A state of two tokens, and two questions of two tokens: the picture, drawn by hand."""
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
