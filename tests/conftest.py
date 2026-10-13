"""The fixtures that every test file can ask for.

What the tests of the model check is exact, and holds for any weights. So the fixtures need no
trained model, no download and no GPU.
"""

import pytest
from transformers import PreTrainedTokenizerFast

from babykev.model import DecisionModel
from tests.tiny import tiny_backbone, tiny_tokenizer


@pytest.fixture(scope="session")
def tok() -> PreTrainedTokenizerFast:  # The tiny tokenizer, built once for the whole test run
    """Provide the tiny tokenizer."""
    return tiny_tokenizer()


@pytest.fixture
def model(
    tok: PreTrainedTokenizerFast,  # The tiny tokenizer
) -> DecisionModel:  # A decision model on the tiny backbone, with no adapter, ready to answer
    """Provide a fresh tiny model for each test, on the CPU."""
    return DecisionModel("tiny", "cpu", backbone=tiny_backbone(len(tok))).eval()
