"""The fixtures that every test file can ask for.

What the tests of the model check is exact, and holds for any weights. So the fixtures need no
trained model, no download and no GPU.
"""

import pytest
from transformers import PreTrainedTokenizerFast

from tests.tiny import tiny_tokenizer


@pytest.fixture(scope="session")
def tok() -> PreTrainedTokenizerFast:  # The tiny tokenizer, built once for the whole test run
    """Provide the tiny tokenizer."""
    return tiny_tokenizer()
