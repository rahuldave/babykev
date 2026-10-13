"""The fixtures that every test file can ask for.

What the tests of the model check is exact, and holds for any weights. So the fixtures need no
trained model, no download and no GPU.
"""

from typing import Any

import pytest
from hypothesis import settings
from transformers import PreTrainedTokenizerFast

from babykev.model import DecisionModel
from tests.tiny import tiny_backbone, tiny_tokenizer

# hypothesis gives each example of a test 200 ms. The first example that runs torch can take
# longer on a slow or busy machine, such as a container, and the test then fails by chance. The
# tests that hypothesis feeds check a rule, and not a speed, so no example has a deadline
settings.register_profile("babykev", deadline=None)
settings.load_profile("babykev")


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


@pytest.fixture
def record() -> dict[str, Any]:  # A record with a state and three questions
    """Provide one record, as `to_record` would return it."""
    return {
        "state": "Roquefort. A blue cheese from France, made from sheep's milk.",
        "questions": [
            {"instr": "Is this a blue cheese?", "options": ["no", "yes"], "label": 1},
            {"instr": "Which milk?", "options": ["cow", "goat", "sheep"], "label": 2},
            {"instr": "How firm?", "options": ["Soft", "Semi-soft", "Hard"], "label": 1},
        ],
    }
