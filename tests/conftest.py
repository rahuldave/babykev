"""Shared fixtures: the tiny tokenizer, a tiny model and a record.

The properties the tests check (the mask, the packing, the isolation of questions) are exact. They
hold for any weights, so they need no trained model, no download and no GPU.
"""

from typing import Any

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


# hypothesis runs a test again and again, and by default allows each run 200 ms. The first run
# that uses torch can take longer than that on a slow or busy machine, for example inside a
# container, and the test then fails by chance: `DeadlineExceeded`, reported as `FlakyFailure`.
# The tests that hypothesis feeds check a rule and not a speed, so the deadline is off for the
# whole run. The import sits here, below the fixtures, so that they keep their places in the file.
from hypothesis import settings  # noqa: E402

settings.register_profile("babykev", deadline=None)
settings.load_profile("babykev")
