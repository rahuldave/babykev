"""The smoke on the real model, the one test here that is not hermetic.

The tests under `tests/unit/` need nothing but our code: the tiny model is built in memory. This
test loads Qwen2.5-0.5B from the Hugging Face Hub, about 1 GB the first time and from the cache
after that, onto the best device of the machine. The network, the download and the device can
each make it fail.
"""

import pytest

from babykev.smoke import main

# Every test of this file needs the network, the real weights and a device
pytestmark = pytest.mark.system


def test_the_real_model_answers_every_sample_question(
    capsys: pytest.CaptureFixture[str],  # What the smoke prints
) -> None:
    """Five requests of three questions: fifteen verdicts, the count of those right, exit 0."""
    assert main([]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("babykev smoke: Qwen/Qwen2.5-0.5B on ")
    verdicts = [line for line in printed if line.startswith(("  ok", "  --"))]
    assert len(verdicts) == 15
    right = sum(line.startswith("  ok") for line in verdicts)
    assert f"{right} of 15 right" in "\n".join(printed)
