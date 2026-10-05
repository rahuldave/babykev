"""The smoke on the real model: the one test here that is not hermetic.

Every test under `tests/unit/` needs nothing but our code: the tiny model is built in memory, with
random weights. This one loads Qwen2.5-0.5B from the Hugging Face Hub (953 MB the first time, from
the cache after that) onto the best device the machine has. The network, the download and the
device can each make it fail, so it is marked `system` and runs only when asked for:
`just test -m system`. It asserts what is true on any device: the model answers every question.
"""

import pytest

from babykev.smoke import main

pytestmark = pytest.mark.system


def test_the_real_model_answers_every_sample_question(
    capsys: pytest.CaptureFixture[str],  # What the smoke prints
) -> None:
    """Five requests of three questions: fifteen verdicts, a count of those right, exit code 0."""
    assert main([]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith("babykev smoke: Qwen/Qwen2.5-0.5B on ")
    verdicts = [line for line in printed if line.startswith(("  ok", "  --"))]
    assert len(verdicts) == 15
    right = sum(line.startswith("  ok") for line in verdicts)
    assert f"{right} of 15 right" in "\n".join(printed)
