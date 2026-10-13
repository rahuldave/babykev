"""A smoke run: everything that exists so far, end to end, on the real model.

It loads the base model with a fresh adapter and an untrained pointer head, asks it the questions
in the sample data, and prints each answer beside its label. Nothing has been trained yet, so the
answers are guesses. What the run shows is that the pieces fit: a line of data passes the contract,
is packed into one sequence, goes through the model on whatever device there is, and comes back as
answers.
"""

import json
from pathlib import Path
from typing import Any

SAMPLE = Path("data/cheese/sample.jsonl")


def read_lines(
    path: Path = SAMPLE,  # A JSONL file of labelled requests
) -> list[dict[str, Any]]:  # One labelled request per line of the file
    """Read labelled requests from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
