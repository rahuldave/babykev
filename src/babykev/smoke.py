"""A smoke run: everything that exists so far, end to end, on the real model.

It loads the base model with a fresh adapter and an untrained pointer head, asks it the questions
in the sample data, and prints each answer beside its label. Nothing has been trained yet, so the
answers are guesses. What the run shows is that the pieces fit: a line of data passes the contract,
is packed into one sequence, goes through the model on whatever device there is, and comes back as
answers.
"""

import json
import time
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from babykev.api import SystemOneRequest, to_answers, to_record
from babykev.model import DecisionModel, encode

SAMPLE = Path("data/cheese/sample.jsonl")


def read_lines(
    path: Path = SAMPLE,  # A JSONL file of labelled requests
) -> list[dict[str, Any]]:  # One labelled request per line of the file
    """Read labelled requests from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def answer(
    model: DecisionModel,  # The model that answers
    tok: PreTrainedTokenizerBase,  # Its tokenizer
    line: dict[str, Any],  # One labelled request
) -> tuple[dict[str, Any], int, float]:  # The answers, the tokens packed, the seconds taken
    """Answer one request: the contract, then packing, then one forward pass, then answers."""
    record, meta = to_record(SystemOneRequest.model_validate(line))
    enc = encode(tok, record)
    started = time.perf_counter()
    probs = [p.tolist() for p in model.probs(enc)]
    return to_answers(probs, meta), len(enc["ids"]), time.perf_counter() - started


def verdict(
    question: dict[str, Any],  # One question of a labelled request, with its `label`
    given: dict[str, Any],  # The model's answer to it
) -> tuple[str, str, bool]:  # What the model said, what the label says, and whether they agree
    """Put one answer beside its label."""
    label = question["label"]
    if given["type"] == "noul":
        said = given["noul"] >= 0.5
        return (
            f"{'yes' if said else 'no'} ({given['noul']:.2f})",
            "yes" if label else "no",
            said == label,
        )
    if given["type"] == "choice":
        chosen = given["choice"]
        return f"{chosen} ({given['probabilities'][chosen]:.2f})", str(label), chosen == label
    level = max(given["probabilities"], key=lambda i: given["probabilities"][i])
    return f"level {level} (score {given['score']:.2f})", f"level {label}", int(level) == label
