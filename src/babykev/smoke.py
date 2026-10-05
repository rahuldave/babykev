"""A smoke run: everything that exists so far, end to end, on the real model.

    babykev smoke

It loads the base model with a fresh adapter and an untrained pointer head, asks it the questions
in the sample data, and prints each answer beside its label. Nothing has been trained yet, so the
answers are guesses. What the run shows is that the pieces fit: a line of data passes the contract,
is packed into one sequence, goes through the model on whatever device there is, and comes back as
answers.
"""

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from babykev.api import SystemOneRequest, to_answers, to_record
from babykev.model import DecisionModel, encode, load_tokenizer, pick_device

BASE = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
SAMPLE = Path("data/cheese/sample.jsonl")
USAGE = """\
usage: babykev smoke    load the base model on this machine; answer the sample requests, untrained
"""


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


def smoke(
    model: DecisionModel,  # The model to try
    tok: PreTrainedTokenizerBase,  # Its tokenizer
    lines: list[dict[str, Any]],  # Labelled requests to ask it
) -> Iterator[str]:  # The report, a line at a time
    """Ask the model every question in `lines` and report each answer beside its label."""
    answer(model, tok, lines[0])  # the first pass on a device is slow, so it is not timed
    right, total, seconds = 0, 0, []
    for line in lines:
        answers, tokens, took = answer(model, tok, line)
        seconds.append(took)
        first = SystemOneRequest.model_validate(line)
        state = to_record(first)[0]["state"].splitlines()[0]
        yield f"\n{state[:70]}   [{tokens} tokens, {took * 1000:.0f} ms]"
        for qid, question in line["questions"].items():
            said, label, agree = verdict(question, answers[qid])
            right, total = right + agree, total + 1
            mark = "ok" if agree else "--"
            yield f"  {mark}  {qid:12s} {question['type']:7s} says {said:28s} label: {label}"
    typical = sorted(seconds)[len(seconds) // 2] * 1000
    yield f"\n{right} of {total} right, {typical:.0f} ms a request."
    yield "Nothing is trained yet: these are guesses."


def main(
    words: list[str],  # The words after `babykev smoke`: none to run, `help` for the usage
) -> int:  # The exit code: 0, or 2 for words that are no command
    """Load the real model on the best device there is and run the smoke on the sample data."""
    if words in (["help"], ["--help"], ["-h"]):
        print(USAGE, end="")
        return 0
    if words:
        print(USAGE, end="", file=sys.stderr)
        return 2
    device = pick_device()
    where = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"babykev smoke: {BASE} on {where}", flush=True)
    started = time.perf_counter()
    torch.manual_seed(0)
    tok = load_tokenizer(BASE)
    model = DecisionModel(BASE, device, lora=LORA_RANK).eval()
    everything = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.trainable_parameters())
    loaded = time.perf_counter() - started
    sizes = f"{everything / 1e6:.0f}M parameters, {trainable / 1e6:.1f}M of them trainable"
    print(f"loaded in {loaded:.1f} s: {sizes}", flush=True)
    for text in smoke(model, tok, read_lines()):
        print(text, flush=True)
    return 0
