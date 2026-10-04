"""The contract: request and answer shapes, mapped onto the single pointer primitive.

Noul   -> two options, no and yes;         answer = p(yes)
Choice -> one option per criterion;        answer = the likeliest, probabilities by name, confidence
Score  -> one option per level, in order;  answer = expected level, legend, probabilities by level
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255


class Noul(BaseModel):
    """A yes-or-no question."""

    type: Literal["noul"]
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    """A question answered by picking one of several named options."""

    type: Literal["choice"]
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    """A question answered with a level on an ordered scale."""

    type: Literal["score"]
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Noul | Choice | Score


class SystemOneRequest(BaseModel):
    """One request: a state, and the questions to answer about it."""

    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}"
        if isinstance(x, (dict, list))
        else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    """Write one option as the model sees it."""
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def to_record(req: SystemOneRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """-> internal record for encode(), plus per-question metadata to map probabilities back."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            meta.append({"id": qid, "type": "noul"})
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
            meta.append({"id": qid, "type": "choice", "keys": list(q.criteria.keys())})
        else:
            opts = [render(x) for x in q.criteria]
            meta.append(
                {
                    "id": qid,
                    "type": "score",
                    "legend": {str(i): render(x) for i, x in enumerate(q.criteria)},
                }
            )
        qs.append({"instr": instr, "options": opts, "label": 0})
    return {"state": render(req.state), "questions": qs}, meta


def choice_confidence(p: list[float]) -> float:
    """Measure how far the most likely option stands above a guess among `len(p)` options."""
    k = len(p)
    return 1.0 if k == 1 else (max(p) - 1 / k) / (1 - 1 / k)


def score_confidence(p: list[float]) -> float:
    """Measure how closely a score's probabilities sit around the most likely level.

    1 when one level has all the probability, 0 for a guess or anything as spread out. The expected
    distance from the most likely level is divided by the distance a guess would have: equal
    probability on every level, measured from the middle of the scale.
    """
    n = len(p)
    if n == 1:
        return 1.0
    mode = max(range(n), key=lambda i: p[i])
    spread = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    guess = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - spread / guess)


def round_prob(x: float) -> float:
    """Round a number for output.

    Four decimals keep a rounded distribution's sum within 0.02 of 1 even with MAX_OPTIONS options.
    Two decimals did not: 77 equal options summed to 0.77.
    """
    return round(float(x), 4)


def to_answers(probs: list[list[float]], meta: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn the model's probabilities into answers of each question's type."""
    out = {}
    for p, m in zip(probs, meta, strict=False):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1])}
        elif m["type"] == "choice":
            dist = {k: round_prob(v) for k, v in zip(m["keys"], p, strict=False)}
            out[m["id"]] = {
                "type": "choice",
                "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                "confidence": round_prob(choice_confidence(p)),
                "probabilities": dist,
            }
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {
                "type": "score",
                "score": round_prob(score),
                "legend": m["legend"],
                "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                "confidence": round_prob(score_confidence(p)),
            }
    return out
