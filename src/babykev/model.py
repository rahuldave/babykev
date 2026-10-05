"""The model: a language model that reads a packed request, and a head that points at options.

Four ideas make a language model into a decision model.

1. **Packing.** `encode` writes the state and every question into one sequence of tokens. Five
   delimiter tokens mark the parts: the state, a question, the start and the end of an option, and
   the place where the question is decided.
2. **The mask.** `branch_mask` lets each token see the state and its own question, and never another
   question. The questions share one reading of the state and cannot influence each other.
3. **Positions restart.** Each question's positions begin right after the state, so a question looks
   the same to the model wherever it comes in the request.
4. **Pointing.** `PointerHead` scores each option's closing token against the question's decide
   token. The model has no vocabulary head and generates nothing.

`DecisionModel` puts them together around a backbone, with a LoRA adapter if asked for.
"""

import math
import re
from typing import Any, TypedDict

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# The delimiters are special tokens the base model already has and rarely uses, in this order:
# state, question, option start, option end, decide. No embedding has to be added or trained for
# them; the adapter learns their new meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Encoded(TypedDict):
    """One packed record, ready for the model."""

    ids: list[int]  # The token ids of the whole sequence
    seg: list[int]  # The segment of each token: 0 for the state, k for question k
    pos: list[int]  # The position of each token; every question restarts right after the state
    decide_idx: list[int]  # For each question, where its decide token is
    opt_idx: list[list[int]]  # For each question, where the closing token of each option is
    labels: list[int]  # For each question, the index of the right option


def pick_device(
    override: str | None = None,  # A device name to use whatever the machine has
) -> str:  # `cuda`, `mps` or `cpu`
    """Choose where the model runs: a CUDA GPU if there is one, then Apple's GPU, then the CPU."""
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer(
    name: str,  # A model name on the Hugging Face Hub, or a local folder
) -> PreTrainedTokenizerBase:  # Its tokenizer
    """Load the tokenizer that belongs to a base model."""
    return AutoTokenizer.from_pretrained(name)


def user_tokens(
    tok: PreTrainedTokenizerBase,  # The base model's tokenizer
    text: str,  # Text that came from a caller: a state, an instruction or an option
) -> list[int]:  # Its token ids, none of which is a delimiter or any other special token
    """Tokenize a caller's text so that it can never forge a delimiter.

    A tokenizer turns the string `<|box_end|>` into the special token of that name, wherever the
    string appears. Text that contained it could close an option early. So `<|name|>` is rewritten
    to `<¦name¦>` before tokenizing, which reads the same to a person and is ordinary text to the
    tokenizer.
    """
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(
    tok: PreTrainedTokenizerBase,  # The base model's tokenizer
    rec: dict[str, Any],  # A record, as `to_record` returns it: a state and its questions
    max_state: int = MAX_STATE,  # The longest state kept, in tokens
    max_branch: int = MAX_BRANCH,  # The longest state and question together, in tokens
) -> Encoded:  # The packed record
    """Pack a record into one sequence: the state, then one branch for each question.

    The state is the state delimiter and its text. A branch is the question delimiter, the
    instruction, each option between its two delimiters, and the decide delimiter. Raises
    `ValueError` when a branch does not fit in `max_branch`.
    """
    s_id, q_id, o_id, c_id, d_id = tok.convert_tokens_to_ids(SPECIAL)
    state = [s_id] + user_tokens(tok, rec["state"])[: max_state - 1]
    ids, seg, pos = list(state), [0] * len(state), list(range(len(state)))
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        br = [q_id] + user_tokens(tok, q["instr"])
        oi = []
        for o in q["options"]:
            br += [o_id] + user_tokens(tok, o) + [c_id]
            oi.append(len(br) - 1)
        br.append(d_id)
        if len(br) > max_branch - len(state):
            raise ValueError(f"branch too long: {len(br)}")
        base = len(ids)
        ids += br
        seg += [k] * len(br)
        pos += list(range(len(state), len(state) + len(br)))
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + i for i in oi])
    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": [q["label"] for q in rec["questions"]],
    }


def branch_mask(
    seg: list[int],  # The segment of each token: 0 for the state, k for question k
    device: str,  # Where to build the mask
    dtype: torch.dtype = torch.float32,  # The number type of the model's attention scores
) -> torch.Tensor:  # Shape [1, 1, n, n]: 0 where i may attend to j, very negative elsewhere
    """Build the block-causal mask: a token sees the state and its own question, never a sibling.

    Token i may attend to token j when j is not after i, and j is in the state or in the same
    question as i. The mask is added to the attention scores, so a blocked pair gets the most
    negative number the type can hold.
    """
    s = torch.tensor(seg, device=device)
    n = len(seg)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    same = torch.ones_like(causal) | (s[None, :] == 0)
    allow = causal & same
    blocked = torch.finfo(dtype).min
    return torch.zeros(n, n, dtype=dtype, device=device).masked_fill(~allow, blocked)[None, None]


class PointerHead(nn.Module):
    """Scores the options of one question against its decide token."""

    def __init__(
        self,
        d: int,  # The width of the backbone's hidden states
        dp: int = 256,  # The width of the space in which options and decide tokens are compared
    ) -> None:
        """Make the two projections, one for decide tokens and one for options."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(
        self,
        h_decide: torch.Tensor,  # The hidden state at the decide token, shape [d]
        h_opts: torch.Tensor,  # The hidden states at the K options' closing tokens, shape [K, d]
    ) -> torch.Tensor:  # One logit per option, shape [K]
        """Project both sides and take their scaled dot products."""
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    """A language model without its vocabulary head, read through a mask, with a `PointerHead`."""

    def __init__(
        self,
        name: str,  # The base model to load from the Hugging Face Hub, when no backbone is given
        device: str,  # Where the model runs: `cuda`, `mps` or `cpu`
        lora: int | None = None,  # The rank of a LoRA adapter to add, or `None` for no adapter
        backbone: PreTrainedModel | None = None,  # A ready backbone, such as a small one in tests
    ) -> None:
        """Load or take the backbone, wrap it in an adapter if asked, and add the head."""
        super().__init__()
        if backbone is None:
            # Only the transformer is kept. The vocabulary head is dropped: nothing is generated.
            lm = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float32, attn_implementation="eager"
            )
            backbone = lm.model
        width: int = backbone.config.hidden_size
        self.lm: nn.Module = backbone
        if lora:
            cfg = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=lora,
                lora_alpha=2 * lora,
                lora_dropout=0.05,
                target_modules=LORA_TARGETS,
            )
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(width)
        self.device = device
        self.to(device)

    def hidden(
        self,
        enc: Encoded,  # A packed record
    ) -> torch.Tensor:  # The hidden state of every token, shape [n, d]
        """Run the backbone over a packed record, with its restarted positions and its mask."""
        ids = torch.tensor([enc["ids"]], device=self.device)
        pos = torch.tensor([enc["pos"]], device=self.device)
        mask = branch_mask(enc["seg"], self.device)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state[0]

    def forward(
        self,
        enc: Encoded,  # A packed record
    ) -> list[torch.Tensor]:  # For each question, one logit per option
        """Score every option of every question in one pass."""
        h = self.hidden(enc)
        return [
            self.head(h[d], h[torch.tensor(oi, device=self.device)])
            for d, oi in zip(enc["decide_idx"], enc["opt_idx"], strict=True)
        ]

    @torch.no_grad()
    def probs(
        self,
        enc: Encoded,  # A packed record
    ) -> list[torch.Tensor]:  # For each question, the probability of each option, on the CPU
        """Answer without tracking gradients: the logits of `forward` turned into probabilities."""
        return [torch.softmax(z, -1).cpu() for z in self.forward(enc)]

    def trainable_parameters(self) -> list[nn.Parameter]:  # The adapter's and the head's
        """List the parameters that training changes."""
        return [p for p in self.parameters() if p.requires_grad]
