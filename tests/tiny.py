"""A tokenizer and a backbone small enough to build in memory, with random weights.

The tokenizer has one token per character. The backbone has the base model's architecture, with
two layers and 32 dimensions. Neither needs a download.
"""

import string

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2Model

from babykev.model import SPECIAL


def tiny_tokenizer() -> PreTrainedTokenizerFast:  # A tokenizer with one token per character
    """Build a real fast tokenizer in memory, with the five delimiters as its special tokens.

    Like the base model's tokenizer, it turns the text of a special token into that token wherever
    it appears, which is the behaviour `user_tokens` has to defend against.
    """
    chars = sorted(set(string.printable))
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    return tok


def tiny_backbone(
    vocab_size: int,  # How many tokens the tokenizer has
    seed: int = 0,  # The seed of the random weights
) -> Qwen2Model:  # A backbone of the base model's architecture, with two small layers
    """Build a randomly initialised backbone, the same every time for the same seed."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return Qwen2Model(config)
