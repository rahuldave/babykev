"""Stand-ins for the parts of the real model, built in memory, with no download."""

import string

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

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
