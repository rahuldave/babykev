"""The babykev command. `babykev help` says what it can do; every command arrives here."""

import sys

from babykev import docs

HELP = """\
babykev: a small decision model. Context, typed questions, and a probability for every answer.

usage: babykev COMMAND [FLAGS]
       babykev help

commands:
  docs    what the code says about itself, as Markdown: docs list, docs module NAME, docs check
"""


def main() -> None:
    """Run the command named by the first word, print the help, or refuse any other word."""
    words = sys.argv[1:]
    if not words or words[0] in ("help", "--help", "-h"):
        print(HELP, end="")
        return
    if words[0] == "docs":
        raise SystemExit(docs.main(words[1:]))
    print(f"babykev: no command named {words[0]!r}. Run `babykev help`.", file=sys.stderr)
    raise SystemExit(2)
