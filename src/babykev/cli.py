"""The babykev command. `babykev help` says what it can do; every command arrives here."""

import sys

HELP = """\
babykev: a small decision model. Context, typed questions, and a probability for every answer.

usage: babykev COMMAND [FLAGS]
       babykev help

There are no commands yet.
"""


def main() -> None:
    """Print the help, or refuse a command that does not exist yet."""
    words = sys.argv[1:]
    if words and words[0] not in ("help", "--help", "-h"):
        print(f"babykev: no command named {words[0]!r}. Run `babykev help`.", file=sys.stderr)
        raise SystemExit(2)
    print(HELP, end="")
