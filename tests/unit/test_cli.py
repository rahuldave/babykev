"""Unit tests for the `babykev` command: `help` prints the help, any other word is refused."""

import pytest

from babykev.cli import HELP, main


def run(monkeypatch, *words):
    """Run `babykev` with these words as its command line."""
    monkeypatch.setattr("sys.argv", ["babykev", *words])
    main()


@pytest.mark.parametrize("words", [(), ("help",), ("--help",), ("-h",)])
def test_help_is_printed_for_help_and_for_no_words(monkeypatch, capsys, words):
    """`babykev`, `babykev help`, `--help` and `-h` all print the help text, and nothing else."""
    run(monkeypatch, *words)
    assert capsys.readouterr().out == HELP
