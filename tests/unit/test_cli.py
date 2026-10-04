"""Unit tests for the `babykev` command: `help`, `docs`, and the refusal of any other word."""

import pytest

from babykev.cli import HELP, main


def run(
    monkeypatch: pytest.MonkeyPatch,  # Replaces `sys.argv` for this test only
    *words: str,  # The command line, as words
) -> None:
    """Run `babykev` with these words as its command line."""
    monkeypatch.setattr("sys.argv", ["babykev", *words])
    main()


@pytest.mark.parametrize("words", [(), ("help",), ("--help",), ("-h",)])
def test_help_is_printed_for_help_and_for_no_words(
    monkeypatch: pytest.MonkeyPatch,  # Replaces `sys.argv` for this test only
    capsys: pytest.CaptureFixture[str],  # Captures what the command prints
    words: tuple[str, ...],  # The command line, as words
) -> None:
    """`babykev`, `babykev help`, `--help` and `-h` all print the help text, and nothing else."""
    run(monkeypatch, *words)
    assert capsys.readouterr().out == HELP


def test_an_unknown_command_is_refused_with_exit_code_2(
    monkeypatch: pytest.MonkeyPatch,  # Replaces `sys.argv` for this test only
    capsys: pytest.CaptureFixture[str],  # Captures what the command prints
) -> None:
    """A word that is not a command is named, the help is pointed to, and the exit code is 2."""
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "train")
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "no command named 'train'" in err
    assert "babykev help" in err


def test_docs_is_passed_to_the_docs_command(
    monkeypatch: pytest.MonkeyPatch,  # Replaces `sys.argv` for this test only
    capsys: pytest.CaptureFixture[str],  # Captures what the command prints
) -> None:
    """`babykev docs list` runs the docs command and exits with its code."""
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "docs", "list")
    assert exit_info.value.code == 0
    assert "babykev.api" in capsys.readouterr().out
