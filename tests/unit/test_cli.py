"""Unit tests for the `babykev` command: `help`, `smoke help`, and the words that it refuses."""

import pytest

from babykev.cli import HELP, main
from babykev.smoke import USAGE


def run(
    monkeypatch: pytest.MonkeyPatch,  # pytest's fixture, which undoes each change after the test
    *words: str,  # The words after `babykev` on the command line
) -> None:
    """Run `babykev` with these words as its command line."""
    monkeypatch.setattr("sys.argv", ["babykev", *words])
    main()


@pytest.mark.parametrize("words", [(), ("help",), ("--help",), ("-h",)])
def test_help_is_printed_for_help_and_for_no_words(
    monkeypatch: pytest.MonkeyPatch,  # Sets the command line for the test
    capsys: pytest.CaptureFixture[str],  # pytest's fixture, which catches what the test prints
    words: tuple[str, ...],  # The words of the command line, which ask for the help
) -> None:
    """`babykev`, `babykev help`, `--help` and `-h` all print the help text, and nothing else."""
    run(monkeypatch, *words)
    assert capsys.readouterr().out == HELP


def test_an_unknown_command_is_refused_with_exit_code_2(
    monkeypatch: pytest.MonkeyPatch,  # Sets the command line for the test
    capsys: pytest.CaptureFixture[str],  # Catches the message of the refusal
) -> None:
    """A word that is not a command is named, the help is pointed to, and the exit code is 2."""
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "train")
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "no command named 'train'" in err
    assert "babykev help" in err


def test_smoke_help_prints_the_usage(
    monkeypatch: pytest.MonkeyPatch,  # Sets the command line for the test
    capsys: pytest.CaptureFixture[str],  # Catches the usage
) -> None:
    """`babykev smoke help` prints the usage of the smoke, loads no model, and exits with 0."""
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "smoke", "help")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == USAGE
