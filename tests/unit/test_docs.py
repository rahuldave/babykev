"""Unit tests for `babykev.docs`: the inventory, the Markdown of a module, the gaps, the command."""

import sys
from pathlib import Path

import pytest

from babykev import docs

# A small package to document: one module with a documented function, a bare one, and a class
THING = '''
"""A thing, with one function that is documented and one that is not."""

from typing import Literal

from pydantic import BaseModel, Field

Pair = tuple[int, int]


def greet(
    name: str,  # Who to greet
    times: int = 1,  # How many times over
) -> str:  # The greeting
    """Say hello."""
    return f"hello {name}" * times


def bare(x, y=2):
    return x + y


def span(
    p: Pair,  # Two numbers
) -> None:
    """Print the pair."""
    print(p)


class Box(BaseModel):
    """A box, as a pydantic model: its fields are its signature."""

    kind: Literal["big", "small"] = Field(description="How big")
    label: str | None = None


class Thing:
    """A thing with a method."""

    def size(self) -> int:  # Always one
        """Measure the thing."""
        return 1

    def _hidden(self):
        return 0
'''


def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write the example package into a folder, make it importable, and return its name."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text('"""The example package."""\n', encoding="utf-8")
    (tmp_path / "pkg" / "thing.py").write_text(THING, encoding="utf-8")
    for name in ("pkg", "pkg.thing"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(tmp_path)
    return "pkg"


# The inventory


def test_inventory_lists_the_public_symbols_in_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every module, function, class and public method is listed once, with its kind and summary."""
    pkg = example(tmp_path, monkeypatch)
    found = [(s.kind, s.name, s.summary) for s in docs.inventory(pkg)]
    assert found == [
        ("module", "", "A thing, with one function that is documented and one that is not."),
        ("function", "greet", "Say hello."),
        ("function", "bare", ""),
        ("function", "span", "Print the pair."),
        ("class", "Box", "A box, as a pydantic model: its fields are its signature."),
        ("class", "Thing", "A thing with a method."),
        ("method", "Thing.size", "Measure the thing."),
    ]


def test_inventory_knows_the_line_of_each_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line of a symbol is where its `def` or `class` is in the source."""
    pkg = example(tmp_path, monkeypatch)
    lines = {s.name: s.line for s in docs.inventory(pkg)}
    source = THING.splitlines()
    assert source[lines["greet"] - 1].startswith("def greet(")
    assert source[lines["Thing"] - 1] == "class Thing:"
    assert source[lines["Thing.size"] - 1].strip().startswith("def size(")


# The Markdown of a module


def test_markdown_has_a_heading_a_signature_a_docstring_and_a_table_per_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function's entry shows its signature as Python and its parameters with their docments."""
    text = docs.markdown("thing", example(tmp_path, monkeypatch))
    assert text.startswith("# pkg.thing\n\nA thing, with one function")
    assert (
        "## greet\n\n```python\ngreet(name: str, times: int = 1) -> str\n```\n\nSay hello." in text
    )
    assert "| `name` | str | required | Who to greet |" in text
    assert "| `times` | int | `1` | How many times over |" in text
    assert "| returns | str |  | The greeting |" in text


def test_markdown_writes_a_type_alias_by_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Pair` is written as the source wrote it, and a `None` return gets no row in the table."""
    text = docs.markdown("thing", example(tmp_path, monkeypatch))
    assert "span(p: Pair) -> None" in text
    assert "| `p` | Pair | required | Two numbers |" in text
    assert "| returns | None" not in text


def test_markdown_shows_a_class_and_its_public_methods_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class is a heading and its docstring; each public method is a heading one level down."""
    text = docs.markdown("thing", example(tmp_path, monkeypatch))
    assert "## class Thing\n\nA thing with a method.\n\n### Thing.size\n\n" in text
    assert "### Thing.size\n\n```python\nThing.size() -> int\n```\n\nMeasure the thing." in text
    assert "_hidden" not in text


def test_markdown_shows_a_pydantic_model_as_a_table_of_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model has no signature to show; its fields are the table, each with its description."""
    text = docs.markdown("thing", example(tmp_path, monkeypatch))
    assert (
        "## class Box\n\nA box, as a pydantic model: its fields are its signature.\n\n| Field |"
        in text
    )
    assert "| `kind` | Literal['big', 'small'] | required | How big |" in text
    assert "| `label` | str \\| None | `None` |  |" in text


def test_markdown_has_no_links_and_no_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The program's Markdown is plain: the site adds the ids and the links itself."""
    text = docs.markdown("thing", example(tmp_path, monkeypatch))
    assert "{#" not in text
    assert "](" not in text
    assert "http" not in text


def test_an_unknown_module_is_a_lookup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a module the package does not have raises, with the name."""
    with pytest.raises(LookupError, match="nothing"):
        docs.markdown("nothing", example(tmp_path, monkeypatch))


# The gaps


def test_gaps_names_what_each_symbol_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A documented symbol has no gaps; a bare one is missing its docstring, types and docments.

    `y` has a default, and fastcore takes a type from a default, so only its docment is missing.
    """
    pkg = example(tmp_path, monkeypatch)
    found = docs.gaps(docs.module("thing", pkg))
    assert found["pkg.thing"] == []
    assert found["greet"] == []
    assert found["Thing"] == []
    assert found["Thing.size"] == []
    assert found["Box"] == ["pkg.thing:Box.label: no description"]
    assert found["bare"] == [
        "pkg.thing:bare: no docstring",
        "pkg.thing:bare(x): no type",
        "pkg.thing:bare(x): no docment",
        "pkg.thing:bare(y): no docment",
        "pkg.thing:bare: no return type",
    ]
    assert found["Thing._hidden"] == [
        "pkg.thing:Thing._hidden: no docstring",
        "pkg.thing:Thing._hidden: no return type",
    ]


def test_a_return_that_is_not_none_needs_a_docment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`span` returns `None` and needs no docment for it; `greet` returns a value and has one."""
    pkg = example(tmp_path, monkeypatch)
    found = docs.gaps(docs.module("thing", pkg))
    assert found["span"] == []
    (tmp_path / "pkg" / "thing.py").write_text(THING.replace("-> str:  # The greeting", "-> str:"))
    monkeypatch.delitem(sys.modules, "pkg.thing")
    found = docs.gaps(docs.module("thing", pkg))
    assert found["greet"] == ["pkg.thing:greet: no return docment"]


# The command


def test_check_prints_a_count_per_module_then_every_gap_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check` prints one line per module, the gaps, the total, and exits 1 when there is a gap."""
    pkg = example(tmp_path, monkeypatch)
    assert docs.main(["check"], pkg, tmp_path / "no-tests") == 1
    out = capsys.readouterr().out
    assert "pkg.thing                          4 of   7" in out
    assert "pkg.thing:bare(x): no docment" in out
    assert out.endswith("4 of 7 functions, methods and classes are typed and documented\n")


def test_check_holds_the_tests_to_the_convention_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A folder of tests is checked like the package, each file under its path."""
    pkg = example(tmp_path, monkeypatch)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        '"""Tests."""\n\n\ndef test_x() -> None:\n    """One."""\n'
    )
    assert docs.main(["check"], pkg, tmp_path / "tests") == 1
    assert "/tests/test_x.py" in capsys.readouterr().out


def test_list_prints_one_line_per_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`list` prints the kind, the full name and the summary, aligned."""
    assert docs.main(["list"], example(tmp_path, monkeypatch)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("module    pkg.thing             A thing, with one function")
    assert lines[-1] == "method    pkg.thing.Thing.size  Measure the thing."


def test_module_prints_the_markdown_and_an_unknown_name_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`module NAME` prints the Markdown; a name that is no module is refused, the modules named."""
    pkg = example(tmp_path, monkeypatch)
    assert docs.main(["module", "thing"], pkg) == 0
    assert capsys.readouterr().out.startswith("# pkg.thing\n")
    assert docs.main(["module", "nothing"], pkg) == 2
    assert "no module 'nothing'. The modules: thing" in capsys.readouterr().err


def test_help_and_a_bad_word(capsys: pytest.CaptureFixture[str]) -> None:
    """`help` prints the usage and exits 0; anything else prints it to stderr and exits 2."""
    assert docs.main(["help"]) == 0
    assert capsys.readouterr().out == docs.USAGE
    assert docs.main(["render"]) == 2
    assert capsys.readouterr().err == docs.USAGE
