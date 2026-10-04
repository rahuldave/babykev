"""Build the reference pages of the docs site from what the program says about itself.

    python docs/apidocs.py build     one page per module into docs/api/, and the index of names

`babykev docs module NAME` prints plain Markdown with no links. This script adds what a site needs:
a front-matter title, an explicit id on every heading so that anchors are stable, a link to the line
in the source, and `_symbols.json`, the index that `linkify.lua` reads to turn a name in backticks
into a link. Quarto runs it before every render (`pre-render` in `_quarto.yml`); nothing is edited.
"""

import importlib
import json
import re
import sys
from pathlib import Path

from babykev.docs import PACKAGE, Symbol, inventory, markdown

REPO = "https://github.com/rahuldave/babykev"
DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
OUT = DOCS / "api"
HEADING = re.compile(r"^(#+) (?:class )?(\S+)$")


def source_link(
    sym: Symbol,  # One entry of the inventory
) -> str:  # A Markdown link to the line that defines it on GitHub
    """Link a symbol to the line that defines it."""
    file = Path(importlib.import_module(sym.module).__file__ or "").resolve().relative_to(ROOT)
    return f"[source]({REPO}/blob/main/{file.as_posix()}#L{sym.line}){{.source-link}}"


def page(
    name: str,  # The module's short name, such as `api`
    syms: list[Symbol],  # The inventory entries of that module, the module's own first
) -> str:  # The text of its `.qmd` page
    """Turn one module's Markdown into a Quarto page: front matter, an id and a link per heading."""
    by_name = {s.name: s for s in syms if s.name}
    mod = next(s for s in syms if not s.name)
    body = markdown(name).split("\n", 1)[1].strip()
    if body.startswith(mod.summary):
        body = body[len(mod.summary) :].strip()
    out = []
    for line in body.split("\n"):
        m = HEADING.match(line)
        if m and m.group(2) in by_name:
            sym = by_name[m.group(2)]
            out += [f"{line} {{#{sym.name}}}", "", source_link(sym)]
        else:
            out.append(line)
    front = f"---\ntitle: {json.dumps(mod.module)}\ndescription: {json.dumps(mod.summary)}\n---\n\n"
    return front + "\n".join(out) + "\n"


def build() -> int:  # The exit code, always 0
    """Write one reference page per module, and the index of names that cross-references use."""
    OUT.mkdir(exist_ok=True)
    index: dict[str, str] = {}
    modules = sorted({s.module for s in inventory(PACKAGE)})
    for name in modules:
        syms = [s for s in inventory(PACKAGE) if s.module == name]
        short = name.rpartition(".")[2]
        (OUT / f"{short}.qmd").write_text(page(short, syms), encoding="utf-8")
        for s in syms:
            if s.name:
                target = f"/api/{short}.qmd#{s.name}"
                index.setdefault(s.name, target)  # `render`
                index[f"{name}.{s.name}"] = target  # `babykev.api.render`
    (DOCS / "_symbols.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {len(modules)} page(s) to {OUT.relative_to(ROOT)}/")
    return 0


def main(
    command: str = "build",  # The one command the script has
) -> int:  # The exit code: 0, or 2 for a word that is no command
    """Run one command of the script: only `build`, so far."""
    if command == "build":
        return build()
    print(f"unknown command {command!r}: use build")
    return 2


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
