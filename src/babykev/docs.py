"""What the code says about itself, as plain Markdown: every symbol, one module, or the gaps.

    babykev docs list          every module, class, method and function, one line each
    babykev docs module NAME   one module: a heading per symbol, its signature, docstring and table
    babykev docs check         count what is typed and documented, list every gap, fail if any

Types come from the annotations, the description of a parameter from its docment (the comment beside
it, read by fastcore) and the summary from the docstring. Nothing is written twice. The output is
plain Markdown with no links, ids or URLs, so that a person, an agent or the docs site can read it.
`list` and `module` show the package; `check` holds the package, the tests, and the Python files at
the root and under `docs/` to the convention.

How to write a docment, so that the formatter and the linter leave it alone: one parameter per
line, a trailing comma after the last, the comment two spaces after the code; the return's comment
after the colon of the `def` line. Every function in this file is an example. The whole line must
fit in 100 characters: the formatter wraps a longer line, the comment lands away from its parameter,
and `check` reports it as missing. The formatter never joins a line that carries a comment, and no
lint rule looks inside one.
"""

import importlib
import importlib.util
import inspect
import re
import sys
import types
import typing
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from pkgutil import iter_modules
from types import FunctionType, ModuleType

from fastcore.docments import docments
from pydantic import BaseModel

PACKAGE = "babykev"
ROOT = Path()
EMPTY = inspect.Parameter.empty
KEY_COMMENT = re.compile(r"^\s+(\w+)\s*:[^#\n]+#\s*(.+)$", re.MULTILINE)
USAGE = """\
usage: babykev docs list            every module, class, method and function, one line each
       babykev docs module NAME     one module as Markdown: api, cli or docs
       babykev docs check           what is typed and documented, and every gap; exit 1 if any
"""

Code = FunctionType | type


@dataclass(frozen=True)
class Symbol:
    """One line of the inventory: a module, a class, a method or a function, and where it is."""

    module: str  # The module's full name, such as `babykev.api`
    name: str  # `render` or `Choice._check`: the name inside the module; empty for the module
    kind: str  # `module`, `class`, `method` or `function`
    summary: str  # The first line of the docstring, or empty when there is none
    line: int  # The line the definition starts on; 1 for a module


def package_modules(
    package: str = PACKAGE,  # The name of an importable package
) -> list[ModuleType]:  # Its public modules, in name order
    """Import every public module of a package, in name order."""
    pkg = importlib.import_module(package)
    names = sorted(m.name for m in iter_modules(pkg.__path__) if not m.name.startswith("_"))
    return [importlib.import_module(f"{package}.{n}") for n in names]


def python_files(
    root: Path,  # The repository's root
) -> list[Path]:  # Every `.py` file at the root, under `docs/` and under `tests/`, in that order
    """Find the Python files that are not the package and are held to the convention too."""
    found = sorted(root.glob("*.py")) + sorted((root / "docs").glob("*.py"))
    return found + sorted((root / "tests").rglob("*.py"))


def file_modules(
    files: list[Path],  # Python files that are not part of a package
) -> list[ModuleType]:  # One module per file, named by its path, in the order given
    """Import Python files as modules of their own, named by their paths."""
    mods = []
    for file in files:
        name = file.as_posix()
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods.append(mod)
    return mods


def module(
    name: str,  # The module's short name, `api`, or its full name, `babykev.api`
    package: str = PACKAGE,  # The package to look in
) -> ModuleType:  # The imported module; `LookupError` when the package has no such module
    """Find one module of the package by its short name or its full name."""
    for mod in package_modules(package):
        if name in (mod.__name__, mod.__name__.rpartition(".")[2]):
            return mod
    raise LookupError(name)


def doc(
    obj: object,  # A module, class or function
) -> str:  # Its own docstring, cleaned of its indentation; empty when there is none
    """Read an object's own docstring, cleaned of its indentation; empty when there is none."""
    return inspect.cleandoc(obj.__doc__) if isinstance(obj.__doc__, str) else ""


def summary(
    obj: object,  # A module, class or function
) -> str:  # The first line of its docstring; empty when there is none
    """Read the first line of an object's docstring."""
    return doc(obj).partition("\n")[0]


def line_of(
    sym: Code,  # A function, method or class
) -> int:  # The line its definition starts on
    """Find the line a symbol's definition starts on, to keep symbols in source order."""
    return inspect.getsourcelines(sym)[1]


def unwrapped(
    obj: object,  # Anything a module or a class holds under a name
) -> object:  # The function inside it when it is a decorator's wrapper; otherwise the object itself
    """See through a decorator that kept the function, as `functools.wraps` and a fixture do."""
    inner = getattr(obj, "__func__", obj)
    return inspect.unwrap(inner) if callable(inner) and hasattr(inner, "__wrapped__") else inner


def own(
    mod: ModuleType,  # An imported module
    private: bool = False,  # Include names that start with an underscore
) -> list[Code]:  # The functions and classes the module defines, in source order
    """List the functions and classes a module defines itself, in source order; not its imports."""
    found = [
        o
        for n, o in ((n, unwrapped(o)) for n, o in vars(mod).items())
        if (private or not n.startswith("_"))
        and (inspect.isfunction(o) or inspect.isclass(o))
        and o.__module__ == mod.__name__
    ]
    return sorted(found, key=line_of)


def is_dunder(
    name: str,  # An attribute name
) -> bool:  # Whether it is one of Python's own, written `__like_this__`
    """Tell whether a name is one of Python's own, such as `__init__`: a dataclass writes those."""
    return name.startswith("__") and name.endswith("__")


def methods(
    klass: type,  # A class
    private: bool = False,  # Include names that start with one underscore
) -> list[FunctionType]:  # The methods the class defines, in source order
    """List the methods a class defines itself, in source order, not what it inherits."""
    found = [(n, unwrapped(o)) for n, o in vars(klass).items() if not is_dunder(n)]
    own_ = [o for n, o in found if (private or not n.startswith("_")) and inspect.isfunction(o)]
    return sorted(own_, key=line_of)


def held(
    mod: ModuleType,  # An imported module
) -> list[Code]:  # Every function, class and method in it, private ones too
    """List everything `check` holds to the convention: every function, class and method."""
    tops = own(mod, private=True)
    return [s for t in tops for s in [t, *(methods(t, private=True) if inspect.isclass(t) else [])]]


def inventory(
    package: str = PACKAGE,  # The name of an importable package
) -> list[Symbol]:  # One entry per public module, class, method and function, in source order
    """List every public module, class, method and function of a package, with its summary."""
    out = []
    for mod in package_modules(package):
        out.append(Symbol(mod.__name__, "", "module", summary(mod), 1))
        for top in own(mod):
            kind = "class" if inspect.isclass(top) else "function"
            out.append(Symbol(mod.__name__, top.__qualname__, kind, summary(top), line_of(top)))
            if inspect.isclass(top):
                out += [
                    Symbol(mod.__name__, m.__qualname__, "method", summary(m), line_of(m))
                    for m in methods(top)
                ]
    return out


def name_of(
    anno: object,  # A type, as an annotation holds it
    mod: ModuleType,  # The module the annotation was written in
) -> str | None:  # The name that module gives the type, or `None` when it has none
    """Find what a module calls a type: an alias such as `JSONContent`, or the name it imported."""
    for name, value in vars(mod).items():
        if name.startswith("_") or inspect.ismodule(value):
            continue
        if value is anno or (typing.get_origin(value) is not None and value == anno):
            return name
    return None


def type_text(
    anno: object,  # A type, as an annotation holds it
    mod: ModuleType,  # The module the annotation was written in
) -> str:  # The type as the source would write it
    """Write a type the way the source does: `JSONContent`, not the union it stands for."""
    if isinstance(anno, str):
        return anno
    if name := name_of(anno, mod):
        return name
    origin, args = typing.get_origin(anno), typing.get_args(anno)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(type_text(a, mod) for a in args)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin is not None:
        return f"{type_text(origin, mod)}[{', '.join(type_text(a, mod) for a in args)}]"
    if anno is None or anno is type(None):
        return "None"
    if anno is Ellipsis:
        return "..."
    if isinstance(anno, type):
        return anno.__name__
    return repr(anno).replace("typing.", "")


def signature(
    sym: Code,  # A function, method or class
    mod: ModuleType,  # The module it was written in
) -> str:  # Its parameters and return as one line of Python, without the name
    """Write a signature as one line of Python: each parameter with its type and default."""
    sig, parts, star = inspect.signature(sym), [], False
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
        star = star or p.kind in (p.KEYWORD_ONLY, p.VAR_POSITIONAL)
        prefix = {p.VAR_POSITIONAL: "*", p.VAR_KEYWORD: "**"}.get(p.kind, "")
        text = prefix + p.name
        if p.annotation is not EMPTY:
            text += f": {type_text(p.annotation, mod)}"
        if p.default is not EMPTY:
            text += f" = {p.default!r}"
        parts.append(text)
    if inspect.isclass(sym) or sig.return_annotation is EMPTY:
        return f"({', '.join(parts)})"
    return f"({', '.join(parts)}) -> {type_text(sig.return_annotation, mod)}"


def cell(
    text: str,  # The text of one cell
) -> str:  # The same text with every `|` escaped
    """Make text safe inside a Markdown table cell."""
    return text.replace("|", "\\|")


def table(
    rows: list[list[str]],  # The cells of each row
    head: list[str],  # The column titles
) -> str:  # A Markdown table; empty when there are no rows
    """Write a Markdown table, or nothing when there are no rows."""
    if not rows:
        return ""
    lines = [head, ["---"] * len(head), *rows]
    return "\n".join("| " + " | ".join(cell(c) for c in row) + " |" for row in lines)


def params_table(
    sym: Code,  # A function, method or dataclass
    mod: ModuleType,  # The module it was written in
) -> str:  # A Markdown table of its parameters and its return: type, default, description
    """Describe a function by its parameters and its return: type, default, docment."""
    rows = []
    for name, d in docments(sym, full=True).items():
        if name in ("self", "cls"):
            continue
        anno = "" if d.anno is EMPTY else type_text(d.anno, mod)
        if name == "return":
            if d.anno is not EMPTY and d.anno is not None:
                rows.append(["returns", anno, "", d.docment or ""])
            continue
        default = "required" if d.default is EMPTY else f"`{d.default!r}`"
        rows.append([f"`{name}`", anno, default, d.docment or ""])
    return table(rows, ["Parameter", "Type", "Default", "Description"])


def fields_table(
    model: type[BaseModel],  # A pydantic model class
    mod: ModuleType,  # The module it was written in
) -> str:  # A Markdown table of its fields: type, default, description
    """Describe a pydantic model by its fields, because its constructor signature says nothing."""
    rows = []
    for name, f in model.model_fields.items():
        default = "required" if f.is_required() else f"`{f.default!r}`"
        rows.append([f"`{name}`", type_text(f.annotation, mod), default, f.description or ""])
    return table(rows, ["Field", "Type", "Default", "Description"])


def key_comments(
    klass: type,  # A TypedDict class
) -> dict[str, str]:  # For each key that has one, the comment beside it
    """Read the docments of a TypedDict: the comment beside each key in the class body."""
    return dict(KEY_COMMENT.findall(inspect.getsource(klass)))


def keys_table(
    klass: type,  # A TypedDict class
    mod: ModuleType,  # The module it was written in
) -> str:  # A Markdown table of its keys: type, description
    """Describe a TypedDict by its keys, because calling it says nothing."""
    said = key_comments(klass)
    keys = klass.__annotations__.items()
    rows = [[f"`{n}`", type_text(a, mod), said.get(n, "")] for n, a in keys]
    return table(rows, ["Key", "Type", "Description"])


def has_init(
    klass: type,  # A class
) -> bool:  # Whether it has a constructor of its own, written by hand or by `dataclass`
    """Tell whether a class has a constructor of its own to show, as a dataclass does."""
    return is_dataclass(klass) or "__init__" in vars(klass)


def entry(
    sym: Code,  # A function, method or class
    level: int,  # The heading level of its entry: 2 for a top-level symbol, 3 for a method
) -> str:  # The Markdown of one entry
    """Write one symbol: a heading, its signature, its docstring and the table of its parameters."""
    mod = sys.modules[sym.__module__]
    is_class = inspect.isclass(sym)
    head = f"{'#' * level} {'class ' if is_class else ''}{sym.__qualname__}"
    if is_class and issubclass(sym, BaseModel):
        parts = [head, doc(sym), fields_table(sym, mod)]
    elif is_class and typing.is_typeddict(sym):
        parts = [head, doc(sym), keys_table(sym, mod)]
    elif is_class and not has_init(sym):
        parts = [head, doc(sym)]
    else:
        code = f"```python\n{sym.__qualname__}{signature(sym, mod)}\n```"
        parts = [head, code, doc(sym), params_table(sym, mod)]
    return "\n\n".join(p for p in parts if p)


def markdown(
    name: str,  # The module's short name, `api`, or its full name, `babykev.api`
    package: str = PACKAGE,  # The package to look in
) -> str:  # The whole module as plain Markdown
    """Write one module as plain Markdown: its docstring, then an entry for each public symbol."""
    mod = module(name, package)
    parts = [f"# {mod.__name__}", doc(mod)]
    for top in own(mod):
        parts.append(entry(top, 2))
        if inspect.isclass(top):
            parts += [entry(m, 3) for m in methods(top)]
    return "\n\n".join(p for p in parts if p) + "\n"


def gaps_of(
    sym: Code,  # A function, method or class
) -> list[str]:  # One message per gap; empty when the symbol is fully typed and documented
    """Find what one symbol is missing: a docstring, a type, a docment, or a field description."""
    name, out = f"{sym.__module__}:{sym.__qualname__}", []
    if not doc(sym):
        out.append(f"{name}: no docstring")
    is_class = inspect.isclass(sym)
    if is_class and issubclass(sym, BaseModel):
        fields = sym.model_fields.items()
        return out + [f"{name}.{n}: no description" for n, f in fields if not f.description]
    if is_class and typing.is_typeddict(sym):
        said = key_comments(sym)
        return out + [f"{name}.{n}: no docment" for n in sym.__annotations__ if n not in said]
    if is_class and not has_init(sym):
        return out
    for pname, d in docments(sym, full=True).items():
        if pname in ("self", "cls"):
            continue
        if pname == "return":
            if d.anno is EMPTY and not is_class:
                out.append(f"{name}: no return type")
            elif d.anno is not None and not d.docment and not is_class:
                out.append(f"{name}: no return docment")
            continue
        if d.anno is EMPTY:
            out.append(f"{name}({pname}): no type")
        if not d.docment:
            out.append(f"{name}({pname}): no docment")
    return out


def gaps(
    mod: ModuleType,  # An imported module
) -> dict[str, list[str]]:  # The gaps of each symbol under its name, the module's under its own
    """Find every gap in a module, under the module's name and then under each symbol's."""
    out = {mod.__name__: [] if doc(mod) else [f"{mod.__name__}: no module docstring"]}
    for sym in held(mod):
        out[sym.__qualname__] = gaps_of(sym)
    return out


def check(
    package: str = PACKAGE,  # The package to hold to the convention
    root: Path = ROOT,  # The repository's root, whose own Python files are held to it too
) -> int:  # The exit code: 0 when nothing is missing, 1 otherwise
    """Print how much of each module is typed and documented, then every gap, then the total."""
    found, done, total = [], 0, 0
    if str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))  # a test imports its helpers as tests.tiny
    for mod in package_modules(package) + file_modules(python_files(root)):
        missing = gaps(mod)
        found += missing.pop(mod.__name__)
        complete = sum(not g for g in missing.values())
        print(f"{mod.__name__:32s} {complete:3d} of {len(missing):3d}")
        found += [message for g in missing.values() for message in g]
        done, total = done + complete, total + len(missing)
    print("\n".join(["", *found]) if found else "")
    print(f"{done} of {total} functions, methods and classes are typed and documented")
    return 1 if found else 0


def main(
    words: list[str],  # The words after `babykev docs`: `list`, `module NAME`, `check`, `help`
    package: str = PACKAGE,  # The package to document
    root: Path = ROOT,  # The repository's root, for the files `check` holds to the convention
) -> int:  # The exit code: 0, 1 when `check` finds a gap, 2 for words that are no command
    """Run one docs command and return its exit code."""
    if words in (["help"], ["--help"], ["-h"]):
        print(USAGE, end="")
        return 0
    if words == ["list"]:
        found = inventory(package)
        names = [f"{s.module}.{s.name}" if s.name else s.module for s in found]
        width = max(len(n) for n in names)
        for s, name in zip(found, names, strict=True):
            print(f"{s.kind:9s} {name:{width}s}  {s.summary}")
        return 0
    if len(words) == 2 and words[0] == "module":
        try:
            print(markdown(words[1], package), end="")
        except LookupError:
            names = ", ".join(m.__name__.rpartition(".")[2] for m in package_modules(package))
            print(f"babykev docs: no module {words[1]!r}. The modules: {names}", file=sys.stderr)
            return 2
        return 0
    if words == ["check"]:
        return check(package, root)
    print(USAGE, end="", file=sys.stderr)
    return 2
