# babykev: every task is a recipe here. `just` lists them; `just help VERB` explains one.

program := "babykev"

# Each word after a recipe's name reaches its commands whole, as "$1", "$2" and so on; "$@" is all of them
set positional-arguments

# uv refuses a stale lockfile instead of quietly re-locking it: a lockfile changes only on purpose
export UV_LOCKED := "1"

[private]
default:
    @just --list --unsorted

# Explain a verb: its line, and the program's flags where the program has them. Alone, the list
[no-exit-message]
help *verb:
    #!/usr/bin/env sh
    if [ "$#" -eq 0 ]; then just --list --unsorted; exit 0; fi
    shown=$(just --show "$1") || exit 1
    printf '%s\n' "$shown" | head -1 | sed 's/^# //'
    if printf '%s\n' "$shown" | grep -qF '{{ "{{ program }}" }} '"$1"; then
        uv run {{ program }} "$1" help
    fi

# Install Python and every dependency into .venv, exactly as the lockfile says, and the git hook
[group('setup')]
setup *args:
    {{ if args == "help" { "just help setup" } else { "uv sync --locked && uv run pre-commit install" } }}

# Format the code, one way, no arguments. Extra arguments go to ruff: just fmt --check, just fmt --diff
[group('quality')]
[no-exit-message]
fmt *args:
    {{ if args == "help" { "just help fmt" } else { 'uv run ruff format "$@"' } }}

# Check the code against the rules in pyproject.toml. Extra arguments go to ruff: just lint --statistics
[group('quality')]
[no-exit-message]
lint *args:
    {{ if args == "help" { "just help lint" } else { 'uv run ruff check "$@"' } }}

# Lint the config files: the YAML files, pyproject.toml, the lockfile and this justfile
[group('quality')]
[no-exit-message]
lint-config *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help lint-config; exit 0; fi
    uv run yamllint --strict .
    uv run validate-pyproject pyproject.toml
    uv lock --check
    just --fmt --check --unstable

# Run the tests, with coverage of src/babykev. Extra arguments go to pytest: just test -k render
[group('quality')]
[no-exit-message]
test *args:
    {{ if args == "help" { "just help test" } else { 'uv run pytest "$@"' } }}

# Everything a commit must pass: format, lint, the tests over the coverage floor, the config files
[group('quality')]
[no-exit-message]
check *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help check; exit 0; fi
    just fmt --check
    just lint
    just test -q --cov-fail-under=90
    just lint-config
