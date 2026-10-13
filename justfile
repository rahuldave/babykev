# babykev: every task is a recipe here. `just` lists them; `just help VERB` explains one.

program := "babykev"

# Each word after a recipe's name reaches its commands whole, as "$1", "$2" and so on; "$@" is all of them
set positional-arguments

# Every uv run in a recipe refuses a lockfile that is out of date, in place of writing a new one
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

# Install Python and every dependency into .venv, exactly as the lockfile says
[group('setup')]
setup *args:
    @{{ if args == "help" { "just help setup" } else { "uv sync --locked" } }}

# Install the git hook, which runs just check before each commit. Once in each clone: just setup never touches git
[group('setup')]
hook *args:
    @{{ if args == "help" { "just help hook" } else { "uv run pre-commit install" } }}

# Write the Python of the project in one style. Extra words go to ruff: just fmt --check
[group('quality')]
[no-exit-message]
fmt *args:
    @{{ if args == "help" { "just help fmt" } else { 'uv run ruff format "$@"' } }}

# Check the code against the rules in pyproject.toml. Extra words go to ruff: just lint --statistics
[group('quality')]
[no-exit-message]
lint *args:
    @{{ if args == "help" { "just help lint" } else { 'uv run ruff check "$@"' } }}

# Check the config files: pyproject.toml, the lockfile, this justfile, and every YAML file
[group('quality')]
[no-exit-message]
lint-config *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help lint-config; exit 0; fi
    uv run validate-pyproject pyproject.toml
    uv lock --check
    just --fmt --check --unstable
    uv run yamllint --strict .

# Run the tests. Extra words go to pytest: just test -k render
[group('quality')]
[no-exit-message]
test *args:
    @{{ if args == "help" { "just help test" } else { 'uv run pytest "$@"' } }}

# Run the checks one after another, and stop at the first that fails: the format, the linter, the tests, the config files
[group('quality')]
[no-exit-message]
check *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help check; exit 0; fi
    just fmt --check
    just lint
    just test -q
    just lint-config
