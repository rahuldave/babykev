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

# The two images of the Dockerfile: just image build dev|run, just image shell, just image run smoke
[group('setup')]
[no-exit-message]
image *args:
    #!/usr/bin/env bash
    set -e
    if [ "$*" = help ]; then just help image; exit 0; fi
    command -v docker > /dev/null || { echo "image: there is no docker on this machine" >&2; exit 2; }
    # The cache of the Hugging Face Hub on this machine, mounted at /hf, so the model is downloaded once
    cache="${HF_HOME:-$HOME/.cache/huggingface}"
    case "$1" in
        build)
            case "$2" in dev | run) ;; *) echo "image build: say which image, dev or run" >&2; exit 2 ;; esac
            kind=$2
            shift 2
            exec docker build --target "$kind" --tag "{{ program }}-$kind" "$@" .
            ;;
        shell)
            # The words before -- go to docker, and the words after it are the command to run
            shift
            options=()
            while [ $# -gt 0 ] && [ "$1" != -- ]; do options+=("$1"); shift; done
            if [ "$1" = -- ]; then shift; fi
            if [ -t 0 ]; then options+=(-it); fi
            exec docker run --rm -v "$PWD:/work" -v "$cache:/hf" "${options[@]}" {{ program }}-dev "$@"
            ;;
        run)
            shift
            exec docker run --rm -v "$cache:/hf" {{ program }}-run "$@"
            ;;
        *) echo "image: build dev|run, shell [OPTIONS] [-- COMMAND], or run WORDS" >&2; exit 2 ;;
    esac

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

# Check the types, by reading the code and not running it. Extra words go to ty: just types src
[group('quality')]
[no-exit-message]
types *args:
    @{{ if args == "help" { "just help types" } else { 'uv run ty check "$@"' } }}

# Run the tests. Extra words go to pytest: just test -k render
[group('quality')]
[no-exit-message]
test *args:
    @{{ if args == "help" { "just help test" } else { 'uv run pytest "$@"' } }}

# Run the checks one after another, and stop at the first that fails: the format, the linter, the types, the tests and their coverage, the docs, the config files
[group('quality')]
[no-exit-message]
check *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help check; exit 0; fi
    just fmt --check
    just lint
    just types
    just test -q --cov-fail-under=90
    just docs check
    just lint-config

# The docs, by smalldocs. Alone, the site in docs/_site. Words go to smalldocs api: just docs check
[group('docs')]
[no-exit-message]
docs *args:
    #!/usr/bin/env sh
    if [ "$*" = help ]; then just help docs; exit 0; fi
    if [ "$#" -eq 0 ]; then exec uv run smalldocs render docs; fi
    exec uv run smalldocs api "$@" --dir docs

# Run the smoke on this machine: the real model answers the sample requests, untrained
[group('model')]
[no-exit-message]
smoke *args:
    #!/usr/bin/env sh
    if [ "$*" = help ]; then just help smoke; exit 0; fi
    exec uv run {{ program }} smoke "$@"
