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

# Install Python and every dependency into .venv, exactly as the lockfile says, and the git hook. Says if Quarto is missing
[group('setup')]
setup *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help setup; exit 0; fi
    uv sync --locked
    uv run pre-commit install
    command -v quarto > /dev/null || echo "Quarto is not on this machine; just docs needs it: https://quarto.org/docs/get-started/"

# The two images of the Dockerfile, by commit: just image build dev|run, just image start|shell|stop, just image run smoke
[group('setup')]
[no-exit-message]
image *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help image; exit 0; fi
    if ! command -v docker > /dev/null; then
        echo "image: there is no docker here. Inside the dev image, run the recipes themselves: just check" >&2
        exit 2
    fi
    tag=$(git rev-parse --short HEAD)
    cache="${HF_HOME:-$HOME/.cache/huggingface}"
    name="{{ program }}-dev"
    if [ -t 0 ]; then terminal=-it; else terminal=-i; fi
    # The step browser's port in the dev image, 8765, is published on the first free port here from 8765 up
    port=8765
    while command -v nc > /dev/null && nc -z 127.0.0.1 "$port" 2> /dev/null; do port=$((port + 1)); done
    case "$1" in
        build)
            case "$2" in dev | run) ;; *) echo "image build: say which image, dev or run" >&2; exit 2 ;; esac
            if [ -n "$(git status --porcelain)" ]; then
                echo "image: the working copy has changes the commit $tag does not; the image holds them" >&2
            fi
            kind=$2
            shift 2
            exec docker build --target "$kind" --tag "{{ program }}-$kind:$tag" "$@" .
            ;;
        stop)
            exec docker rm --force "$name"
            ;;
        shell)
            if [ -n "$(docker ps --quiet --filter "name=^$name\$")" ]; then
                running=$(docker inspect --format '{{ "{{" }}.Config.Image{{ "}}" }}' "$name")
                there=$(docker port "$name" 8765 | head -1 | sed 's/.*://')
                echo "image: into $name, which runs $running; its port 8765 is http://localhost:$there here" >&2
                exec docker exec "$terminal" "$name" bash
            fi
            ;;
    esac
    case "$1" in
        start | shell | run)
            if [ "$1" = run ]; then kind=run; else kind=dev; fi
            image="{{ program }}-$kind:$tag"
            if ! docker image inspect "$image" > /dev/null 2>&1; then
                echo "image: there is no $image yet; build it with: just image build $kind" >&2
                exit 1
            fi
            case "$1" in
                run) shift; exec docker run --rm -v "$cache:/hf" "$image" "$@" ;;
                shell)
                    shift
                    echo "image: a new container of $image; its port 8765 is http://localhost:$port here" >&2
                    exec docker run --rm "$terminal" -p "$port:8765" -v "$PWD:/work" -v "$cache:/hf" "$@" "$image"
                    ;;
            esac
            if [ -n "$(docker ps --all --quiet --filter "name=^$name\$")" ]; then
                echo "image: $name is there already: just image shell goes into it, just image stop removes it" >&2
                exit 1
            fi
            shift
            docker run --detach -it --name "$name" -p "$port:8765" -v "$PWD:/work" -v "$cache:/hf" \
                "$@" "$image" > /dev/null
            echo "image: $name runs $image in the background; its port 8765 is http://localhost:$port here"
            echo "image: just image shell goes into it, just image stop ends it"
            ;;
        *) echo "image: build dev|run, start, shell, stop, or run WORDS" >&2; exit 2 ;;
    esac

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

# Lint the config files: the YAML files, the Dockerfile, pyproject.toml, the lockfile and this justfile
[group('quality')]
[no-exit-message]
lint-config *args:
    #!/usr/bin/env sh
    set -e
    if [ "$*" = help ]; then just help lint-config; exit 0; fi
    uv run yamllint --strict .
    uv run hadolint Dockerfile
    uv run validate-pyproject pyproject.toml
    uv lock --check
    just --fmt --check --unstable

# Check the types: what a caller passes against what a function declares. Extra arguments go to ty
[group('quality')]
[no-exit-message]
types *args:
    {{ if args == "help" { "just help types" } else { 'uv run ty check "$@"' } }}

# Run the tests, with coverage of src/babykev. Extra arguments go to pytest: just test -k render
[group('quality')]
[no-exit-message]
test *args:
    {{ if args == "help" { "just help test" } else { 'uv run pytest "$@"' } }}

# Everything a commit must pass: format, lint, types, the tests over the coverage floor, the docs, the config files
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

# Build the docs site into docs/_site; just docs publish puts it on GitHub Pages. Other words go to the program: just docs list
[group('docs')]
[no-exit-message]
docs *args:
    #!/usr/bin/env sh
    if [ "$*" = help ]; then just help docs; exit 0; fi
    if [ "$#" -eq 0 ]; then exec uv run quarto render docs; fi
    if [ "$*" = publish ]; then
        if ! origin=$(git remote get-url origin 2> /dev/null); then
            echo "docs publish: this repository has no remote named origin; GitHub Pages needs one on GitHub" >&2
            exit 2
        fi
        case "$origin" in
            *github.com*) ;;
            *) echo "docs publish: origin is $origin, not GitHub; GitHub Pages needs a GitHub repository" >&2; exit 2 ;;
        esac
        if ! git ls-remote --exit-code --heads origin gh-pages > /dev/null; then
            empty=$(git commit-tree "$(git hash-object -t tree /dev/null)" -m "An empty branch for GitHub Pages")
            git push origin "$empty:refs/heads/gh-pages"
        fi
        # The site is committed on gh-pages, which has no .pre-commit-config.yaml: the hook lets that pass
        PRE_COMMIT_ALLOW_NO_CONFIG=1 exec uv run quarto publish gh-pages docs --no-prompt --no-browser
    fi
    exec uv run {{ program }} docs "$@"

# Run everything that exists so far on the real model: just smoke local. Other platforms arrive later
[group('model')]
[no-exit-message]
smoke *args:
    #!/usr/bin/env sh
    case "$*" in
        help) just help smoke ;;
        local) exec uv run {{ program }} smoke ;;
        *) echo "smoke: the platform is local; $* is not here yet" >&2; exit 2 ;;
    esac
