# babykev: every task is a recipe here. `just` lists them.

[private]
default:
    @just --list --unsorted

# Install Python and every dependency into .venv, exactly as the lockfile says
[group('setup')]
setup:
    uv sync --locked
