# The images of babykev, built from uv.lock, so that they hold the versions that the lockfile names.
#
#   docker build --target run --tag babykev-run .
#       the program and what it needs, and no tool of the developers. It starts babykev
#   docker build --target dev --tag babykev-dev .
#       the whole project, with every group, uv, just and git. It starts a shell
#
# The environment is in /opt/venv, and never in the folder of the project. So the dev image can run
# the folder of this machine, mounted at /work, and never uses its .venv.

# uv, copied from its own image, at the version that wrote the lockfile
FROM ghcr.io/astral-sh/uv:0.12.13 AS uv

# The environment, with no group of the developers. The packages come first and the code after, so a
# change to the code installs one package again, and not torch
FROM python:3.13-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --no-install-project
COPY README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

# The run image: Python, the environment, the program and its sample data. It has no uv
FROM python:3.13-slim-trixie AS run
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY data ./data
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["babykev"]
CMD ["help"]

# The dev image: the whole project with every group, and the tools to work on it
FROM python:3.13-slim-trixie AS dev
ARG JUST=1.50.0
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1
# just, as a wheel from PyPI. git refuses a repository that another user owns, and a mounted folder
# can belong to the user of this machine, so every folder is declared safe
RUN uv tool install "rust-just==${JUST}" \
    && git config --system --add safe.directory '*'
WORKDIR /work
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked
CMD ["bash"]
