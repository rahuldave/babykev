# The images of babykev, built from uv.lock, so that they hold the versions that the lockfile names.
#
#   docker build --target run --tag babykev-run .
#       the program and what it needs, and no tool of the developers. It starts babykev
#
# The environment is in /opt/venv, and never in the folder of the project.

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
