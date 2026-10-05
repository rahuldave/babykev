# Two images from one file, both built from uv.lock, so they hold the versions the lockfile names.
#
#   docker build --target dev --tag babykev-dev .
#       everything to work on the project: the dev tools, uv, just and Quarto. Starts a shell
#   docker build --target run --tag babykev-run .
#       only what the program needs, with no dev tools, no uv and no just. Starts `babykev`
#
# `just image build dev|run` runs these and names the image by the commit. The environment lives
# in /opt/venv, never in the project folder, so a mounted folder's own .venv is never used.

# uv itself, copied from its own image: the same version as on the machine that wrote the lockfile
FROM ghcr.io/astral-sh/uv:0.12.13 AS uv

# The environment without the dev tools. Its packages first and the code after, so a change to the
# code reinstalls one package, not torch.
FROM python:3.13-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# The run image: Python, the environment, the program and its sample data. What a platform starts
FROM python:3.13-slim-trixie AS run
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY data ./data
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["babykev"]
CMD ["help"]

# The dev image: the whole project with its dev group, and the three tools the README asks for
FROM python:3.13-slim-trixie AS dev
ARG TARGETARCH
ARG JUST=1.50.0
ARG QUARTO=1.8.26
# hadolint asks for a pinned version of each Debian package (DL3008). Debian replaces a package
# with each security fix, so a pinned version stops building within weeks. The base image is the pin
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
ARG RELEASES=https://github.com/quarto-dev/quarto-cli/releases/download
RUN curl --fail --silent --show-error --location --output /tmp/quarto.deb \
        "${RELEASES}/v${QUARTO}/quarto-${QUARTO}-linux-${TARGETARCH}.deb" \
    && dpkg --install /tmp/quarto.deb \
    && rm /tmp/quarto.deb
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1
# just, as a wheel from PyPI. git refuses a repository owned by another user, and a mounted folder
# is owned by the host's user, so every folder is declared safe
RUN uv tool install "rust-just==${JUST}" \
    && git config --system --add safe.directory '*'
WORKDIR /work
COPY pyproject.toml uv.lock .python-version ./
RUN UV_COMPILE_BYTECODE=1 uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked
# The step browser, when it runs in here; `docker run -p 8765:8765` makes it reachable from outside
EXPOSE 8765
CMD ["bash"]
