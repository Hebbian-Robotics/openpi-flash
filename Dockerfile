# Dockerfile for the openpi-flash inference engine.
# Based on openpi's scripts/docker/serve_policy.Dockerfile
#
# Local dev (builds everything including openpi-flash-transport):
#   docker build .. -t openpi-flash -f Dockerfile
#
# CI base image (skips Rust, transport binary added later from CI artifact):
#   docker build .. -t openpi-flash-base -f Dockerfile --target base
#
# Run:
#   docker run --rm -it --gpus=all \
#     -v ./config.json:/config/config.json:ro \
#     -e INFERENCE_CONFIG_PATH=/config/config.json \
#     -p 8000:8000 -p 5555:5555/udp \
#     openpi-flash

FROM rust:1-slim-bookworm AS rust-builder

WORKDIR /build
COPY hosting/flash-transport/Cargo.toml /build/Cargo.toml
COPY hosting/flash-transport/src /build/src
RUN cargo build --release

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# System dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs linux-headers-generic build-essential clang \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Virtual environment outside project directory.
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"

# Install hosting dependencies using the hosting lockfile as the single source
# of truth. The hosting project depends on the sibling openpi checkout via
# path dependencies, so we recreate that relative layout under /build.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=hosting/uv.lock,target=/build/hosting/uv.lock \
    --mount=type=bind,source=hosting/pyproject.toml,target=/build/hosting/pyproject.toml \
    --mount=type=bind,source=openpi/LICENSE,target=/build/openpi/LICENSE \
    --mount=type=bind,source=openpi/README.md,target=/build/openpi/README.md \
    --mount=type=bind,source=openpi/pyproject.toml,target=/build/openpi/pyproject.toml \
    --mount=type=bind,source=openpi/src,target=/build/openpi/src \
    --mount=type=bind,source=openpi/packages/openpi-client/pyproject.toml,target=/build/openpi/packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=openpi/packages/openpi-client/src,target=/build/openpi/packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --project /build/hosting --frozen --no-install-project --no-dev

# Copy transformers_replace files (required for PyTorch models).
COPY openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Copy application code.
COPY openpi/src /app/openpi-src
COPY openpi/packages/openpi-client/src /app/openpi-client-src
COPY hosting/src /app/hosting-src
COPY hosting/main.py /app/main.py
ENV PYTHONPATH="/app/openpi-src:/app/openpi-client-src:/app/hosting-src"

# PyTorch inductor cache — persists within container lifetime (use a volume
# mount at /cache for persistence across restarts).
ENV TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1

# openpi data home for downloaded checkpoints and norm stats.
ENV OPENPI_DATA_HOME=/cache/models

# Limit JAX GPU memory when both slots are loaded (combined mode, JAX planner +
# PyTorch action co-resident). Harmless in single-slot modes — only takes
# effect when JAX actually allocates GPU memory.
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# Slot-specific ports:
#   action  : 8000/tcp (WebSocket), 5555/udp (QUIC)
#   planner : 8002/tcp (WebSocket), 5556/udp (QUIC), 8001/tcp (admin, loopback only)
EXPOSE 5555/udp
EXPOSE 5556/udp

CMD ["python", "main.py", "serve"]

# Final stage: adds the openpi-flash-transport binary to the base image.
# This is the default target for local dev builds.
FROM base AS final
COPY --from=rust-builder /build/target/release/openpi-flash-transport /usr/local/bin/openpi-flash-transport
