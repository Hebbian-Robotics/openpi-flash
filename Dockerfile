# Dockerfile for the hosted inference service.
# Based on openpi's scripts/docker/serve_policy.Dockerfile
#
# Build (from ~/openpi/hosting/):
#   docker build .. -t openpi-hosted -f Dockerfile
#
# Run:
#   docker run --rm -it --gpus=all \
#     -v ./config.json:/config/config.json:ro \
#     -e INFERENCE_CONFIG_PATH=/config/config.json \
#     -p 8000:8000 -p 5555:5555/udp \
#     openpi-hosted

FROM rust:1.88-bookworm AS quic-sidecar-builder

WORKDIR /build
COPY hosting/quic-sidecar/Cargo.toml /build/Cargo.toml
COPY hosting/quic-sidecar/src /build/src
RUN cargo build --release

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
COPY --from=quic-sidecar-builder /build/target/release/openpi-quic-sidecar /usr/local/bin/openpi-quic-sidecar

WORKDIR /app

# System dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs linux-headers-generic build-essential clang \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Virtual environment outside project directory.
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"

# Install openpi dependencies using its lockfile.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=openpi/uv.lock,target=uv.lock \
    --mount=type=bind,source=openpi/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=openpi/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=openpi/packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Install hosting dependencies.
# - pydantic: config validation
# - gsutil: download checkpoints from gs://openpi-assets (openpi's download
#   module shells out to gsutil for this bucket; without it falls back to gcsfs
#   which requires explicit GCP credentials)
# - s3fs: fsspec S3 backend so maybe_download() handles s3:// checkpoint URLs
#   (EC2 instances authenticate via IAM instance profile, no credentials needed)
RUN uv pip install pydantic gsutil s3fs "quic-portal @ git+https://github.com/Hebbian-Robotics/quic-portal.git" pytest

# Copy transformers_replace files (required for PyTorch models).
COPY openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Copy application code.
COPY openpi/src /app/openpi-src
COPY openpi/packages/openpi-client/src /app/openpi-client-src
COPY hosting/src /app/hosting-src
ENV PYTHONPATH="/app/openpi-src:/app/openpi-client-src:/app/hosting-src"

# PyTorch inductor cache — persists within container lifetime (use a volume
# mount at /cache for persistence across restarts).
ENV TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor
ENV TORCHINDUCTOR_FX_GRAPH_CACHE=1

# openpi data home for downloaded checkpoints and norm stats.
ENV OPENPI_DATA_HOME=/cache/models
EXPOSE 5555/udp

CMD ["python", "-m", "hosting.serve"]
