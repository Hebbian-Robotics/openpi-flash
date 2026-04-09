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
#     -p 8000:8000 \
#     openpi-hosted

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app

# System dependencies.
RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# Copy from the cache instead of linking since it's a mounted volume.
ENV UV_LINK_MODE=copy

# Virtual environment outside project directory.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install openpi dependencies using its lockfile.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=openpi/uv.lock,target=uv.lock \
    --mount=type=bind,source=openpi/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=openpi/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=openpi/packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Install hosting dependencies.
RUN uv pip install pydantic

# Copy transformers_replace files (required for PyTorch models).
COPY openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Copy application code.
COPY openpi/src /app/openpi-src
COPY hosting/src /app/hosting-src
ENV PYTHONPATH="/app/openpi-src:/app/hosting-src"

CMD ["python", "-m", "hosting.serve"]
