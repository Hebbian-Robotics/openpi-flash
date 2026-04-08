"""Modal deployment for openpi inference.

Serves an openpi policy over WebSocket using Modal's GPU infrastructure.
Supports both JAX and PyTorch checkpoints (auto-detected).

Usage:
    modal serve modal_app.py    # development (hot-reload)
    modal deploy modal_app.py   # production
"""

from typing import Any

import modal

app = modal.App("openpi-inference")

openpi_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    .pip_install("uv")
    # Layer 1: Copy only dependency metadata (changes rarely).
    .add_local_file("../openpi/pyproject.toml", "/build/openpi/pyproject.toml", copy=True)
    .add_local_file("../openpi/uv.lock", "/build/openpi/uv.lock", copy=True)
    .add_local_file(
        "../openpi/packages/openpi-client/pyproject.toml",
        "/build/openpi/packages/openpi-client/pyproject.toml",
        copy=True,
    )
    # Layer 2: Install dependencies only (cached unless pyproject.toml/uv.lock change).
    # Source is mounted via PYTHONPATH at runtime, so no need to install the project itself.
    .run_commands(
        "cd /build/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-install-workspace",
        "cd /build/openpi && uv pip install gsutil starlette pydantic",
    )
    # Layer 3: Copy transformers_replace patch source and apply it.
    .add_local_dir(
        "../openpi/src/openpi/models_pytorch/transformers_replace",
        "/build/transformers_replace",
        copy=True,
    )
    .run_commands(
        'TRANSFORMERS_DIR=$(/build/openpi/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname) && '
        "cp -r /build/transformers_replace/* $TRANSFORMERS_DIR/"
    )
    .env(
        {
            "OPENPI_DATA_HOME": "/model-cache",
            "PYTHONPATH": "/app/openpi-src:/app/openpi-client-src:/app/hosting-src",
            "VIRTUAL_ENV": "/build/openpi/.venv",
            "PATH": "/build/openpi/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .add_local_dir("src", "/app/hosting-src")
    .add_local_dir("../openpi/src", "/app/openpi-src")
    .add_local_dir("../openpi/packages/openpi-client/src", "/app/openpi-client-src")
)

model_weights_volume = modal.Volume.from_name("openpi-model-weights", create_if_missing=True)


@app.cls(
    image=openpi_image,
    gpu="L40S",
    region="ap",
    volumes={"/model-cache": model_weights_volume},
    scaledown_window=300,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class OpenPIInference:
    model_config_name: str = modal.parameter(default="pi05_aloha")
    checkpoint_dir: str = modal.parameter(default="gs://openpi-assets/checkpoints/pi05_base")
    model_version: str = modal.parameter(default="pi05_v1")
    default_prompt: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def load_model(self) -> None:
        import logging

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info(
            "Loading model: config=%s, checkpoint=%s",
            self.model_config_name,
            self.checkpoint_dir,
        )

        train_config = _config.get_config(self.model_config_name)
        self._policy = _policy_config.create_trained_policy(
            train_config,
            self.checkpoint_dir,
            default_prompt=self.default_prompt or None,
        )
        self._metadata = train_config.policy_metadata
        logger.info("Model loaded successfully")

    @modal.asgi_app()
    def serve(self) -> Any:
        from hosting.modal_asgi import create_openpi_asgi_app

        return create_openpi_asgi_app(
            policy=self._policy,
            metadata=self._metadata,
        )
