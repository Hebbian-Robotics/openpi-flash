"""Modal deployment for openpi inference.

Serves an openpi policy over WebSocket using Modal's GPU infrastructure.
Supports both JAX and PyTorch checkpoints (auto-detected).

Usage:
    modal serve modal_app.py    # development (hot-reload)
    modal deploy modal_app.py   # production
"""

import modal

app = modal.App("openpi-inference")

openpi_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    # Install openpi dependencies using its lockfile.
    .pip_install("uv")
    .copy_local_dir("../openpi", "/build/openpi", ignore=[".git", "__pycache__", "*.pyc"])
    .run_commands(
        "cd /build/openpi && uv venv --python 3.11 /opt/venv",
        "cd /build/openpi && GIT_LFS_SKIP_SMUDGE=1 /opt/venv/bin/python -m uv pip install .",
    )
    # Apply transformers_replace patch for PyTorch model support.
    .run_commands(
        'TRANSFORMERS_DIR=$(/opt/venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname) && '
        "cp -r /build/openpi/src/openpi/models_pytorch/transformers_replace/* $TRANSFORMERS_DIR/"
    )
    # Install hosting dependencies.
    .run_commands("/opt/venv/bin/python -m uv pip install starlette pydantic")
    .copy_local_dir("src", "/app/hosting-src")
    .copy_local_dir("../openpi/src", "/app/openpi-src")
    .env(
        {
            "PYTHONPATH": "/app/openpi-src:/app/hosting-src",
            "VIRTUAL_ENV": "/opt/venv",
            "PATH": "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
)

model_weights_volume = modal.Volume.from_name("openpi-model-weights", create_if_missing=True)


@app.cls(
    image=openpi_image,
    gpu="L4",
    volumes={"/model-cache": model_weights_volume},
    container_idle_timeout=300,
    allow_concurrent_inputs=1,
)
class OpenPIInference:
    model_config_name: str = modal.parameter(default="pi05_aloha")
    checkpoint_dir: str = modal.parameter(default="gs://openpi-assets/checkpoints/pi05_base")
    model_version: str = modal.parameter(default="pi05_v1")
    default_prompt: str | None = modal.parameter(default=None)

    @modal.enter()
    def load_model(self):
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
            default_prompt=self.default_prompt,
        )
        self._metadata = train_config.policy_metadata
        logger.info("Model loaded successfully")

    @modal.asgi_app()
    def serve(self):
        from hosting.modal_asgi import create_openpi_asgi_app

        return create_openpi_asgi_app(
            policy=self._policy,
            metadata=self._metadata,
        )
