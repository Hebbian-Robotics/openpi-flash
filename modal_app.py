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
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "clang",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
    )
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
        "cp -r /build/transformers_replace/* $TRANSFORMERS_DIR/ && "
        "echo 'transformers patched successfully'"
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
)
@modal.concurrent(max_inputs=1)
class OpenPIInference:
    model_config_name: str = modal.parameter(default="pi05_aloha")
    checkpoint_dir: str = modal.parameter(default="/model-cache/pi05_base_pytorch")
    model_version: str = modal.parameter(default="pi05_v1")
    default_prompt: str = modal.parameter(default="")

    @modal.enter(snap=False)
    def load_model(self) -> None:
        import shutil
        import sys
        from pathlib import Path

        # Apply transformers patches before any model imports.
        # The build-time cp may be cached by Modal, so re-apply at runtime.
        import transformers

        transformers_dir = str(transformers.__path__[0])
        patch_source = "/app/openpi-src/openpi/models_pytorch/transformers_replace"
        shutil.copytree(patch_source, transformers_dir, dirs_exist_ok=True)

        # Delete stale .pyc bytecode caches so Python recompiles from the patched .py files.
        # shutil.copytree preserves source timestamps, which can be older than the .pyc
        # files generated during the image build, causing Python to use the stale bytecache.
        for pycache_dir in Path(transformers_dir, "models").glob("*/__pycache__"):
            shutil.rmtree(pycache_dir)

        # Evict cached transformers submodules so Python reloads from the patched files.
        patched_modules = [key for key in sys.modules if key.startswith("transformers.models.")]
        for mod_name in patched_modules:
            del sys.modules[mod_name]

        # Log container location for debugging network latency.
        import json
        import urllib.request

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        try:
            with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as resp:
                ip_info = json.loads(resp.read())
            print(
                f"Container location: {ip_info.get('city')}, {ip_info.get('region')} "
                f"({ip_info.get('country')}) — IP: {ip_info.get('ip')}, org: {ip_info.get('org')}"
            )
        except Exception as e:
            print(f"Could not determine container location: {e}")

        print(f"Loading model: config={self.model_config_name}, checkpoint={self.checkpoint_dir}")

        train_config = _config.get_config(self.model_config_name)

        # Use default compile mode — reliable, compiles in ~2.5 min, gives ~76ms
        # policy forward vs ~160ms eager.
        import dataclasses

        from openpi.models import pi0_config as _pi0_config

        if isinstance(train_config.model, _pi0_config.Pi0Config):
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, pytorch_compile_mode="default"),
            )

        import time

        load_start = time.monotonic()
        self._policy = _policy_config.create_trained_policy(
            train_config,
            self.checkpoint_dir,
            default_prompt=self.default_prompt or None,
        )
        load_elapsed = time.monotonic() - load_start
        self._metadata = train_config.policy_metadata
        print(f"Model loaded in {load_elapsed:.1f}s")

        import numpy as np

        dummy_observation = {
            "state": np.ones((14,)),
            "images": {
                "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            },
            "prompt": "do something",
        }
        print("Compiling model (warmup inference) ...")
        compile_start = time.monotonic()
        self._policy.infer(dummy_observation)
        compile_elapsed = time.monotonic() - compile_start
        print(f"Compilation done in {compile_elapsed:.1f}s")

    @modal.asgi_app()
    def serve(self) -> Any:
        from hosting.modal_asgi import create_openpi_asgi_app

        return create_openpi_asgi_app(
            policy=self._policy,
            metadata=self._metadata,
        )
