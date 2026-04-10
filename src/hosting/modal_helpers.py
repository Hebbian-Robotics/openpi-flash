"""Shared helpers for Modal deployments.

Contains the image builder and runtime helpers (transformers patching,
location logging, compile mode, model loading, warmup) shared by all
three Modal app variants (ASGI, tunnel, QUIC).

Image builder: imported at module level in modal apps (only depends on `modal`).
Runtime helpers: imported inside function bodies where heavy deps are available.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Shared Modal infrastructure settings — single source of truth for all apps.
# ---------------------------------------------------------------------------
GPU_TYPE = "L40S"
REGION = "ap"
MODEL_WEIGHTS_VOLUME_NAME = "openpi-model-weights"
MODEL_CACHE_MOUNT_PATH = "/model-cache"
DEFAULT_MODEL_CONFIG_NAME = "pi05_aloha"
DEFAULT_CHECKPOINT_DIR = f"{MODEL_CACHE_MOUNT_PATH}/pi05_base_pytorch"
WEBSOCKET_PORT = 8000
QUIC_PORT = 5555

model_weights_volume = modal.Volume.from_name(MODEL_WEIGHTS_VOLUME_NAME, create_if_missing=True)

# Container paths set by the image builder.
OPENPI_SRC_DIR = "/app/openpi-src"
HOSTING_SRC_DIR = "/app/hosting-src"
OPENPI_CLIENT_SRC_DIR = "/app/openpi-client-src"
TRANSFORMERS_PATCH_DIR = f"{OPENPI_SRC_DIR}/openpi/models_pytorch/transformers_replace"


def create_openpi_image(extra_pip_packages: list[str] | None = None) -> modal.Image:
    """Build the standard Modal image for openpi inference.

    All three deployment variants share the same base image. The only difference
    is extra pip packages (e.g. quic-portal for the QUIC variant).
    """
    base_pip_packages = ["gsutil", "starlette", "pydantic"]
    all_pip_packages = base_pip_packages + (extra_pip_packages or [])
    pip_install_command = f"cd /build/openpi && uv pip install {' '.join(all_pip_packages)}"

    return (
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
            pip_install_command,
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
                "TORCHINDUCTOR_CACHE_DIR": "/model-cache/torch_inductor_cache",
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "PYTHONPATH": f"{OPENPI_SRC_DIR}:{OPENPI_CLIENT_SRC_DIR}:{HOSTING_SRC_DIR}",
                "VIRTUAL_ENV": "/build/openpi/.venv",
                "PATH": "/build/openpi/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            }
        )
        .add_local_dir("src", HOSTING_SRC_DIR)
        .add_local_dir("../openpi/src", OPENPI_SRC_DIR)
        .add_local_dir("../openpi/packages/openpi-client/src", OPENPI_CLIENT_SRC_DIR)
    )


# ---------------------------------------------------------------------------
# Runtime helpers — import these inside @modal.enter / @app.function bodies.
# Heavy imports (torch, transformers, openpi) are deferred to function bodies.
# ---------------------------------------------------------------------------


def apply_transformers_patches() -> None:
    """Re-apply transformers patches at runtime.

    The build-time cp may be cached by Modal, so we re-apply at runtime.
    Also clears stale .pyc bytecode and evicts cached modules so Python
    reloads from the patched files.
    """
    import shutil
    import sys
    from pathlib import Path

    import transformers

    transformers_dir = str(transformers.__path__[0])
    shutil.copytree(TRANSFORMERS_PATCH_DIR, transformers_dir, dirs_exist_ok=True)

    # Delete stale .pyc bytecode caches so Python recompiles from the patched .py files.
    # shutil.copytree preserves source timestamps, which can be older than the .pyc
    # files generated during the image build, causing Python to use the stale bytecache.
    for pycache_dir in Path(transformers_dir, "models").glob("*/__pycache__"):
        shutil.rmtree(pycache_dir)

    # Evict cached transformers submodules so Python reloads from the patched files.
    patched_modules = [key for key in sys.modules if key.startswith("transformers.models.")]
    for mod_name in patched_modules:
        del sys.modules[mod_name]


def log_container_location() -> None:
    """Log the container's geographic location via ipinfo.io.

    Useful for debugging network latency when the container lands in
    an unexpected region.
    """
    import json
    import urllib.request

    try:
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=5) as resp:
            ip_info = json.loads(resp.read())
        print(
            f"Container location: {ip_info.get('city')}, {ip_info.get('region')} "
            f"({ip_info.get('country')}) — IP: {ip_info.get('ip')}, org: {ip_info.get('org')}"
        )
    except Exception as e:
        print(f"Could not determine container location: {e}")


def load_openpi_policy(
    model_config_name: str,
    checkpoint_dir: str,
    default_prompt: str = "",
) -> tuple:
    """Load an openpi policy with all necessary patches and workarounds.

    Sequence: transformers patches → location log → config → compile mode →
    model load (timed) → warmup inference (timed).

    Returns:
        Tuple of (policy, train_config) so callers can access policy_metadata
        and other config attributes.
    """
    import dataclasses
    import time

    import numpy as np

    apply_transformers_patches()
    log_container_location()

    from openpi.models import pi0_config as _pi0_config
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    print(f"Loading model: config={model_config_name}, checkpoint={checkpoint_dir}")
    train_config = _config.get_config(model_config_name)

    # Use default compile mode — reliable, compiles in ~2.5 min, gives ~76ms
    # policy forward vs ~160ms eager.
    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode="default"),
        )

    load_start = time.monotonic()
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt or None,
    )
    load_elapsed = time.monotonic() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s")

    # Warmup inference to trigger torch.compile and populate inductor cache.
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
    policy.infer(dummy_observation)
    compile_elapsed = time.monotonic() - compile_start
    print(f"Compilation done in {compile_elapsed:.1f}s")

    return policy, train_config
