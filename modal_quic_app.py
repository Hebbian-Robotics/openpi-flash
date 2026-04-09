"""Modal deployment using QUIC portal for lowest-latency inference.

Uses quic-portal for direct peer-to-peer QUIC transport with automatic
NAT traversal via STUN + UDP hole punching. The connection is coordinated
through a shared Modal Dict — no URL or port to manage manually.

This provides lower latency than both the ASGI variant (HTTP proxy overhead)
and the TCP tunnel variant (TCP head-of-line blocking).

Note: quic-portal is experimental. NAT traversal only works with "easy" NATs.
Fall back to modal_tunnel_app.py if connectivity issues arise.

Usage:
    # Deploy the server
    uv run modal deploy modal_quic_app.py

    # Start the server (triggers container startup + QUIC portal creation)
    uv run modal run modal_quic_app.py

    # Client connects via the shared Modal Dict (no URL needed)
    from modal import Dict
    from hosting.quic_client_policy import QuicClientPolicy
    quic_dict = Dict.from_name("openpi-quic-info")
    policy = QuicClientPolicy(portal_dict=quic_dict)
"""

import modal

app = modal.App("openpi-inference-quic")

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
        "cd /build/openpi && uv pip install gsutil starlette pydantic quic-portal",
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
quic_dict = modal.Dict.from_name("openpi-quic-info", create_if_missing=True)

QUIC_PORT = 5555


@app.function(
    image=openpi_image,
    gpu="L40S",
    region="ap",
    volumes={"/model-cache": model_weights_volume},
    timeout=86400,
)
def serve_quic(
    model_config_name: str = "pi05_aloha",
    checkpoint_dir: str = "/model-cache/pi05_base_pytorch",
    default_prompt: str = "",
) -> None:
    import dataclasses
    import logging
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
    for pycache_dir in Path(transformers_dir, "models").glob("*/__pycache__"):
        shutil.rmtree(pycache_dir)

    # Evict cached transformers submodules so Python reloads from the patched files.
    patched_modules = [key for key in sys.modules if key.startswith("transformers.models.")]
    for mod_name in patched_modules:
        del sys.modules[mod_name]

    from openpi.models import pi0_config as _pi0_config
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from hosting.quic_server import QuicPolicyServer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load model.
    logger.info("Loading model: config=%s, checkpoint=%s", model_config_name, checkpoint_dir)
    train_config = _config.get_config(model_config_name)

    # Workaround for PyTorch checkpoints:
    # 1. Disable torch.compile — max-autotune crashes on mixed fp32/bf16 matmuls.
    # 2. Skip to_bfloat16_for_selected_params — it keeps layernorm weights in
    #    float32 while the patched siglip casts hidden states to bfloat16, causing
    #    dtype mismatches. The checkpoint is already in the correct precision.
    is_pytorch = (Path(checkpoint_dir) / "model.safetensors").exists()
    if isinstance(train_config.model, _pi0_config.Pi0Config) and is_pytorch:
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )

        from openpi.models_pytorch import gemma_pytorch

        gemma_pytorch.PaliGemmaWithExpertModel.to_bfloat16_for_selected_params = (
            lambda self, *a, **kw: None
        )
        logger.info(
            "Applied PyTorch inference workarounds (disabled torch.compile and selective precision)"
        )

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt or None,
    )
    logger.info("Model loaded successfully")

    # Serve over QUIC portal (blocks forever, handles reconnections).
    print(f"\nQUIC portal server ready (Dict: 'openpi-quic-info', port: {QUIC_PORT})")
    print("Test with: uv run python test_modal_quic.py")

    server = QuicPolicyServer(
        policy=policy,
        portal_dict=quic_dict,
        metadata=train_config.policy_metadata,
        local_port=QUIC_PORT,
    )
    server.serve_forever()
