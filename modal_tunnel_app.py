"""Modal deployment using direct tunnel for lower-latency inference.

Uses modal.forward to expose a raw TCP tunnel that bypasses Modal's
web endpoint proxy. The tunnel address is stored in a Modal Dict
for client discovery.

Usage:
    # Deploy the server (persistent, scales to zero when idle)
    uv run modal deploy modal_tunnel_app.py

    # Start the server (triggers container startup + tunnel creation)
    uv run modal run modal_tunnel_app.py

    # Read the tunnel address from a client
    from modal import Dict
    tunnel_dict = Dict.from_name("openpi-tunnel-info")
    print(tunnel_dict["address"])  # -> ("host", port)
"""

import modal

app = modal.App("openpi-inference-tunnel")

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
            "TORCHINDUCTOR_CACHE_DIR": "/model-cache/torch_inductor_cache",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
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
tunnel_dict = modal.Dict.from_name("openpi-tunnel-info", create_if_missing=True)

WEBSOCKET_PORT = 8000


@app.function(
    image=openpi_image,
    gpu="L40S",
    region="ap",
    volumes={"/model-cache": model_weights_volume},
    timeout=86400,
)
def serve_tunnel(
    model_config_name: str = "pi05_aloha",
    checkpoint_dir: str = "/model-cache/pi05_base_pytorch",
    default_prompt: str = "",
) -> None:
    import shutil

    # Apply transformers patches before any model imports.
    # The build-time cp may be cached by Modal, so re-apply at runtime.
    import sys
    import threading
    from pathlib import Path

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

    from openpi.policies import policy_config as _policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as _config

    # Load model.
    print(f"Loading model: config={model_config_name}, checkpoint={checkpoint_dir}")
    train_config = _config.get_config(model_config_name)

    # Use default compile mode — reliable, compiles in ~2.5 min, gives ~76ms
    # policy forward vs ~160ms eager.
    import dataclasses

    from openpi.models import pi0_config as _pi0_config

    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode="default"),
        )

    # Log container location for debugging network latency.
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

    import time

    load_start = time.monotonic()
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt or None,
    )
    load_elapsed = time.monotonic() - load_start
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
    policy.infer(dummy_observation)
    compile_elapsed = time.monotonic() - compile_start
    print(f"Compilation done in {compile_elapsed:.1f}s")

    # Persist inductor cache so subsequent cold starts skip compilation.
    model_weights_volume.commit()
    print("Inductor cache committed to volume")

    # Start WebSocket server in a background thread.
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=WEBSOCKET_PORT,
        metadata=train_config.policy_metadata,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Expose via direct tunnel (bypasses Modal's web endpoint proxy).
    with modal.forward(WEBSOCKET_PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        # Resolve relay location for debugging.
        import socket

        try:
            relay_ip = socket.gethostbyname(host)
            with urllib.request.urlopen(f"https://ipinfo.io/{relay_ip}/json", timeout=5) as resp:
                relay_info = json.loads(resp.read())
            print(
                f"Relay location: {relay_info.get('city')}, {relay_info.get('region')} "
                f"({relay_info.get('country')}) — IP: {relay_ip}, org: {relay_info.get('org')}"
            )
        except Exception as e:
            print(f"Could not resolve relay location: {e}")

        # Store address in Modal Dict for client discovery.
        tunnel_dict["address"] = (host, port)
        tunnel_dict["url"] = f"ws://{host}:{port}"

        print(f"\nTunnel available at: ws://{host}:{port}")
        print("Test with: uv run python test_modal_tunnel.py")

        # Keep alive until timeout or killed.
        server_thread.join()
