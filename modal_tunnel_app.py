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
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base",
    default_prompt: str = "",
) -> None:
    import logging
    import threading

    from openpi.policies import policy_config as _policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as _config

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load model.
    logger.info("Loading model: config=%s, checkpoint=%s", model_config_name, checkpoint_dir)
    train_config = _config.get_config(model_config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt or None,
    )
    logger.info("Model loaded successfully")

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
        logger.info(f"Tunnel available at {host}:{port}")

        # Store address in Modal Dict for client discovery.
        tunnel_dict["address"] = (host, port)
        tunnel_dict["url"] = f"ws://{host}:{port}"

        print(f"\nTunnel available at: ws://{host}:{port}")
        print(f"Test with: uv run python test_modal_tunnel.py")

        # Keep alive until timeout or killed.
        server_thread.join()
