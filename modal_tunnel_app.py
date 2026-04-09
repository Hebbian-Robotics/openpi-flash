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

from hosting.modal_helpers import create_openpi_image

app = modal.App("openpi-inference-tunnel")

openpi_image = create_openpi_image()

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
    import json
    import socket
    import threading
    import urllib.request

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    from hosting.modal_helpers import load_openpi_policy

    # Open tunnel early to check relay location before spending time on model loading.
    # modal.forward doesn't need the server to be listening yet.
    with modal.forward(WEBSOCKET_PORT, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket

        # Resolve relay location.
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

        # Now load model + compile (the expensive part).
        policy, train_config = load_openpi_policy(
            model_config_name,
            checkpoint_dir,
            default_prompt,
        )

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

        # Store address in Modal Dict for client discovery.
        tunnel_dict["address"] = (host, port)
        tunnel_dict["url"] = f"ws://{host}:{port}"

        print(f"\nTunnel available at: ws://{host}:{port}")
        print("Test with: uv run python test_modal_tunnel.py")

        # Keep alive until timeout or killed.
        server_thread.join()
