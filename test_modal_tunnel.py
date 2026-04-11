"""Quick smoke test for the Modal tunnel deployment.

Reads the tunnel address from Modal Dict and runs benchmark inferences.

Usage:
    uv run python test_modal_tunnel.py
"""

import sys
import time

import requests
from modal import Dict
from openpi_client import websocket_client_policy as _websocket_client_policy

from hosting.benchmark import run_benchmark

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha


def _wait_for_server(ws_url: str) -> None:
    """Poll the server's /healthz endpoint until it's ready."""
    # Derive HTTP URL from WebSocket URL (ws://host:port -> http://host:port).
    health_url = ws_url.replace("wss://", "https://").replace("ws://", "http://") + "/healthz"
    print(f"Health check {health_url} (waiting for server to be ready) ...")
    while True:
        try:
            response = requests.get(health_url, timeout=30)
            if response.ok:
                print(f"Health check: {response.text.strip()}")
                return
            print(f"  Server returned {response.status_code}, retrying ...")
        except requests.RequestException as e:
            print(f"  {e}, retrying ...")
        time.sleep(5)


def main() -> None:
    # Read tunnel address from Modal Dict.
    print("Reading tunnel address from Modal Dict ...")
    tunnel_dict = Dict.from_name("openpi-tunnel-info")
    ws_url = tunnel_dict["url"]
    print(f"Tunnel address: {ws_url}")

    _wait_for_server(ws_url)

    # Connect WebSocket (unencrypted, direct tunnel).
    print(f"\nConnecting to {ws_url} ...")
    policy = _websocket_client_policy.WebsocketClientPolicy(host=ws_url)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, _random_observation_aloha)
    result.print_summary()


if __name__ == "__main__":
    main()
