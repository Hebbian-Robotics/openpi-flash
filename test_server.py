"""Quick smoke test for any WebSocket deployment (EC2, Docker, Modal ASGI).

Sends random ALOHA observations to the server and prints benchmark results.

Usage:
    uv run python test_server.py ws://localhost:8000          # EC2/Docker (plain)
    uv run python test_server.py wss://your-domain            # EC2 with HTTPS
    uv run python test_server.py wss://your-modal-url         # Modal
"""

import sys
import time
import urllib.parse
from typing import cast

import numpy as np
import requests
from openpi_client import websocket_client_policy as _websocket_client_policy

from hosting.benchmark import InferablePolicy, run_benchmark


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python test_server.py ws://host:port  OR  wss://host")
        sys.exit(1)

    requested_server_url = sys.argv[1]
    parsed_server_url = urllib.parse.urlparse(requested_server_url)
    if parsed_server_url.scheme not in {"ws", "wss"} or parsed_server_url.hostname is None:
        raise ValueError(
            "Expected first argument to be a WebSocket URL like ws://host[:port] or wss://host[:port]."
        )

    is_secure = parsed_server_url.scheme == "wss"
    default_port = 443 if is_secure else 8000
    effective_port = (
        int(sys.argv[2]) if len(sys.argv) > 2 else (parsed_server_url.port or default_port)
    )
    websocket_url = f"{parsed_server_url.scheme}://{parsed_server_url.hostname}"

    # Health check (retries until server is ready, handles cold start).
    health_scheme = "https" if is_secure else "http"
    health_url = f"{health_scheme}://{parsed_server_url.hostname}:{effective_port}/healthz"
    print(f"Health check {health_url} (waiting for server to be ready) ...")
    while True:
        try:
            response = requests.get(health_url, timeout=30)
            if response.ok:
                print(f"Health check: {response.text.strip()}")
                break
            print(f"  Server returned {response.status_code}, retrying ...")
        except requests.RequestException as e:
            print(f"  {e}, retrying ...")
        time.sleep(5)

    # Connect WebSocket.
    print(f"\nConnecting to {websocket_url}:{effective_port} ...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=websocket_url,
        port=effective_port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(cast(InferablePolicy, policy), _random_observation_aloha)
    result.print_summary()


if __name__ == "__main__":
    main()
