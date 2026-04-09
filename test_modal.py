"""Quick smoke test for the Modal ASGI deployment.

Sends random ALOHA observations to the server and prints benchmark results.

Usage:
    uv run python test_modal.py wss://your-modal-url
"""

import sys
import time

import requests
from openpi_client import websocket_client_policy as _websocket_client_policy

from hosting.benchmark import run_benchmark

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python test_modal.py wss://your-modal-url")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 443

    # Health check (retries until server is ready, handles cold start).
    https_host = host.removeprefix("wss://").removeprefix("ws://")
    health_url = f"https://{https_host}/healthz"
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
    print(f"\nConnecting to {host}:{port} ...")
    policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, _random_observation_aloha)
    result.print_summary()


if __name__ == "__main__":
    main()
