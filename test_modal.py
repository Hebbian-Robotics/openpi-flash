"""Quick smoke test for the Modal deployment.

Usage:
    uv run python test_modal.py wss://your-modal-url
"""

import sys
import time

import requests
from openpi_client import websocket_client_policy as _websocket_client_policy

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha  # resolved via sys.path at runtime


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python test_modal.py wss://your-modal-url")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 443

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

    print(f"Connecting to {host}:{port} ...")
    connect_start = time.time()
    policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Connected in {1000 * (time.time() - connect_start):.0f}ms")
    print(f"Server metadata: {policy.get_server_metadata()}")

    print("Running inference ...")
    start = time.time()
    action = policy.infer(_random_observation_aloha())
    elapsed_ms = 1000 * (time.time() - start)

    print(f"\nAction shape: {action['actions'].shape}")
    print(f"Client round trip: {elapsed_ms:.0f}ms")
    for key, value in action.get("server_timing", {}).items():
        print(f"Server {key}: {value:.0f}ms")
    for key, value in action.get("policy_timing", {}).items():
        print(f"Policy {key}: {value:.0f}ms")


if __name__ == "__main__":
    main()
