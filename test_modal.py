"""Quick smoke test for the Modal deployment.

Sends a random ALOHA observation to the server and prints timing breakdown.

Usage:
    uv run python test_modal.py wss://your-modal-url
"""

import sys
import time

import requests
from openpi_client import websocket_client_policy as _websocket_client_policy

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
    connect_start = time.monotonic()
    policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"Connected in {1000 * (time.monotonic() - connect_start):.0f}ms")
    print(f"Server metadata: {policy.get_server_metadata()}")

    # Warmup (first inference includes JAX compilation, not representative).
    print("\nWarmup (1 inference, includes JAX compilation) ...")
    warmup_start = time.monotonic()
    policy.infer(_random_observation_aloha())
    print(f"Warmup done in {1000 * (time.monotonic() - warmup_start):.0f}ms")

    # Actual test inference.
    print("\nRunning inference ...")
    start = time.monotonic()
    action = policy.infer(_random_observation_aloha())
    client_ms = 1000 * (time.monotonic() - start)

    print(f"\nAction shape: {action['actions'].shape}")
    print()
    print("Timing breakdown:")
    print(f"  Client round trip:  {client_ms:.0f}ms  (total time measured on this machine)")
    for key, value in action.get("policy_timing", {}).items():
        print(f"  Policy {key}:  {value:.0f}ms  (model forward pass on GPU)")
    for key, value in action.get("server_timing", {}).items():
        if key == "infer_ms":
            print(f"  Server {key}:   {value:.0f}ms  (policy.infer on server, includes pre/post processing)")
        elif key == "prev_total_ms":
            print(f"  Server {key}: {value:.0f}ms  (server's total time for previous request)")
    if "server_timing" in action and "policy_timing" in action:
        server_ms = action["server_timing"].get("infer_ms", 0)
        policy_ms = action["policy_timing"].get("infer_ms", 0)
        print(f"\n  Network overhead:   {client_ms - server_ms:.0f}ms  (client round trip - server infer)")
        print(f"  Server overhead:    {server_ms - policy_ms:.0f}ms  (server infer - model forward pass)")


if __name__ == "__main__":
    main()
