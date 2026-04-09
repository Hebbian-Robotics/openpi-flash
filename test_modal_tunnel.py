"""Quick smoke test for the Modal tunnel deployment.

Reads the tunnel address from Modal Dict and runs a test inference.

Usage:
    uv run python test_modal_tunnel.py
"""

import sys
import time

from modal import Dict
from openpi_client import websocket_client_policy as _websocket_client_policy

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha


def main() -> None:
    # Read tunnel address from Modal Dict.
    print("Reading tunnel address from Modal Dict ...")
    tunnel_dict = Dict.from_name("openpi-tunnel-info")
    ws_url = tunnel_dict["url"]
    print(f"Tunnel address: {ws_url}")

    # Connect WebSocket (unencrypted, direct tunnel).
    print(f"\nConnecting to {ws_url} ...")
    connect_start = time.monotonic()
    policy = _websocket_client_policy.WebsocketClientPolicy(host=ws_url)
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
