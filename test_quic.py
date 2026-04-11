"""Smoke test for the direct QUIC transport (Docker/EC2 deployment).

Connects directly to a QUIC server and runs benchmark inferences.
No Modal Dict, no STUN, no relay — just a direct UDP connection.

Usage:
    uv run python test_quic.py <host>
    uv run python test_quic.py <host> --quic-port 5555 --ws-port 8000
"""

import argparse
import sys
import time

import requests

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha

from hosting.benchmark import run_benchmark
from hosting.direct_quic_client_policy import DirectQuicClientPolicy

DEFAULT_QUIC_PORT = 5555
DEFAULT_WS_PORT = 8000


def _wait_for_server(host: str, ws_port: int) -> None:
    """Poll the WebSocket server's /healthz endpoint until it's ready.

    Both transports share the same policy load, so once the WebSocket
    health check passes the QUIC server is ready too.
    """
    health_url = f"http://{host}:{ws_port}/healthz"
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
    parser = argparse.ArgumentParser(description="Smoke test for direct QUIC transport")
    parser.add_argument("host", help="Server host (IP or hostname)")
    parser.add_argument(
        "--quic-port", type=int, default=DEFAULT_QUIC_PORT, help="Server QUIC/UDP port"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=DEFAULT_WS_PORT,
        help="Server WebSocket/TCP port (for health check)",
    )
    args = parser.parse_args()

    _wait_for_server(args.host, args.ws_port)

    print(f"\nConnecting to {args.host}:{args.quic_port} via QUIC...")
    policy = DirectQuicClientPolicy(host=args.host, port=args.quic_port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, _random_observation_aloha)
    result.print_summary()

    policy.close()


if __name__ == "__main__":
    main()
