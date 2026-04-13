"""Smoke test for the direct QUIC transport (Docker/EC2 deployment).

Connects directly to a QUIC server and runs benchmark inferences.
"""

from hosting.benchmark import run_benchmark
from hosting.direct_quic_client_policy import DirectQuicClientPolicy
from tests.helpers import random_observation_aloha, wait_for_server

DEFAULT_QUIC_PORT = 5555
DEFAULT_WS_PORT = 8000


def run(host: str, quic_port: int = DEFAULT_QUIC_PORT, ws_port: int = DEFAULT_WS_PORT) -> None:
    health_url = f"http://{host}:{ws_port}/healthz"
    wait_for_server(health_url)

    print(f"\nConnecting to {host}:{quic_port} via QUIC...")
    policy = DirectQuicClientPolicy(host=host, port=quic_port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, random_observation_aloha)
    result.print_summary()

    policy.close()
