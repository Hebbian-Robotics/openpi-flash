"""Smoke test for the Modal QUIC portal deployment.

Connects via QUIC portal using the shared Modal Dict and runs benchmark inferences.
"""

from modal import Dict

from hosting.benchmark import run_benchmark
from hosting.quic_client_policy import QuicClientPolicy
from tests.helpers import random_observation_aloha


def run() -> None:
    print("Connecting via QUIC portal (Dict: 'openpi-quic-info') ...")
    quic_dict = Dict.from_name("openpi-quic-info")

    policy = QuicClientPolicy(portal_dict=quic_dict)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, random_observation_aloha)
    result.print_summary()

    policy.close()
