"""Smoke test for the Modal QUIC portal deployment.

Connects via QUIC portal using the shared Modal Dict and runs benchmark inferences.
"""

from modal import Dict

from hosting.benchmark import run_benchmark
from hosting.modal_dict_names import OPENPI_MODAL_QUIC_INFO_DICT_NAME
from hosting.quic_client_policy import QuicClientPolicy
from tests.helpers import random_observation_aloha


def run(mode: str | None = None) -> None:
    print(f"Connecting via QUIC portal (Dict: '{OPENPI_MODAL_QUIC_INFO_DICT_NAME}') ...")
    quic_dict = Dict.from_name(OPENPI_MODAL_QUIC_INFO_DICT_NAME)

    policy = QuicClientPolicy(portal_dict=quic_dict)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, lambda: random_observation_aloha(mode=mode))
    result.print_summary()

    policy.close()
