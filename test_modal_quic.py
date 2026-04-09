"""Quick smoke test for the Modal QUIC portal deployment.

Connects via QUIC portal using the shared Modal Dict and runs benchmark inferences.

Usage:
    uv run python test_modal_quic.py
"""

import sys

# Reuse the random observation generator from the original repo.
sys.path.insert(0, "../openpi/examples/simple_client")
from main import _random_observation_aloha


def main() -> None:
    from modal import Dict

    from hosting.benchmark import run_benchmark
    from hosting.quic_client_policy import QuicClientPolicy

    # Connect via QUIC portal (NAT traversal coordinated via Modal Dict).
    print("Connecting via QUIC portal (Dict: 'openpi-quic-info') ...")
    quic_dict = Dict.from_name("openpi-quic-info")

    policy = QuicClientPolicy(portal_dict=quic_dict)
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(policy, _random_observation_aloha)
    result.print_summary()

    policy.close()


if __name__ == "__main__":
    main()
