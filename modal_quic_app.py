"""Modal deployment using QUIC portal for lowest-latency inference.

Uses quic-portal for direct peer-to-peer QUIC transport with automatic
NAT traversal via STUN + UDP hole punching. The connection is coordinated
through a shared Modal Dict — no URL or port to manage manually.

This provides lower latency than both the ASGI variant (HTTP proxy overhead)
and the TCP tunnel variant (TCP head-of-line blocking).

Note: quic-portal is experimental. NAT traversal only works with "easy" NATs.
Fall back to modal_tunnel_app.py if connectivity issues arise.

Usage:
    # Deploy the server
    uv run modal deploy modal_quic_app.py

    # Start the server (triggers container startup + QUIC portal creation)
    uv run modal run modal_quic_app.py

    # Client connects via the shared Modal Dict (no URL needed)
    from modal import Dict
    from hosting.modal_dict_names import OPENPI_MODAL_QUIC_INFO_DICT_NAME
    from hosting.quic_client_policy import QuicClientPolicy
    quic_dict = Dict.from_name(OPENPI_MODAL_QUIC_INFO_DICT_NAME)
    policy = QuicClientPolicy(portal_dict=quic_dict)
"""

import os

import modal

from hosting.modal_dict_names import OPENPI_MODAL_QUIC_INFO_DICT_NAME
from hosting.modal_helpers import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MODEL_CONFIG_NAME,
    GPU_TYPE,
    MODEL_CACHE_MOUNT_PATH,
    QUIC_PORT,
    REGION,
    create_openpi_image,
    model_weights_volume,
)

app = modal.App("openpi-inference-quic")

openpi_image = create_openpi_image(
    extra_pip_packages=["quic-portal @ git+https://github.com/Hebbian-Robotics/quic-portal.git"]
)

quic_dict = modal.Dict.from_name(OPENPI_MODAL_QUIC_INFO_DICT_NAME, create_if_missing=True)

# Inject QUIC_RELAY_IP into the container environment so the server can
# fall back to the relay when direct hole punching fails.
relay_secret = modal.Secret.from_dotenv()

RELAY_PORT = 4433
QUIC_RELAY_IP_ENV_VAR = "QUIC_RELAY_IP"
QUIC_RELAY_ONLY_ENV_VAR = "QUIC_RELAY_ONLY"
TRUTHY_ENV_VAR_VALUES = ("1", "true", "yes")


def _get_relay_addr() -> tuple[str, int] | None:
    """Read QUIC_RELAY_IP from environment. Returns None if unset."""
    relay_ip = os.environ.get(QUIC_RELAY_IP_ENV_VAR)
    if not relay_ip:
        return None
    return (relay_ip.strip(), RELAY_PORT)


def _is_relay_only() -> bool:
    """Check if QUIC_RELAY_ONLY is set to skip hole punching entirely."""
    return os.environ.get(QUIC_RELAY_ONLY_ENV_VAR, "").strip().lower() in TRUTHY_ENV_VAR_VALUES


@app.function(
    image=openpi_image,
    gpu=GPU_TYPE,
    region=REGION,
    volumes={MODEL_CACHE_MOUNT_PATH: model_weights_volume},
    secrets=[relay_secret],
    timeout=86400,
)
def serve_quic(
    model_config_name: str = DEFAULT_MODEL_CONFIG_NAME,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    default_prompt: str = "",
) -> None:
    from hosting.modal_helpers import load_openpi_policy
    from hosting.quic_server import QuicPolicyServer

    policy, train_config = load_openpi_policy(
        model_config_name,
        checkpoint_dir,
        default_prompt,
    )

    # Persist inductor cache so subsequent cold starts skip compilation.
    model_weights_volume.commit()
    print("Inductor cache committed to volume")

    # Clear stale coordination data from previous runs so the server doesn't
    # immediately try to punch NAT with a non-existent client.
    quic_dict.clear()

    relay_addr = _get_relay_addr()
    relay_only = _is_relay_only()

    # Resolve relay location for debugging network latency.
    if relay_addr:
        from hosting.modal_helpers import log_ip_location

        log_ip_location("Relay", relay_addr[0])

    # Serve over QUIC portal (blocks forever, handles reconnections).
    print(
        f"\nQUIC portal server ready (Dict: '{OPENPI_MODAL_QUIC_INFO_DICT_NAME}', "
        f"port: {QUIC_PORT})"
    )
    if relay_only and relay_addr:
        print(f"Relay-only mode: {relay_addr[0]}:{relay_addr[1]}")
    elif relay_addr:
        print(f"Relay fallback: {relay_addr[0]}:{relay_addr[1]}")
    else:
        print("No relay configured — direct hole punch only")
    print("Test with: uv run python main.py test modal-quic")

    server = QuicPolicyServer(
        policy=policy,
        portal_dict=quic_dict,
        metadata=train_config.policy_metadata,
        local_port=QUIC_PORT,
        relay_addr=relay_addr,
        relay_only=relay_only,
    )
    server.serve_forever()
