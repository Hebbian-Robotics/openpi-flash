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
    from hosting.quic_client_policy import QuicClientPolicy
    quic_dict = Dict.from_name("openpi-quic-info")
    policy = QuicClientPolicy(portal_dict=quic_dict)
"""

import modal

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

openpi_image = create_openpi_image(extra_pip_packages=["quic-portal"])

quic_dict = modal.Dict.from_name("openpi-quic-info", create_if_missing=True)


@app.function(
    image=openpi_image,
    gpu=GPU_TYPE,
    region=REGION,
    volumes={MODEL_CACHE_MOUNT_PATH: model_weights_volume},
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

    # Serve over QUIC portal (blocks forever, handles reconnections).
    print(f"\nQUIC portal server ready (Dict: 'openpi-quic-info', port: {QUIC_PORT})")
    print("Test with: uv run python test_modal_quic.py")

    server = QuicPolicyServer(
        policy=policy,
        portal_dict=quic_dict,
        metadata=train_config.policy_metadata,
        local_port=QUIC_PORT,
    )
    server.serve_forever()
