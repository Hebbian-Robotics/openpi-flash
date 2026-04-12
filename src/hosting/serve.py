"""Hosted inference server: WebSocket (TCP) + QUIC (UDP).

Loads the policy once and serves it over both transports simultaneously.
WebSocket runs on the configured port (default 8000, TCP), QUIC listens on
port 5555 (UDP). A threading lock serializes GPU inference calls from both
transports.

For Modal deployments, use the dedicated modal_*.py entry points instead.

Usage:
    INFERENCE_CONFIG_PATH=config.json python -m hosting.serve
"""

import dataclasses
import datetime
import logging
import threading
import time
import traceback

from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy
from quic_portal import Portal, PortalError

from hosting.config import ServiceConfig, load_config
from hosting.quic_protocol import DEFAULT_TRANSPORT_OPTIONS, serve_quic_connection
from hosting.warmup import make_aloha_warmup_observation

logger = logging.getLogger(__name__)

# Compile mode "default" is proven reliable on Modal (~2.5 min compile, ~76ms
# inference). The config default "max-autotune" optimises for training
# throughput and takes longer to compile with no serving-speed benefit.
PYTORCH_COMPILE_MODE = "default"

QUIC_PORT = 5555


def _quic_log(msg: str) -> None:
    """Log with UTC timestamp for correlating with relay/portal logs."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S.%f")[:-3]
    logger.info("%s %s", ts, msg)


class ThreadSafePolicy(_base_policy.BasePolicy):
    """Wraps a BasePolicy with a lock for concurrent access from multiple servers."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy
        self._lock = threading.Lock()

    def infer(self, obs: dict) -> dict:
        with self._lock:
            return self._policy.infer(obs)

    def reset(self) -> None:
        with self._lock:
            self._policy.reset()


def load_policy(
    service_config: ServiceConfig,
) -> tuple[_base_policy.BasePolicy, _config.TrainConfig]:
    """Load, compile, and warm up a policy. Returns (policy, train_config).

    Also used by the Modal entry points (modal_helpers.py) to share model
    loading logic.
    """
    logger.info(
        "Loading model: config=%s, checkpoint=%s",
        service_config.model_config_name,
        service_config.checkpoint_dir,
    )

    train_config = _config.get_config(service_config.model_config_name)

    # Override compile mode for faster, reliable compilation during serving.
    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, pytorch_compile_mode=PYTORCH_COMPILE_MODE
            ),
        )

    load_start = time.monotonic()
    policy = _policy_config.create_trained_policy(
        train_config,
        service_config.checkpoint_dir,
        default_prompt=service_config.default_prompt,
    )
    load_elapsed = time.monotonic() - load_start
    logger.info("Model loaded in %.1fs", load_elapsed)

    # Warmup inference triggers torch.compile and populates the inductor cache
    # so the first real client request doesn't block for minutes.
    logger.info("Compiling model (warmup inference) ...")
    compile_start = time.monotonic()
    policy.infer(make_aloha_warmup_observation())
    compile_elapsed = time.monotonic() - compile_start
    logger.info("Compilation done in %.1fs", compile_elapsed)

    return policy, train_config


def _run_quic_server(policy: ThreadSafePolicy, metadata: dict) -> None:
    """Block forever, accepting and serving one QUIC client at a time.

    Uses Portal.listen() for direct connections — no STUN, no relay, no dict
    coordination. Clients connect directly to <host>:<QUIC_PORT>.
    """
    while True:
        try:
            _quic_log(f"[quic-server] Listening on UDP port {QUIC_PORT}...")
            portal = Portal()
            portal.listen(QUIC_PORT, DEFAULT_TRANSPORT_OPTIONS)
            _quic_log("[quic-server] Client connected")

            serve_quic_connection(portal, policy, metadata, log=_quic_log)
        except PortalError as e:
            _quic_log(f"[quic-server] Portal error, will retry: {e}")
        except Exception:
            _quic_log(f"[quic-server] Error, will retry:\n{traceback.format_exc()}")
        finally:
            time.sleep(1)  # Brief pause before accepting next client.


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    service_config = load_config()
    policy, train_config = load_policy(service_config)

    thread_safe_policy = ThreadSafePolicy(policy)
    metadata = train_config.policy_metadata or {}

    # Start WebSocket server in a daemon thread.
    websocket_server = WebsocketPolicyServer(
        policy=thread_safe_policy,
        port=service_config.port,
        metadata=metadata,
    )
    websocket_thread = threading.Thread(
        target=websocket_server.serve_forever,
        name="websocket-server",
        daemon=True,
    )
    websocket_thread.start()
    logger.info("WebSocket server started on port %d", service_config.port)

    # Run QUIC server on the main thread (blocks forever).
    logger.info("Starting QUIC server on UDP port %d", QUIC_PORT)
    _run_quic_server(thread_safe_policy, metadata)


if __name__ == "__main__":
    main()
