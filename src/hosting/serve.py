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
import urllib.parse

from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import download as _download
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


def _log_service_milestone(message: str) -> None:
    """Emit a concise startup milestone to stdout for container logs."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{timestamp} {message}", flush=True)


def _quic_log(msg: str) -> None:
    """Log with UTC timestamp for correlating with relay/portal logs."""
    _log_service_milestone(msg)


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
    _log_service_milestone(
        "Preparing policy load "
        f"(config={service_config.model_config_name}, checkpoint={service_config.checkpoint_dir})"
    )

    train_config = _config.get_config(service_config.model_config_name)
    _log_service_milestone(
        f"Resolved training config {service_config.model_config_name}; preparing checkpoint path"
    )

    checkpoint_resolution_start_time = time.monotonic()
    resolved_checkpoint_dir = _download.maybe_download(service_config.checkpoint_dir)
    checkpoint_resolution_elapsed_seconds = time.monotonic() - checkpoint_resolution_start_time
    checkpoint_source_kind = (
        "remote" if urllib.parse.urlparse(service_config.checkpoint_dir).scheme else "local"
    )
    _log_service_milestone(
        "Checkpoint path ready "
        f"(source={checkpoint_source_kind}, local_path={resolved_checkpoint_dir}, "
        f"elapsed={checkpoint_resolution_elapsed_seconds:.1f}s)"
    )

    # Override compile mode for faster, reliable compilation during serving.
    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, pytorch_compile_mode=PYTORCH_COMPILE_MODE
            ),
        )

    load_start = time.monotonic()
    _log_service_milestone("Creating trained policy from resolved checkpoint")
    policy = _policy_config.create_trained_policy(
        train_config,
        resolved_checkpoint_dir,
        default_prompt=service_config.default_prompt,
    )
    load_elapsed = time.monotonic() - load_start
    _log_service_milestone(f"Policy loaded in {load_elapsed:.1f}s")

    # Warmup inference triggers torch.compile and populates the inductor cache
    # so the first real client request doesn't block for minutes.
    _log_service_milestone("Starting warmup inference for torch.compile")
    compile_start = time.monotonic()
    policy.infer(make_aloha_warmup_observation())
    compile_elapsed = time.monotonic() - compile_start
    _log_service_milestone(f"Warmup inference complete in {compile_elapsed:.1f}s")

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

            serve_quic_connection(
                portal,
                policy,
                metadata,
                log=_quic_log,
                client_initiates_handshake=True,
            )
        except PortalError as e:
            _quic_log(f"[quic-server] Portal error, will retry: {e}")
        except Exception:
            _quic_log(f"[quic-server] Error, will retry:\n{traceback.format_exc()}")
        finally:
            time.sleep(1)  # Brief pause before accepting next client.


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    _log_service_milestone("Loading service configuration")
    service_config = load_config()
    _log_service_milestone(
        "Service configuration loaded "
        f"(port={service_config.port}, max_concurrent_requests={service_config.max_concurrent_requests})"
    )
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
    _log_service_milestone(f"WebSocket server thread started on TCP port {service_config.port}")

    # Run QUIC server on the main thread (blocks forever).
    _log_service_milestone(f"Starting QUIC server on UDP port {QUIC_PORT}")
    _run_quic_server(thread_safe_policy, metadata)


if __name__ == "__main__":
    main()
