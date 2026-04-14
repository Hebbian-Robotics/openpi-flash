"""openpi-flash inference server: WebSocket (TCP) + QUIC (UDP).

Loads the policy once and serves it over both transports simultaneously.
WebSocket runs on the configured port (default 8000, TCP), QUIC listens on
port 5555 (UDP). A threading lock serializes GPU inference calls from both
transports.

For Modal deployments, use the dedicated modal_*.py entry points instead.

Usage:
    uv run python main.py serve --config config.json
"""

import dataclasses
import datetime
import os
import pathlib
import subprocess
import threading
import time
import urllib.parse

from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import download as _download
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy

from hosting.compile_mode import get_serving_pytorch_compile_mode
from hosting.config import ServiceConfig, load_config
from hosting.local_policy_socket_server import LocalPolicySocketServer
from hosting.warmup import make_aloha_warmup_observation

DEFAULT_LOCAL_POLICY_SOCKET_PATH = pathlib.Path("/tmp/openpi-policy.sock")
DEFAULT_RUST_QUIC_SIDECAR_BINARY_PATH = pathlib.Path("/usr/local/bin/openpi-quic-sidecar")


def _log_service_milestone(message: str) -> None:
    """Emit a concise startup milestone to stdout for container logs."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{timestamp} {message}", flush=True)


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

    serving_pytorch_compile_mode = get_serving_pytorch_compile_mode()
    _log_service_milestone(f"Using PyTorch compile mode {serving_pytorch_compile_mode!r}")

    # Override compile mode for faster, reliable compilation during serving.
    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model,
                pytorch_compile_mode=serving_pytorch_compile_mode,
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


def _get_local_policy_socket_path() -> pathlib.Path:
    configured_socket_path = os.environ.get("OPENPI_LOCAL_POLICY_SOCKET_PATH")
    if configured_socket_path:
        return pathlib.Path(configured_socket_path)
    return DEFAULT_LOCAL_POLICY_SOCKET_PATH


def _get_rust_quic_sidecar_binary_path() -> pathlib.Path:
    configured_binary_path = os.environ.get("OPENPI_QUIC_SIDECAR_BINARY")
    if configured_binary_path:
        return pathlib.Path(configured_binary_path)
    return DEFAULT_RUST_QUIC_SIDECAR_BINARY_PATH


def _run_rust_quic_sidecar(local_policy_socket_path: pathlib.Path, quic_port: int) -> None:
    sidecar_binary_path = _get_rust_quic_sidecar_binary_path()
    sidecar_command = [
        str(sidecar_binary_path),
        "server",
        "--listen-port",
        str(quic_port),
        "--backend-socket-path",
        str(local_policy_socket_path),
    ]

    while True:
        _log_service_milestone(
            f"Starting Rust QUIC sidecar ({sidecar_binary_path}) on UDP port {quic_port}"
        )
        try:
            sidecar_process = subprocess.Popen(sidecar_command)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Rust QUIC sidecar binary not found. "
                f"Expected {sidecar_binary_path}. "
                "Set OPENPI_QUIC_SIDECAR_BINARY to override the path."
            ) from exc

        sidecar_exit_code = sidecar_process.wait()
        _log_service_milestone(
            f"Rust QUIC sidecar exited with code {sidecar_exit_code}; restarting in 1s"
        )
        time.sleep(1)


def main() -> None:
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
    threading.Thread(
        target=websocket_server.serve_forever,
        name="websocket-server",
        daemon=True,
    ).start()
    _log_service_milestone(f"WebSocket server thread started on TCP port {service_config.port}")

    local_policy_socket_path = _get_local_policy_socket_path()
    local_policy_socket_server = LocalPolicySocketServer(
        policy=thread_safe_policy,
        socket_path=local_policy_socket_path,
        metadata=metadata,
        log=_log_service_milestone,
    )
    threading.Thread(
        target=local_policy_socket_server.serve_forever,
        name="local-policy-socket-server",
        daemon=True,
    ).start()
    _run_rust_quic_sidecar(local_policy_socket_path, service_config.quic_port)


if __name__ == "__main__":
    main()
