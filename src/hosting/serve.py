"""openpi-flash inference server: WebSocket (TCP) + QUIC (UDP) + unix socket.

The server runs in one of three modes, derived from which slots are set in
the loaded ``ServiceConfig``:

- ``action_only``  — PyTorch/JAX action policy on the action transport triple.
- ``planner_only`` — JAX subtask planner on the planner transport triple.
- ``combined``     — both slots loaded; the planner is shared between the
  action endpoint (where it augments the prompt) and the planner endpoint
  (where it's served directly).

Each active slot gets its own websocket port, QUIC port, unix socket, and a
``flash-transport`` subprocess. A threading lock inside ``SubtaskGenerator``
serializes JAX calls when both endpoints share the planner.

For Modal deployments, use the dedicated modal_*.py entry points instead.

Usage:
    uv run python main.py serve --config config.json
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tenacity import retry, wait_exponential

if TYPE_CHECKING:
    from hosting.subtask_generator import SubtaskGenerator

from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared import download as _download
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy

from hosting.admin_server import RuntimeConfig, start_admin_server
from hosting.compile_mode import get_serving_pytorch_compile_mode
from hosting.config import (
    ActionConfig,
    PlannerConfig,
    ServiceConfig,
    SlotTransportConfig,
    load_config,
)
from hosting.flash_transport_binary import (
    BINARY_NAME,
    ENV_OVERRIDE,
    ServerArgs,
    resolve_binary_path,
)
from hosting.local_policy_socket_server import LocalPolicySocketServer
from hosting.warmup import get_action_horizon, make_image_specs, make_warmup_observation

# Finite, mutually exclusive server modes derived from which slots are loaded.
# A Literal (not a bare ``str``) lets the type checker catch typos and keeps
# the three-way switch exhaustive.
ServerMode = Literal["action_only", "planner_only", "combined"]

# The two slot names are finite; typing them lets ``EndpointSpec.name`` be
# meaningful at the call site rather than an arbitrary string.
SlotName = Literal["action", "planner"]


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


@dataclass
class EndpointSpec:
    """One slot's fully assembled transport setup."""

    name: SlotName
    policy: _base_policy.BasePolicy
    transport: SlotTransportConfig
    metadata: dict[str, Any]


def load_action_policy(
    action_config: ActionConfig,
) -> tuple[_base_policy.BasePolicy, _config.TrainConfig]:
    """Load, compile, and warm up the action policy.

    PyTorch vs JAX is auto-detected by ``create_trained_policy`` based on
    whether ``model.safetensors`` exists in the checkpoint directory.
    """
    _log_service_milestone(
        "Preparing action slot "
        f"(config={action_config.model_config_name}, "
        f"checkpoint={action_config.checkpoint_dir})"
    )

    train_config = _config.get_config(action_config.model_config_name)
    _log_service_milestone(
        f"Resolved training config {action_config.model_config_name}; preparing checkpoint path"
    )

    checkpoint_resolution_start_time = time.monotonic()
    resolved_checkpoint_dir = _download.maybe_download(action_config.checkpoint_dir)
    checkpoint_resolution_elapsed_seconds = time.monotonic() - checkpoint_resolution_start_time
    checkpoint_source_kind = (
        "remote" if urllib.parse.urlparse(action_config.checkpoint_dir).scheme else "local"
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
        default_prompt=action_config.default_prompt,
    )
    load_elapsed = time.monotonic() - load_start
    _log_service_milestone(f"Action policy loaded in {load_elapsed:.1f}s")

    # Warmup inference triggers torch.compile and populates the inductor cache
    # so the first real client request doesn't block for minutes.
    _log_service_milestone("Starting warmup inference for torch.compile")
    compile_start = time.monotonic()
    policy.infer(make_warmup_observation(train_config))
    compile_elapsed = time.monotonic() - compile_start
    _log_service_milestone(f"Warmup inference complete in {compile_elapsed:.1f}s")

    return policy, train_config


def _load_planner_slot(
    planner_config: PlannerConfig,
    runtime_config: RuntimeConfig,
) -> SubtaskGenerator:
    """Load and warm up the JAX subtask generator."""
    from hosting.subtask_generator import SubtaskGenerator

    _log_service_milestone(f"Loading JAX planner (checkpoint={planner_config.checkpoint_dir})")
    generator = SubtaskGenerator(
        checkpoint_dir=planner_config.checkpoint_dir,
        max_tokens=planner_config.max_generation_tokens,
        runtime_config=runtime_config,
        generation_prompt_format=planner_config.generation_prompt_format,
    )
    generator.load()

    _log_service_milestone("Running planner warmup")
    generator.warmup()

    _log_service_milestone("Planner ready")
    return generator


def _build_action_metadata(train_config: _config.TrainConfig) -> dict[str, Any]:
    """Metadata advertised on the action endpoint handshake."""
    metadata = dict(train_config.policy_metadata or {})
    metadata["image_specs"] = make_image_specs(train_config)
    advertised_action_horizon = get_action_horizon(train_config)
    if advertised_action_horizon is not None:
        metadata["action_horizon"] = advertised_action_horizon
    return metadata


def _build_planner_metadata(
    action_train_config: _config.TrainConfig | None,
) -> dict[str, Any]:
    """Metadata advertised on the planner endpoint handshake.

    In combined mode (``action_train_config`` is set) we reuse the action
    slot's image_specs so clients resize images consistently regardless of
    which endpoint they're hitting. In planner-only mode there's no canonical
    train config to derive specs from, so we advertise none and let the
    planner's internal ``_normalize_image()`` handle whatever the client
    sends.
    """
    metadata: dict[str, Any] = {}
    if action_train_config is not None:
        metadata["image_specs"] = make_image_specs(action_train_config)
    return metadata


class _FlashTransportExited(RuntimeError):
    """Signals a flash-transport subprocess exit so tenacity can retry.

    The supervisor is expected to run forever (systemd handles true
    container failure). We use tenacity purely for the exponential backoff
    — the retry never terminates on its own.
    """


def _spawn_binary_supervisor(socket_path: pathlib.Path, quic_port: int) -> None:
    """Run ``flash-transport`` in a restart loop. Blocks forever.

    Uses exponential backoff between restarts (1s → 2s → 4s → ... → 60s).
    A crash-looping binary (bad flag, port in use, etc.) therefore backs
    off rather than thrashing the CPU at 1 Hz.
    """
    binary_path = resolve_binary_path()
    args = ServerArgs(
        backend_socket_path=socket_path,
        listen_port=quic_port,
    )
    command = [str(binary_path), *args.to_argv()]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=60))
    def _run_once() -> None:
        _log_service_milestone(
            f"Starting {BINARY_NAME} ({binary_path}) on UDP port {quic_port} (socket={socket_path})"
        )
        try:
            process = subprocess.Popen(command)
        except FileNotFoundError as exc:
            # Binary missing is non-recoverable — let it propagate out of
            # the retry decorator so the supervisor thread dies loudly.
            raise FileNotFoundError(
                f"{BINARY_NAME} binary not found. "
                f"Expected {binary_path}. "
                f"Set {ENV_OVERRIDE} to override the path."
            ) from exc

        exit_code = process.wait()
        _log_service_milestone(
            f"{BINARY_NAME} on port {quic_port} exited with code {exit_code}; restarting"
        )
        # Raise to trigger tenacity's backoff before the next attempt.
        raise _FlashTransportExited(f"exit code {exit_code}")

    _run_once()


def _start_endpoint_transports(spec: EndpointSpec) -> None:
    """Start WebSocket + unix socket listeners for one slot. Non-blocking."""
    thread_safe_policy = ThreadSafePolicy(spec.policy)

    websocket_server = WebsocketPolicyServer(
        policy=thread_safe_policy,
        port=spec.transport.websocket_port,
        metadata=spec.metadata,
    )
    threading.Thread(
        target=websocket_server.serve_forever,
        name=f"websocket-server-{spec.name}",
        daemon=True,
    ).start()
    _log_service_milestone(
        f"WebSocket server [{spec.name}] listening on TCP port {spec.transport.websocket_port}"
    )

    local_policy_socket_server = LocalPolicySocketServer(
        policy=thread_safe_policy,
        socket_path=pathlib.Path(spec.transport.unix_socket_path),
        metadata=spec.metadata,
        log=_log_service_milestone,
    )
    threading.Thread(
        target=local_policy_socket_server.serve_forever,
        name=f"local-policy-socket-server-{spec.name}",
        daemon=True,
    ).start()


def _resolve_mode(service_config: ServiceConfig) -> ServerMode:
    """Derive the server mode from which slots are set.

    ``ServiceConfig``'s model validator already guarantees at least one of
    ``action`` / ``planner`` is present, so we only need to distinguish three
    cases here.
    """
    action_on = service_config.action is not None
    planner_on = service_config.planner is not None
    if action_on and planner_on:
        return "combined"
    if action_on:
        return "action_only"
    return "planner_only"


def main() -> None:
    _log_service_milestone("Loading service configuration")
    service_config = load_config()
    mode = _resolve_mode(service_config)
    _log_service_milestone(f"Service configuration loaded (mode={mode})")

    # Load planner first so the combined-mode action wrapper can reference
    # the same SubtaskGenerator instance the planner endpoint serves.
    planner_generator: SubtaskGenerator | None = None
    runtime_config: RuntimeConfig | None = None
    if service_config.planner is not None:
        runtime_config = RuntimeConfig(
            generation_prompt_format=service_config.planner.generation_prompt_format
        )
        planner_generator = _load_planner_slot(service_config.planner, runtime_config)

    # Load action policy (PyTorch or JAX, auto-detected).
    action_policy: _base_policy.BasePolicy | None = None
    action_train_config: _config.TrainConfig | None = None
    if service_config.action is not None:
        action_policy, action_train_config = load_action_policy(service_config.action)

    # Assemble one EndpointSpec per active slot.
    endpoint_specs: list[EndpointSpec] = []

    if action_policy is not None and action_train_config is not None:
        if planner_generator is not None:
            assert service_config.planner is not None
            from hosting.subtask_policy import SubtaskAugmentedPolicy

            action_endpoint_policy: _base_policy.BasePolicy = SubtaskAugmentedPolicy(
                inner_policy=action_policy,
                subtask_generator=planner_generator,
                prompt_template=service_config.planner.action_prompt_template,
            )
            _log_service_milestone(
                "Action endpoint wrapped with subtask augmentation (combined mode)"
            )
        else:
            action_endpoint_policy = action_policy

        endpoint_specs.append(
            EndpointSpec(
                name="action",
                policy=action_endpoint_policy,
                transport=service_config.action_transport,
                metadata=_build_action_metadata(action_train_config),
            )
        )

    if planner_generator is not None:
        from hosting.subtask_policy import PlannerPolicy

        endpoint_specs.append(
            EndpointSpec(
                name="planner",
                policy=PlannerPolicy(planner_generator),
                transport=service_config.planner_transport,
                metadata=_build_planner_metadata(action_train_config),
            )
        )

    # Admin HTTP endpoint is only relevant when the planner is loaded.
    if runtime_config is not None:
        start_admin_server(runtime_config)
        _log_service_milestone("Admin HTTP server started on port 8001")

    # WebSocket + unix socket listeners per active slot.
    for spec in endpoint_specs:
        _start_endpoint_transports(spec)

    # One flash-transport subprocess per slot, each supervised in its own
    # non-daemon thread — main() blocks by joining all supervisors.
    supervisor_threads: list[threading.Thread] = []
    for spec in endpoint_specs:
        thread = threading.Thread(
            target=_spawn_binary_supervisor,
            args=(
                pathlib.Path(spec.transport.unix_socket_path),
                spec.transport.quic_port,
            ),
            name=f"flash-transport-supervisor-{spec.name}",
            daemon=False,
        )
        thread.start()
        supervisor_threads.append(thread)

    _log_service_milestone(
        f"Service ready (mode={mode}, slots={[spec.name for spec in endpoint_specs]})"
    )

    for thread in supervisor_threads:
        thread.join()


if __name__ == "__main__":
    main()
