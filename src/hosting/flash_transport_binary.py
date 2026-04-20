"""Locate and invoke the openpi-flash-transport binary.

Used by both the backend (``serve.py``) and the client policy
(``flash_transport_policy.py``). Resolution order for the binary:

1. ``OPENPI_FLASH_TRANSPORT_BINARY`` env var override.
2. The standard Docker install path at ``/usr/local/bin/<BINARY_NAME>``.
3. Local cargo build output (``flash-transport/target/{debug,release}/...``)
   — useful for the client-side developer loop where it isn't installed
   globally.

Also holds ``ServerArgs`` / ``ClientArgs`` — typed mirrors of the Rust
``clap`` structs in ``flash-transport/src/main.rs``. Python callers
construct one of these dataclasses instead of hand-building argv strings,
so a Rust flag rename becomes a type error on the Python side.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, fields
from typing import Any

BINARY_NAME = "openpi-flash-transport"
DEFAULT_BINARY_PATH = pathlib.Path(f"/usr/local/bin/{BINARY_NAME}")
ENV_OVERRIDE = "OPENPI_FLASH_TRANSPORT_BINARY"

# Defaults shared by both subcommands. Kept in sync with
# ``flash-transport/src/main.rs`` (see ``ServerArgs`` / ``ClientArgs``).
_DEFAULT_MAX_IDLE_TIMEOUT_SECS = 10
_DEFAULT_KEEP_ALIVE_INTERVAL_SECS = 2
_DEFAULT_INITIAL_WINDOW_BYTES = 1024 * 1024
_DEFAULT_QUIC_PORT = 5555
_DEFAULT_LOCAL_CLIENT_PORT = 5556


def _hosting_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _iter_binary_candidates() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    if configured := os.environ.get(ENV_OVERRIDE):
        candidates.append(pathlib.Path(configured))
    candidates.append(DEFAULT_BINARY_PATH)
    repo_root = _hosting_repo_root()
    candidates.append(repo_root / "flash-transport" / "target" / "debug" / BINARY_NAME)
    candidates.append(repo_root / "flash-transport" / "target" / "release" / BINARY_NAME)
    return candidates


def resolve_binary_path() -> pathlib.Path:
    """Return the first existing candidate path, or raise ``FileNotFoundError``."""
    for candidate in _iter_binary_candidates():
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in _iter_binary_candidates())
    raise FileNotFoundError(
        f"{BINARY_NAME} binary not found. Searched:\n"
        f"{searched}\n"
        f"Set {ENV_OVERRIDE} to override the path."
    )


def _args_to_argv(subcommand: str, args: Any) -> list[str]:
    """Turn a dataclass of CLI args into a ``clap``-compatible argv list.

    Each field becomes ``--kebab-case-name value``. Fields are emitted in
    declaration order.
    """
    argv: list[str] = [subcommand]
    for field in fields(args):
        flag = "--" + field.name.replace("_", "-")
        argv.extend([flag, str(getattr(args, field.name))])
    return argv


@dataclass(frozen=True)
class ServerArgs:
    """Typed mirror of ``openpi-flash-transport server`` CLI flags."""

    backend_socket_path: pathlib.Path
    listen_port: int = _DEFAULT_QUIC_PORT
    max_idle_timeout_secs: int = _DEFAULT_MAX_IDLE_TIMEOUT_SECS
    keep_alive_interval_secs: int = _DEFAULT_KEEP_ALIVE_INTERVAL_SECS
    initial_window_bytes: int = _DEFAULT_INITIAL_WINDOW_BYTES

    def to_argv(self) -> list[str]:
        return _args_to_argv("server", self)


@dataclass(frozen=True)
class ClientArgs:
    """Typed mirror of ``openpi-flash-transport client`` CLI flags."""

    server_host: str
    local_socket_path: pathlib.Path
    server_port: int = _DEFAULT_QUIC_PORT
    local_port: int = _DEFAULT_LOCAL_CLIENT_PORT
    max_idle_timeout_secs: int = _DEFAULT_MAX_IDLE_TIMEOUT_SECS
    keep_alive_interval_secs: int = _DEFAULT_KEEP_ALIVE_INTERVAL_SECS
    initial_window_bytes: int = _DEFAULT_INITIAL_WINDOW_BYTES

    def to_argv(self) -> list[str]:
        return _args_to_argv("client", self)
