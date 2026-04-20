"""Drift test: verify the Python typed CLI wrapper matches the Rust ``clap``
argument structs.

Spawns the ``openpi-flash-transport`` binary with ``<subcommand> --help``,
parses the ``--flag-name`` tokens out of the help text, and asserts the
Python dataclasses in ``hosting.flash_transport_binary`` have matching
field names.

Skipped when the binary isn't built (e.g. fresh checkout without cargo).
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from dataclasses import fields
from typing import Any

import pytest

from hosting.flash_transport_binary import (
    ClientArgs,
    ServerArgs,
    _iter_binary_candidates,
)

_FLAG_RE = re.compile(r"--([a-z][a-z0-9-]*)")


def _resolve_or_skip() -> pathlib.Path:
    for candidate in _iter_binary_candidates():
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in _iter_binary_candidates())
    pytest.skip(
        "openpi-flash-transport binary not built. Run `cargo build` in "
        f"flash-transport/ to enable these tests. Searched:\n{searched}"
    )


def _parse_help_flags(binary_path: pathlib.Path, subcommand: str) -> set[str]:
    """Return the set of ``--flag-name`` tokens from ``<bin> <subcommand> --help``."""
    completed = subprocess.run(
        [str(binary_path), subcommand, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(_FLAG_RE.findall(completed.stdout))


def _dataclass_field_flags(cls: Any) -> set[str]:
    """Return the kebab-case flag name for each dataclass field."""
    return {field.name.replace("_", "-") for field in fields(cls)}


def _assert_flags_match(expected: set[str], actual: set[str], subcommand: str) -> None:
    # ``--help`` is emitted by clap for every subcommand; not a real arg.
    actual = actual - {"help"}
    missing_in_python = actual - expected
    extra_in_python = expected - actual
    assert not missing_in_python, (
        f"Rust `{subcommand}` has flags Python is missing: {sorted(missing_in_python)}"
    )
    assert not extra_in_python, (
        f"Python `{subcommand}` has flags Rust doesn't expose: {sorted(extra_in_python)}"
    )


def test_server_args_match_rust_cli() -> None:
    binary_path = _resolve_or_skip()
    rust_flags = _parse_help_flags(binary_path, "server")
    python_flags = _dataclass_field_flags(ServerArgs)
    _assert_flags_match(python_flags, rust_flags, "server")


def test_client_args_match_rust_cli() -> None:
    binary_path = _resolve_or_skip()
    rust_flags = _parse_help_flags(binary_path, "client")
    python_flags = _dataclass_field_flags(ClientArgs)
    _assert_flags_match(python_flags, rust_flags, "client")


def test_server_args_argv_contains_required_flags() -> None:
    args = ServerArgs(backend_socket_path=pathlib.Path("/tmp/backend.sock"), listen_port=5555)
    argv = args.to_argv()
    assert argv[0] == "server"
    assert "--backend-socket-path" in argv
    assert "/tmp/backend.sock" in argv
    assert "--listen-port" in argv
    assert "5555" in argv


def test_client_args_argv_contains_required_flags() -> None:
    args = ClientArgs(
        server_host="localhost",
        local_socket_path=pathlib.Path("/tmp/client.sock"),
    )
    argv = args.to_argv()
    assert argv[0] == "client"
    assert "--server-host" in argv
    assert "localhost" in argv
    assert "--local-socket-path" in argv
    assert "/tmp/client.sock" in argv
