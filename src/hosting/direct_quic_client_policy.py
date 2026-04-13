"""Direct QUIC client policy backed by a local Rust sidecar process.

This preserves the normal Python ``BasePolicy`` interface used by openpi
clients while moving QUIC transport handling into the Rust sidecar binary.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import uuid

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import QuicTransportOptions
from typing_extensions import override

from hosting.local_sidecar_protocol import (
    SidecarRequestType,
    SidecarResponseType,
    recv_framed_message,
    send_framed_message,
)

DEFAULT_RUST_QUIC_SIDECAR_BINARY_PATH = pathlib.Path("/usr/local/bin/openpi-quic-sidecar")
DEFAULT_LOCAL_SIDECAR_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_LOCAL_SIDECAR_POLL_INTERVAL_SECONDS = 0.1


def _get_hosting_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _iter_sidecar_binary_candidates() -> list[pathlib.Path]:
    configured_binary_path = None
    if configured_binary_path_string := os.environ.get("OPENPI_QUIC_SIDECAR_BINARY"):
        configured_binary_path = pathlib.Path(configured_binary_path_string)

    hosting_repo_root = _get_hosting_repo_root()
    candidate_binary_paths = [
        path
        for path in [
            configured_binary_path,
            DEFAULT_RUST_QUIC_SIDECAR_BINARY_PATH,
            hosting_repo_root / "quic-sidecar" / "target" / "debug" / "openpi-quic-sidecar",
            hosting_repo_root / "quic-sidecar" / "target" / "release" / "openpi-quic-sidecar",
        ]
        if path is not None
    ]
    return candidate_binary_paths


def _get_rust_quic_sidecar_binary_path() -> pathlib.Path:
    for candidate_binary_path in _iter_sidecar_binary_candidates():
        if candidate_binary_path.exists():
            return candidate_binary_path

    candidate_path_message = "\n".join(
        f"  - {candidate_binary_path}"
        for candidate_binary_path in _iter_sidecar_binary_candidates()
    )
    raise FileNotFoundError(
        "Rust QUIC sidecar binary not found. Searched:\n"
        f"{candidate_path_message}\n"
        "Set OPENPI_QUIC_SIDECAR_BINARY to override the path."
    )


class DirectQuicClientPolicy(_base_policy.BasePolicy):
    """Connects to a direct QUIC server through a local Rust sidecar process."""

    def __init__(
        self,
        host: str,
        port: int = 5555,
        local_port: int = 5556,
        transport_options: QuicTransportOptions | None = None,
    ) -> None:
        if transport_options is not None:
            raise ValueError(
                "Custom transport_options are not supported by the Rust QUIC sidecar client yet"
            )

        self._packer = msgpack_numpy.Packer()
        self._socket_path = pathlib.Path(
            tempfile.gettempdir(),
            f"openpi-quic-client-{uuid.uuid4().hex}.sock",
        )
        self._sidecar_process = self._spawn_local_sidecar(
            host=host,
            port=port,
            local_port=local_port,
            socket_path=self._socket_path,
        )
        self._sidecar_socket = self._connect_to_local_sidecar(self._socket_path)
        self._server_metadata = self._request_metadata()

    def _spawn_local_sidecar(
        self,
        *,
        host: str,
        port: int,
        local_port: int,
        socket_path: pathlib.Path,
    ) -> subprocess.Popen[str]:
        sidecar_binary_path = _get_rust_quic_sidecar_binary_path()
        sidecar_command = [
            str(sidecar_binary_path),
            "client",
            "--server-host",
            host,
            "--server-port",
            str(port),
            "--local-port",
            str(local_port),
            "--local-socket-path",
            str(socket_path),
        ]
        return subprocess.Popen(sidecar_command, text=True)

    def _connect_to_local_sidecar(self, socket_path: pathlib.Path) -> socket.socket:
        wait_deadline = time.monotonic() + DEFAULT_LOCAL_SIDECAR_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < wait_deadline:
            if self._sidecar_process.poll() is not None:
                raise RuntimeError(
                    "Rust QUIC sidecar client exited before opening its local socket "
                    f"(exit_code={self._sidecar_process.returncode})"
                )

            if socket_path.exists():
                local_sidecar_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    local_sidecar_socket.connect(str(socket_path))
                    return local_sidecar_socket
                except OSError:
                    local_sidecar_socket.close()

            time.sleep(DEFAULT_LOCAL_SIDECAR_POLL_INTERVAL_SECONDS)

        raise TimeoutError(f"Timed out waiting for local QUIC sidecar socket at {socket_path}")

    def _request(self, request_type: SidecarRequestType, payload: bytes = b"") -> bytes:
        send_framed_message(self._sidecar_socket, bytes([request_type]) + payload)
        framed_response = recv_framed_message(self._sidecar_socket)
        if framed_response is None:
            raise ConnectionError("Local QUIC sidecar disconnected unexpectedly")
        if not framed_response:
            raise RuntimeError("Received empty response from local QUIC sidecar")

        response_type = SidecarResponseType(framed_response[0])
        response_body = framed_response[1:]

        if response_type == SidecarResponseType.ERROR:
            raise RuntimeError(f"Error from local QUIC sidecar:\n{response_body.decode('utf-8')}")
        if (
            request_type == SidecarRequestType.METADATA
            and response_type != SidecarResponseType.METADATA
        ):
            raise RuntimeError(f"Unexpected metadata response type: {response_type!r}")
        if request_type == SidecarRequestType.INFER and response_type != SidecarResponseType.INFER:
            raise RuntimeError(f"Unexpected inference response type: {response_type!r}")
        if request_type == SidecarRequestType.RESET and response_type != SidecarResponseType.RESET:
            raise RuntimeError(f"Unexpected reset response type: {response_type!r}")

        return response_body

    def _request_metadata(self) -> dict:
        return msgpack_numpy.unpackb(self._request(SidecarRequestType.METADATA))

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    @override
    def infer(self, obs: dict) -> dict:
        return msgpack_numpy.unpackb(
            self._request(SidecarRequestType.INFER, self._packer.pack(obs))
        )

    @override
    def reset(self) -> None:
        self._request(SidecarRequestType.RESET)

    def close(self) -> None:
        """Close the local sidecar connection and stop the sidecar process."""
        with contextlib.suppress(OSError):
            self._sidecar_socket.close()

        if self._sidecar_process.poll() is None:
            self._sidecar_process.terminate()
            try:
                self._sidecar_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sidecar_process.kill()
                self._sidecar_process.wait(timeout=5)

        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
