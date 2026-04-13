"""Local Unix-socket backend for transport sidecars.

This server keeps the OpenPI-specific work in Python while allowing a
transport sidecar to own the network protocol. The sidecar sends framed
requests over a Unix domain socket; this server decodes observations,
invokes ``policy.infer()``, and returns msgpack-encoded responses.
"""

from __future__ import annotations

import pathlib
import socket
import time
import traceback
from collections.abc import Callable
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

from hosting.local_sidecar_protocol import (
    SidecarRequestType,
    SidecarResponseType,
    recv_framed_message,
    send_framed_message,
)


class LocalPolicySocketServer:
    """Serve OpenPI policy inference over a local Unix domain socket."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        socket_path: pathlib.Path,
        metadata: dict[str, Any],
        log: Callable[[str], None],
    ) -> None:
        self._policy = policy
        self._socket_path = socket_path
        self._log = log
        self._metadata_packer = msgpack_numpy.Packer()
        self._packed_metadata = self._metadata_packer.pack(metadata)

    def serve_forever(self) -> None:
        """Listen forever for sidecar connections."""
        self._remove_stale_socket_file()

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(str(self._socket_path))
            server_socket.listen()
            self._log(f"[local-policy-socket] Listening on Unix socket {self._socket_path}")

            while True:
                connection_socket, _ = server_socket.accept()
                with connection_socket:
                    self._log("[local-policy-socket] Sidecar connected")
                    try:
                        self._serve_connection(connection_socket)
                    except Exception:
                        self._log(
                            f"[local-policy-socket] Connection error:\n{traceback.format_exc()}"
                        )

    def _remove_stale_socket_file(self) -> None:
        if self._socket_path.exists():
            if not self._socket_path.is_socket():
                raise RuntimeError(
                    f"Local policy socket path exists and is not a socket: {self._socket_path}"
                )
            self._socket_path.unlink()

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

    def _serve_connection(self, connection_socket: socket.socket) -> None:
        previous_total_duration_seconds: float | None = None
        response_packer = msgpack_numpy.Packer()

        while True:
            framed_request = recv_framed_message(connection_socket)
            if framed_request is None:
                self._log("[local-policy-socket] Sidecar disconnected")
                break
            if not framed_request:
                raise RuntimeError("Received empty framed request from sidecar")

            request_type = framed_request[0]
            request_body = framed_request[1:]

            if request_type == SidecarRequestType.METADATA:
                send_framed_message(
                    connection_socket,
                    bytes([SidecarResponseType.METADATA]) + self._packed_metadata,
                )
                continue

            if request_type == SidecarRequestType.RESET:
                self._policy.reset()
                send_framed_message(connection_socket, bytes([SidecarResponseType.RESET]))
                previous_total_duration_seconds = None
                continue

            if request_type != SidecarRequestType.INFER:
                raise RuntimeError(f"Unexpected sidecar request type: {request_type!r}")

            request_start_time = time.monotonic()
            try:
                observation = msgpack_numpy.unpackb(request_body)

                infer_start_time = time.monotonic()
                action = self._policy.infer(observation)
                infer_duration_milliseconds = (time.monotonic() - infer_start_time) * 1000

                server_timing: dict[str, float] = {"infer_ms": infer_duration_milliseconds}
                if previous_total_duration_seconds is not None:
                    server_timing["prev_total_ms"] = previous_total_duration_seconds * 1000

                response_payload = response_packer.pack({**action, "server_timing": server_timing})
                send_framed_message(
                    connection_socket,
                    bytes([SidecarResponseType.INFER]) + response_payload,
                )
                previous_total_duration_seconds = time.monotonic() - request_start_time
            except Exception:
                send_framed_message(
                    connection_socket,
                    bytes([SidecarResponseType.ERROR]) + traceback.format_exc().encode("utf-8"),
                )
                previous_total_duration_seconds = None
