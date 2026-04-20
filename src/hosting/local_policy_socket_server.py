"""Local Unix-socket backend for ``openpi-flash-transport``.

Keeps OpenPI-specific work in Python while the transport binary owns the
network protocol. It sends framed requests over a Unix domain socket; this
server decodes observations, invokes ``policy.infer()``, and returns
local-frame-encoded responses.
"""

from __future__ import annotations

import pathlib
import socket
import traceback
from collections.abc import Callable
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

from hosting.flash_transport_binary import BINARY_NAME
from hosting.local_frame import pack_local_frame, unpack_local_frame
from hosting.local_transport_protocol import (
    TransportRequestType,
    TransportResponseType,
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
        # Metadata stays msgpack_numpy end to end — it's openpi's blob,
        # forwarded verbatim through the QUIC handshake.
        self._packed_metadata = msgpack_numpy.Packer().pack(metadata)

    def serve_forever(self) -> None:
        """Listen forever for transport connections."""
        self._remove_stale_socket_file()

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(str(self._socket_path))
            server_socket.listen()
            self._log(f"[local-policy-socket] Listening on Unix socket {self._socket_path}")

            while True:
                connection_socket, _ = server_socket.accept()
                with connection_socket:
                    self._log("[local-policy-socket] Transport connected")
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
        while True:
            framed_request = recv_framed_message(connection_socket)
            if framed_request is None:
                self._log("[local-policy-socket] Transport disconnected")
                break
            if not framed_request:
                raise RuntimeError(f"Received empty framed request from {BINARY_NAME}")

            request_type = framed_request[0]
            request_body = framed_request[1:]

            if request_type == TransportRequestType.METADATA:
                send_framed_message(
                    connection_socket,
                    bytes([TransportResponseType.METADATA]) + self._packed_metadata,
                )
                continue

            if request_type == TransportRequestType.RESET:
                self._policy.reset()
                send_framed_message(connection_socket, bytes([TransportResponseType.RESET]))
                continue

            if request_type != TransportRequestType.INFER:
                raise RuntimeError(f"Unexpected transport request type: {request_type!r}")

            try:
                observation = unpack_local_frame(request_body)
                # server_timing is injected by the transport server; we
                # forward `action` untouched so policy_timing (set by the
                # model) reaches the client as-is.
                action = self._policy.infer(observation)
                response_payload = pack_local_frame(action)
                send_framed_message(
                    connection_socket,
                    bytes([TransportResponseType.INFER]) + response_payload,
                )
            except Exception:
                send_framed_message(
                    connection_socket,
                    bytes([TransportResponseType.ERROR]) + traceback.format_exc().encode("utf-8"),
                )
