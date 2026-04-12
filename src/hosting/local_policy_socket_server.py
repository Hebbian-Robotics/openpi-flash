"""Local Unix-socket backend for transport sidecars.

This server keeps the OpenPI-specific work in Python while allowing a
transport sidecar to own the network protocol. The sidecar sends framed
requests over a Unix domain socket; this server decodes observations,
invokes ``policy.infer()``, and returns msgpack-encoded responses.
"""

from __future__ import annotations

import pathlib
import socket
import struct
import time
import traceback
from collections.abc import Callable

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Request message types sent by the local sidecar.
_REQUEST_TYPE_METADATA = 0x01
_REQUEST_TYPE_INFER = 0x02
_REQUEST_TYPE_RESET = 0x03

# Response message types returned to the local sidecar.
_RESPONSE_TYPE_METADATA = 0x11
_RESPONSE_TYPE_INFER = 0x12
_RESPONSE_TYPE_ERROR = 0x13
_RESPONSE_TYPE_RESET = 0x14


def _recv_exactly(stream_socket: socket.socket, num_bytes: int) -> bytes | None:
    """Read exactly ``num_bytes`` from a stream socket or return None on EOF."""
    received_chunks = bytearray()
    while len(received_chunks) < num_bytes:
        chunk = stream_socket.recv(num_bytes - len(received_chunks))
        if not chunk:
            return None
        received_chunks.extend(chunk)
    return bytes(received_chunks)


def _recv_framed_message(stream_socket: socket.socket) -> bytes | None:
    """Receive one length-prefixed message from a Unix stream socket."""
    raw_length_prefix = _recv_exactly(stream_socket, 4)
    if raw_length_prefix is None:
        return None

    message_length = struct.unpack(">I", raw_length_prefix)[0]
    if message_length == 0:
        return b""

    payload = _recv_exactly(stream_socket, message_length)
    if payload is None:
        raise ConnectionError("Unexpected EOF while reading framed Unix-socket message")
    return payload


def _send_framed_message(stream_socket: socket.socket, payload: bytes) -> None:
    """Send one length-prefixed message over a Unix stream socket."""
    stream_socket.sendall(struct.pack(">I", len(payload)))
    if payload:
        stream_socket.sendall(payload)


class LocalPolicySocketServer:
    """Serve OpenPI policy inference over a local Unix domain socket."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        socket_path: pathlib.Path,
        metadata: dict,
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
            framed_request = _recv_framed_message(connection_socket)
            if framed_request is None:
                self._log("[local-policy-socket] Sidecar disconnected")
                break
            if not framed_request:
                raise RuntimeError("Received empty framed request from sidecar")

            request_type = framed_request[0]
            request_body = framed_request[1:]

            if request_type == _REQUEST_TYPE_METADATA:
                _send_framed_message(
                    connection_socket,
                    bytes([_RESPONSE_TYPE_METADATA]) + self._packed_metadata,
                )
                continue

            if request_type == _REQUEST_TYPE_RESET:
                self._policy.reset()
                _send_framed_message(connection_socket, bytes([_RESPONSE_TYPE_RESET]))
                previous_total_duration_seconds = None
                continue

            if request_type != _REQUEST_TYPE_INFER:
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
                _send_framed_message(
                    connection_socket,
                    bytes([_RESPONSE_TYPE_INFER]) + response_payload,
                )
                previous_total_duration_seconds = time.monotonic() - request_start_time
            except Exception:
                _send_framed_message(
                    connection_socket,
                    bytes([_RESPONSE_TYPE_ERROR]) + traceback.format_exc().encode("utf-8"),
                )
                previous_total_duration_seconds = None
