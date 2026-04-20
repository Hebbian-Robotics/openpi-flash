"""Client policy backed by a local ``openpi-flash-transport`` subprocess.

This preserves the normal Python ``BasePolicy`` interface used by openpi
clients while moving QUIC transport, Arrow IPC codec, image preprocessing,
action chunking, and server-timing instrumentation into the transport
binary.

The QUIC path speaks Arrow IPC Streaming Format on the wire (the transport
binary owns the codec translation); ``openpi-client``'s WebSocket path is
the supported pure-Python msgpack alternative for customers who don't want
the transport binary as a dependency.
"""

from __future__ import annotations

import contextlib
import pathlib
import socket
import subprocess
import time
import uuid

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import QuicTransportOptions
from typing_extensions import override

from hosting.flash_transport_binary import BINARY_NAME, ClientArgs, resolve_binary_path
from hosting.local_frame import pack_local_frame, unpack_local_frame
from hosting.local_transport_protocol import (
    TransportRequestType,
    TransportResponseType,
    recv_framed_message,
    send_framed_message,
)

DEFAULT_TRANSPORT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_TRANSPORT_POLL_INTERVAL_SECONDS = 0.1

# Unix sockets must fit in sun_path (104 bytes on macOS, 108 on Linux), so we
# can't use tempfile.gettempdir() here — macOS's default $TMPDIR is a long
# /var/folders/... path that overflows once the UUID filename is appended.
_UNIX_SOCKET_DIR = pathlib.Path("/tmp")


class FlashTransportPolicy(_base_policy.BasePolicy):
    """Connects to a direct QUIC server through a local ``openpi-flash-transport`` subprocess."""

    def __init__(
        self,
        host: str,
        port: int = 5555,
        local_port: int = 5556,
        transport_options: QuicTransportOptions | None = None,
    ) -> None:
        if transport_options is not None:
            raise ValueError(f"Custom transport_options are not supported by {BINARY_NAME} yet")

        self._socket_path = _UNIX_SOCKET_DIR / f"{BINARY_NAME}-client-{uuid.uuid4().hex}.sock"
        self._transport_process = self._spawn_transport_process(
            host=host,
            port=port,
            local_port=local_port,
            socket_path=self._socket_path,
        )
        self._transport_socket = self._connect_to_transport_socket(self._socket_path)
        self._server_metadata = self._request_metadata()

    def _spawn_transport_process(
        self,
        *,
        host: str,
        port: int,
        local_port: int,
        socket_path: pathlib.Path,
    ) -> subprocess.Popen[str]:
        binary_path = resolve_binary_path()
        args = ClientArgs(
            server_host=host,
            local_socket_path=socket_path,
            server_port=port,
            local_port=local_port,
        )
        command = [str(binary_path), *args.to_argv()]
        return subprocess.Popen(command, text=True)

    def _connect_to_transport_socket(self, socket_path: pathlib.Path) -> socket.socket:
        wait_deadline = time.monotonic() + DEFAULT_TRANSPORT_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < wait_deadline:
            if self._transport_process.poll() is not None:
                raise RuntimeError(
                    f"{BINARY_NAME} client exited before opening its local socket "
                    f"(exit_code={self._transport_process.returncode})"
                )

            if socket_path.exists():
                transport_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    transport_socket.connect(str(socket_path))
                    return transport_socket
                except OSError:
                    transport_socket.close()

            time.sleep(DEFAULT_TRANSPORT_POLL_INTERVAL_SECONDS)

        raise TimeoutError(f"Timed out waiting for {BINARY_NAME} socket at {socket_path}")

    def _request(self, request_type: TransportRequestType, payload: bytes = b"") -> bytes:
        send_framed_message(self._transport_socket, bytes([request_type]) + payload)
        framed_response = recv_framed_message(self._transport_socket)
        if framed_response is None:
            raise ConnectionError(f"{BINARY_NAME} disconnected unexpectedly")
        if not framed_response:
            raise RuntimeError(f"Received empty response from {BINARY_NAME}")

        response_type = TransportResponseType(framed_response[0])
        response_body = framed_response[1:]

        if response_type == TransportResponseType.ERROR:
            raise RuntimeError(f"Error from {BINARY_NAME}:\n{response_body.decode('utf-8')}")
        if (
            request_type == TransportRequestType.METADATA
            and response_type != TransportResponseType.METADATA
        ):
            raise RuntimeError(f"Unexpected metadata response type: {response_type!r}")
        if (
            request_type == TransportRequestType.INFER
            and response_type != TransportResponseType.INFER
        ):
            raise RuntimeError(f"Unexpected inference response type: {response_type!r}")
        if (
            request_type == TransportRequestType.RESET
            and response_type != TransportResponseType.RESET
        ):
            raise RuntimeError(f"Unexpected reset response type: {response_type!r}")

        return response_body

    def _request_metadata(self) -> dict:
        # Metadata stays msgpack_numpy end to end — it's openpi's blob,
        # forwarded verbatim through the handshake.
        return msgpack_numpy.unpackb(self._request(TransportRequestType.METADATA))

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    @override
    def infer(self, obs: dict) -> dict:
        frame = pack_local_frame(obs)
        response_body = self._request(TransportRequestType.INFER, frame)
        return unpack_local_frame(response_body)

    @override
    def reset(self) -> None:
        self._request(TransportRequestType.RESET)

    def close(self) -> None:
        """Close the local socket and stop the subprocess."""
        with contextlib.suppress(OSError):
            self._transport_socket.close()

        if self._transport_process.poll() is None:
            self._transport_process.terminate()
            try:
                self._transport_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._transport_process.kill()
                self._transport_process.wait(timeout=5)

        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
