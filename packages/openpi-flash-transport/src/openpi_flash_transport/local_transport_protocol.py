"""Shared framed protocol helpers for the local transport Unix socket.

Both the server-side and client-side transport processes speak the same
length-prefixed protocol over a Unix domain socket to their Python peers.
This module holds the shared message type constants and framing helpers
for the Python side.
"""

from __future__ import annotations

import socket
import struct
from enum import IntEnum


class TransportRequestType(IntEnum):
    """Request message types sent to the transport over the local socket."""

    METADATA = 0x01
    INFER = 0x02
    RESET = 0x03


class TransportResponseType(IntEnum):
    """Response message types returned by the transport over the local socket."""

    METADATA = 0x11
    INFER = 0x12
    ERROR = 0x13
    RESET = 0x14


def recv_exactly(stream_socket: socket.socket, num_bytes: int) -> bytes | None:
    """Read exactly ``num_bytes`` from a stream socket or return ``None`` on EOF."""
    received_chunks = bytearray()
    while len(received_chunks) < num_bytes:
        chunk = stream_socket.recv(num_bytes - len(received_chunks))
        if not chunk:
            return None
        received_chunks.extend(chunk)
    return bytes(received_chunks)


def recv_framed_message(stream_socket: socket.socket) -> bytes | None:
    """Receive one length-prefixed message from a stream socket."""
    raw_length_prefix = recv_exactly(stream_socket, 4)
    if raw_length_prefix is None:
        return None

    message_length = struct.unpack(">I", raw_length_prefix)[0]
    if message_length == 0:
        return b""

    payload = recv_exactly(stream_socket, message_length)
    if payload is None:
        raise ConnectionError("Unexpected EOF while reading framed transport message")
    return payload


def send_framed_message(stream_socket: socket.socket, payload: bytes) -> None:
    """Send one length-prefixed message over a stream socket."""
    stream_socket.sendall(struct.pack(">I", len(payload)))
    if payload:
        stream_socket.sendall(payload)
