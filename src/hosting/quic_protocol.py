"""Shared QUIC protocol types and message framing for server and client.

Defines the wire format, send/recv helpers, and shared interfaces used by
QuicPolicyServer, QuicClientPolicy, and DirectQuicClientPolicy. All message
framing goes through this module so changes to the protocol are made in one
place.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from openpi_client import msgpack_numpy
from quic_portal import Portal

# (host, port) address of a UDP endpoint (relay server, STUN server, etc.).
UdpAddr = tuple[str, int]


class QuicMessageType(Enum):
    """Message type prefixes to distinguish data from errors over raw QUIC bytes.

    WebSocket has text vs binary frame types; QUIC portal only has raw bytes,
    so we prefix each message with a single byte to indicate its type.
    """

    DATA = b"\x00"
    ERROR = b"\x01"


@runtime_checkable
class PortalDictLike(Protocol):
    """Minimal interface for a Modal Dict used for QUIC peer discovery.

    Modal Dicts act as shared key-value stores for coordinating STUN-discovered
    addresses between server and client.
    """

    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: str) -> bool: ...


# ---------------------------------------------------------------------------
# Message framing helpers
# ---------------------------------------------------------------------------


def send_data(portal: Portal, data: bytes) -> None:
    """Send a msgpack-encoded data message with the DATA type prefix."""
    portal.send(QuicMessageType.DATA.value + data)


def send_error(portal: Portal, error_message: str) -> None:
    """Send an error message with the ERROR type prefix."""
    portal.send(QuicMessageType.ERROR.value + error_message.encode("utf-8"))


def recv_data(portal: Portal, *, timeout_ms: int = 30_000) -> dict | None:
    """Receive and unpack a DATA message, or return None on timeout.

    Raises:
        RuntimeError: On empty messages, ERROR messages, or unexpected types.
    """
    raw = portal.recv(timeout_ms=timeout_ms)
    if raw is None:
        return None

    if len(raw) < 1:
        raise RuntimeError("Received empty message")

    message_type = raw[0:1]
    message_body = raw[1:]

    if message_type == QuicMessageType.ERROR.value:
        raise RuntimeError(f"Error from remote:\n{message_body.decode('utf-8')}")
    if message_type != QuicMessageType.DATA.value:
        raise RuntimeError(f"Unexpected message type: {message_type!r}")

    return msgpack_numpy.unpackb(message_body)
