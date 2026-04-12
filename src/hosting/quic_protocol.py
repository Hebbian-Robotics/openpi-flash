"""Shared QUIC protocol types, message framing, and connection handling.

Defines the wire format, send/recv helpers, connection serving logic, and
shared interfaces used by QuicPolicyServer, QuicClientPolicy, and
DirectQuicClientPolicy. All message framing and connection protocol goes
through this module so changes are made in one place.
"""

import contextlib
import time
import traceback
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import Portal, PortalError, QuicTransportOptions

# (host, port) address of a UDP endpoint (relay server, STUN server, etc.).
UdpAddr = tuple[str, int]

# Reliable public STUN servers with global presence. The quic-portal
# defaults include stun.ekiga.net which is frequently unreachable.
DEFAULT_STUN_SERVERS: list[UdpAddr] = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun2.l.google.com", 19302),
]

# Default transport options for QUIC connections. Shared by server and client
# to ensure consistent behavior.
DEFAULT_TRANSPORT_OPTIONS = QuicTransportOptions(
    # 1 MiB initial window for large observation payloads (camera images).
    initial_window=1024 * 1024,
    # Keep-alive to detect dead connections and maintain NAT bindings.
    keep_alive_interval_secs=2,
)

DIRECT_QUIC_HANDSHAKE_KEY = "__openpi_direct_quic_handshake__"
DIRECT_QUIC_HANDSHAKE_VALUE = "hello"


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


def make_direct_quic_handshake_message() -> dict[str, str]:
    """Build the initial hello message for direct QUIC connections."""
    return {DIRECT_QUIC_HANDSHAKE_KEY: DIRECT_QUIC_HANDSHAKE_VALUE}


def is_direct_quic_handshake_message(message: dict) -> bool:
    """Return True when a decoded QUIC message is the direct-connect handshake."""
    return message.get(DIRECT_QUIC_HANDSHAKE_KEY) == DIRECT_QUIC_HANDSHAKE_VALUE


# ---------------------------------------------------------------------------
# Shared connection serving logic
# ---------------------------------------------------------------------------

# Timeout for recv() so the server can periodically check connection health.
RECV_TIMEOUT_MS = 30_000


def serve_quic_connection(
    portal: Portal,
    policy: _base_policy.BasePolicy,
    metadata: dict,
    log: Callable[[str], None],
    *,
    client_initiates_handshake: bool = False,
) -> None:
    """Handle a single QUIC client connection until it disconnects.

    Sends metadata as the first message, then enters a recv-infer-send loop.
    Used by both the direct QUIC server (serve.py) and the NAT-traversal
    QUIC server (quic_server.py).
    """
    packer = msgpack_numpy.Packer()

    if client_initiates_handshake:
        log("[quic-server] Waiting for client hello...")
        handshake_message = recv_data(portal, timeout_ms=RECV_TIMEOUT_MS)
        if handshake_message is None:
            raise TimeoutError("Timed out waiting for direct QUIC client handshake")
        if not is_direct_quic_handshake_message(handshake_message):
            raise RuntimeError(f"Unexpected initial direct QUIC message: {handshake_message!r}")
        send_data(portal, packer.pack(metadata))
        log("[quic-server] Handshake complete, sent metadata")
    else:
        # Send metadata as the first message (same as WebSocket variant).
        send_data(portal, packer.pack(metadata))
        log("[quic-server] Sent metadata, waiting for observations...")

    request_count = 0
    prev_total_time: float | None = None
    while True:
        try:
            start_time = time.monotonic()

            observation = recv_data(portal, timeout_ms=RECV_TIMEOUT_MS)
            if observation is None:
                # Timeout — client may still be there, just no request yet.
                continue

            infer_start = time.monotonic()
            action = policy.infer(observation)
            infer_ms = (time.monotonic() - infer_start) * 1000

            timing: dict[str, float] = {"infer_ms": infer_ms}
            if prev_total_time is not None:
                timing["prev_total_ms"] = prev_total_time * 1000

            response = {**action, "server_timing": timing}
            send_data(portal, packer.pack(response))
            prev_total_time = time.monotonic() - start_time
            total_ms = prev_total_time * 1000

            request_count += 1
            log(
                f"[quic-server] req #{request_count}: infer={infer_ms:.1f}ms total={total_ms:.1f}ms"
            )

        except PortalError:
            log(f"[quic-server] Client disconnected after {request_count} requests")
            break
        except Exception:
            log(f"[quic-server] Error after {request_count} requests:\n{traceback.format_exc()}")
            with contextlib.suppress(PortalError):
                send_error(portal, traceback.format_exc())
            break

    with contextlib.suppress(PortalError):
        portal.close()
