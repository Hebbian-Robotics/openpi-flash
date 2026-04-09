"""QUIC-based inference server using quic-portal.

Serves an openpi policy over a QUIC connection using the quic-portal library
for automatic NAT traversal via STUN + UDP hole punching. This provides lower
latency than TCP-based WebSocket transport by avoiding head-of-line blocking.

Note: quic-portal is experimental and not intended for production use.
Only one client can be connected at a time (typical for robotics — one robot per GPU).
"""

import contextlib
import logging
import time
import traceback
from typing import ClassVar

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import Portal, PortalError, QuicTransportOptions

from hosting.quic_protocol import PortalDictLike, QuicMessageType

logger = logging.getLogger(__name__)

# Timeout for recv() so the server can periodically check connection health.
_RECV_TIMEOUT_MS = 30_000


def _send_data(portal: Portal, data: bytes) -> None:
    """Send a msgpack-encoded data message with the data type prefix."""
    portal.send(QuicMessageType.DATA.value + data)


def _send_error(portal: Portal, error_message: str) -> None:
    """Send an error message with the error type prefix."""
    portal.send(QuicMessageType.ERROR.value + error_message.encode("utf-8"))


class QuicPolicyServer:
    """Serves a policy over QUIC using quic-portal for NAT traversal.

    Uses a Modal Dict for peer discovery and UDP hole punching coordination.
    The server writes its STUN-discovered address to the Dict; clients read it
    to establish a direct QUIC connection.

    Protocol (same wire format as WebsocketPolicyServer, with type prefix):
        1. Server sends metadata (data prefix + msgpack bytes)
        2. Client sends observation (data prefix + msgpack bytes)
        3. Server sends action + timing (data prefix + msgpack bytes)
        4. Repeat 2-3
    """

    # Reliable public STUN servers with global presence. The quic-portal
    # defaults include stun.ekiga.net which is frequently unreachable.
    DEFAULT_STUN_SERVERS: ClassVar[list[tuple[str, int]]] = [
        ("stun.l.google.com", 19302),
        ("stun1.l.google.com", 19302),
        ("stun2.l.google.com", 19302),
    ]

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        portal_dict: PortalDictLike,
        metadata: dict | None = None,
        local_port: int = 5555,
        transport_options: QuicTransportOptions | None = None,
        stun_servers: list[tuple[str, int]] | None = None,
    ) -> None:
        self._policy = policy
        self._portal_dict = portal_dict
        self._metadata = metadata or {}
        self._local_port = local_port
        self._stun_servers = stun_servers or self.DEFAULT_STUN_SERVERS
        self._transport_options = transport_options or QuicTransportOptions(
            # 1 MiB initial window for large observation payloads (camera images).
            initial_window=1024 * 1024,
            # Keep-alive to maintain NAT bindings.
            keep_alive_interval_secs=2,
        )

    def serve_forever(self) -> None:
        """Block forever, accepting and serving one client at a time.

        After a client disconnects, a new Portal is created to accept the next client.
        This is necessary because each Portal instance handles a single peer connection.
        """
        while True:
            try:
                logger.info("Creating QUIC portal server (waiting for client)...")
                portal = Portal.create_server(
                    dict=self._portal_dict,
                    local_port=self._local_port,
                    stun_servers=self._stun_servers,
                    transport_options=self._transport_options,
                )
                logger.info("Client connected via QUIC portal")

                self._serve_connection(portal)
            except PortalError:
                logger.exception("Portal error, will retry")
            except Exception:
                logger.exception("Unexpected error in serve loop, will retry")
            finally:
                time.sleep(1)  # Brief pause before accepting next client.

    def _serve_connection(self, portal: Portal) -> None:
        """Handle a single client connection until it disconnects."""
        packer = msgpack_numpy.Packer()

        # Send metadata as the first message (same as WebSocket variant).
        _send_data(portal, packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()

                raw_message = portal.recv(timeout_ms=_RECV_TIMEOUT_MS)
                if raw_message is None:
                    # Timeout — client may still be there, just no request yet.
                    continue

                if len(raw_message) < 1:
                    logger.warning("Received empty message, ignoring")
                    continue

                message_type = raw_message[0:1]
                message_body = raw_message[1:]

                if message_type != QuicMessageType.DATA.value:
                    logger.warning("Received unexpected message type: %r", message_type)
                    continue

                observation = msgpack_numpy.unpackb(message_body)

                infer_time = time.monotonic()
                action = self._policy.infer(observation)
                infer_time = time.monotonic() - infer_time

                timing: dict[str, float] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    timing["prev_total_ms"] = prev_total_time * 1000

                response = {**action, "server_timing": timing}
                _send_data(portal, packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except PortalError:
                logger.info("Client disconnected (portal error)")
                break
            except Exception:
                logger.exception("Error during inference")
                with contextlib.suppress(PortalError):
                    _send_error(portal, traceback.format_exc())
                break

        with contextlib.suppress(PortalError):
            portal.close()
