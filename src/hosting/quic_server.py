"""QUIC-based inference server using quic-portal.

Serves an openpi policy over a QUIC connection using the quic-portal library
for automatic NAT traversal via STUN + UDP hole punching. This provides lower
latency than TCP-based WebSocket transport by avoiding head-of-line blocking.

Note: quic-portal is experimental and not intended for production use.
Only one client can be connected at a time (typical for robotics — one robot per GPU).
"""

import contextlib
import datetime
import time
import traceback
import uuid
from typing import ClassVar

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import Portal, PortalError, QuicTransportOptions

from hosting.quic_protocol import (
    PortalDictLike,
    UdpAddr,
    recv_data,
    send_data,
    send_error,
)
from hosting.relay import register_with_relay

# Timeout for recv() so the server can periodically check connection health.
_RECV_TIMEOUT_MS = 30_000


def _log(msg: str) -> None:
    """Print with UTC timestamp for correlating with relay logs."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S.%f")[:-3]
    _log(f"{ts} {msg}")


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
    DEFAULT_STUN_SERVERS: ClassVar[list[UdpAddr]] = [
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
        stun_servers: list[UdpAddr] | None = None,
        relay_addr: UdpAddr | None = None,
        relay_only: bool = False,
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
        self._relay_addr = relay_addr
        self._relay_only = relay_only

    def _create_portal(self) -> Portal:
        """Create a portal, falling back to relay if hole punching fails."""
        if self._relay_only:
            if self._relay_addr is None:
                raise ConnectionError("relay_only=True but no relay_addr configured")
            _log("[quic-server] Relay-only mode, skipping hole punch")
            return self._create_portal_via_relay()

        try:
            portal = Portal.create_server(
                dict=self._portal_dict,
                local_port=self._local_port,
                stun_servers=self._stun_servers,
                transport_options=self._transport_options,
            )
            _log("[quic-server] Client connected (direct)")
            return portal
        except (PortalError, ConnectionError):
            if self._relay_addr is None:
                raise
            _log("[quic-server] Hole punch failed, falling back to UDP relay")
            return self._create_portal_via_relay()

    def _create_portal_via_relay(self) -> Portal:
        """Create a portal through the UDP relay server.

        The keepalive socket stays alive alongside Quinn (via SO_REUSEPORT in
        quic-portal) to maintain the NAT mapping so the relay can reach us.
        """
        assert self._relay_addr is not None

        session_id = str(uuid.uuid4())
        self._portal_dict["relay_session"] = session_id
        self._portal_dict["relay_addr"] = self._relay_addr
        _log("[quic-server] Relay session created, registering...")

        keepalive = register_with_relay(self._relay_addr, session_id, self._local_port)

        try:
            _log("[quic-server] Waiting for client to connect through relay...")
            portal = Portal()
            portal.listen(self._local_port, self._transport_options)
            _log("[quic-server] Client connected (relayed)")
            return portal
        finally:
            keepalive.stop()

    def serve_forever(self) -> None:
        """Block forever, accepting and serving one client at a time.

        After a client disconnects, a new Portal is created to accept the next client.
        This is necessary because each Portal instance handles a single peer connection.
        """
        while True:
            try:
                _log("[quic-server] Waiting for client...")
                portal = self._create_portal()

                self._serve_connection(portal)
            except PortalError as e:
                _log(f"[quic-server] Portal error, will retry: {e}")
            except Exception:
                _log(f"[quic-server] Error, will retry:\n{traceback.format_exc()}")
            finally:
                time.sleep(1)  # Brief pause before accepting next client.

    def _serve_connection(self, portal: Portal) -> None:
        """Handle a single client connection until it disconnects."""
        packer = msgpack_numpy.Packer()

        # Send metadata as the first message (same as WebSocket variant).
        send_data(portal, packer.pack(self._metadata))
        _log("[quic-server] Sent metadata, waiting for observations...")

        request_count = 0
        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()

                observation = recv_data(portal, timeout_ms=_RECV_TIMEOUT_MS)
                if observation is None:
                    # Timeout — client may still be there, just no request yet.
                    continue

                infer_start = time.monotonic()
                action = self._policy.infer(observation)
                infer_ms = (time.monotonic() - infer_start) * 1000

                timing: dict[str, float] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    timing["prev_total_ms"] = prev_total_time * 1000

                response = {**action, "server_timing": timing}
                send_data(portal, packer.pack(response))
                prev_total_time = time.monotonic() - start_time
                total_ms = prev_total_time * 1000

                request_count += 1
                _log(
                    f"[quic-server] req #{request_count}: infer={infer_ms:.1f}ms total={total_ms:.1f}ms"
                )

            except PortalError:
                _log(f"[quic-server] Client disconnected after {request_count} requests")
                break
            except Exception:
                _log(
                    f"[quic-server] Error after {request_count} requests:\n{traceback.format_exc()}"
                )
                with contextlib.suppress(PortalError):
                    send_error(portal, traceback.format_exc())
                break

        with contextlib.suppress(PortalError):
            portal.close()
