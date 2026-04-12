"""QUIC-based client policy using quic-portal.

Connects to a QuicPolicyServer via QUIC with automatic NAT traversal.
Uses a shared Modal Dict for peer discovery and UDP hole punching.

Note: quic-portal is experimental and not intended for production use.
NAT traversal only works with "easy" NATs — symmetric NATs (common in
corporate networks) may fail. Fall back to the WebSocket tunnel variant
if connectivity issues arise.
"""

import contextlib
import logging
import time

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import Portal, PortalError, QuicTransportOptions
from typing_extensions import override

from hosting.quic_protocol import (
    DEFAULT_STUN_SERVERS,
    DEFAULT_TRANSPORT_OPTIONS,
    PortalDictLike,
    UdpAddr,
    recv_data,
    send_data,
)
from hosting.relay import register_with_relay

logger = logging.getLogger(__name__)


class QuicClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over QUIC.

    See QuicPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        portal_dict: PortalDictLike,
        local_port: int = 5556,
        transport_options: QuicTransportOptions | None = None,
        stun_servers: list[UdpAddr] | None = None,
        max_connect_attempts: int = 30,
        relay_addr: UdpAddr | None = None,
        relay_only: bool = False,
    ) -> None:
        self._portal_dict = portal_dict
        self._local_port = local_port
        self._stun_servers = stun_servers or DEFAULT_STUN_SERVERS
        self._transport_options = transport_options or DEFAULT_TRANSPORT_OPTIONS
        self._max_connect_attempts = max_connect_attempts
        self._relay_addr = relay_addr
        self._relay_only = relay_only
        self._packer = msgpack_numpy.Packer()
        self._portal, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _create_portal(self) -> Portal:
        """Create a portal, falling back to relay if hole punching fails.

        Relay fallback is available if either ``relay_addr`` was passed to the
        constructor OR the server wrote relay info to the Modal Dict.
        """
        if self._relay_only:
            logger.info("Relay-only mode, skipping hole punch")
            return self._create_portal_via_relay()

        # If the server already wrote relay info, it's in relay-only mode —
        # skip hole punch since the server won't participate in it.
        if "relay_session" in self._portal_dict:
            logger.info("Server is using relay, connecting via relay")
            return self._create_portal_via_relay()

        try:
            portal = Portal.create_client(
                dict=self._portal_dict,
                local_port=self._local_port,
                stun_servers=self._stun_servers,
                transport_options=self._transport_options,
            )
            logger.info("Connected via QUIC portal (direct)")
            return portal
        except (PortalError, ConnectionError):
            # Relay fallback is available if the server wrote relay info to the dict.
            has_relay = self._relay_addr is not None or "relay_session" in self._portal_dict
            if not has_relay:
                raise
            logger.warning("Hole punch failed, falling back to UDP relay")
            return self._create_portal_via_relay()

    def _create_portal_via_relay(self) -> Portal:
        """Create a portal through the UDP relay server.

        The keepalive socket stays alive alongside Quinn (via SO_REUSEPORT in
        quic-portal) to maintain the NAT mapping so the relay can reach us.
        """
        print("[quic-client] Waiting for server to create relay session...")
        while "relay_session" not in self._portal_dict:
            time.sleep(0.5)
        session_id = self._portal_dict["relay_session"]
        relay_addr = tuple(self._portal_dict["relay_addr"])
        relay_ip, relay_port = relay_addr[0], relay_addr[1]
        print(f"[quic-client] Relay session found at {relay_ip}:{relay_port}, registering...")

        keepalive = register_with_relay((relay_ip, relay_port), session_id, self._local_port)

        try:
            print("[quic-client] Connecting to server through relay...")
            portal = Portal()
            portal.connect(relay_ip, relay_port, self._local_port, self._transport_options)
            print("[quic-client] Connected via relay")
            return portal
        finally:
            keepalive.stop()

    def _wait_for_server(self) -> tuple[Portal, dict]:
        """Connect to the QUIC server, retrying until it's available or max attempts reached."""
        logger.info(
            "Connecting to QUIC portal server (max %d attempts)...", self._max_connect_attempts
        )
        for attempt in range(1, self._max_connect_attempts + 1):
            try:
                portal = self._create_portal()

                # First message from server is metadata.
                metadata = recv_data(portal, timeout_ms=30_000)
                if metadata is None:
                    logger.warning(
                        "Timeout waiting for server metadata, retrying... (%d/%d)",
                        attempt,
                        self._max_connect_attempts,
                    )
                    portal.close()
                    time.sleep(2)
                    continue

                logger.info("Connected to QUIC portal server")
                return portal, metadata

            except PortalError:
                logger.info(
                    "Server not ready, retrying in 5s... (%d/%d)",
                    attempt,
                    self._max_connect_attempts,
                )
                time.sleep(5)

        raise ConnectionError(
            f"Failed to connect to QUIC server after {self._max_connect_attempts} attempts"
        )

    @override
    def infer(self, obs: dict) -> dict:
        send_data(self._portal, self._packer.pack(obs))

        response = recv_data(self._portal)
        if response is None:
            raise ConnectionError("QUIC connection lost (recv returned None)")
        return response

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Close the QUIC connection."""
        with contextlib.suppress(PortalError):
            self._portal.close()
