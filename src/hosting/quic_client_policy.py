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

from hosting.quic_protocol import PortalDictLike, QuicMessageType

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
        max_connect_attempts: int = 30,
    ) -> None:
        self._portal_dict = portal_dict
        self._local_port = local_port
        self._transport_options = transport_options or QuicTransportOptions(
            initial_window=1024 * 1024,
            keep_alive_interval_secs=2,
        )
        self._max_connect_attempts = max_connect_attempts
        self._packer = msgpack_numpy.Packer()
        self._portal, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[Portal, dict]:
        """Connect to the QUIC server, retrying until it's available or max attempts reached."""
        logger.info(
            "Connecting to QUIC portal server (max %d attempts)...", self._max_connect_attempts
        )
        for attempt in range(1, self._max_connect_attempts + 1):
            try:
                portal = Portal.create_client(
                    dict=self._portal_dict,
                    local_port=self._local_port,
                    transport_options=self._transport_options,
                )

                # First message from server is metadata.
                raw_metadata = portal.recv(timeout_ms=30_000)
                if raw_metadata is None:
                    logger.warning(
                        "Timeout waiting for server metadata, retrying... (%d/%d)",
                        attempt,
                        self._max_connect_attempts,
                    )
                    portal.close()
                    time.sleep(2)
                    continue

                if len(raw_metadata) < 1 or raw_metadata[0:1] != QuicMessageType.DATA.value:
                    raise RuntimeError(
                        f"Expected data message for metadata, got prefix: {raw_metadata[0:1]!r}"
                    )

                metadata = msgpack_numpy.unpackb(raw_metadata[1:])
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
        data = self._packer.pack(obs)
        self._portal.send(QuicMessageType.DATA.value + data)

        response = self._portal.recv()
        if response is None:
            raise ConnectionError("QUIC connection lost (recv returned None)")

        if len(response) < 1:
            raise RuntimeError("Received empty response from server")

        message_type = response[0:1]
        message_body = response[1:]

        if message_type == QuicMessageType.ERROR.value:
            raise RuntimeError(f"Error in inference server:\n{message_body.decode('utf-8')}")

        if message_type != QuicMessageType.DATA.value:
            raise RuntimeError(f"Unexpected message type from server: {message_type!r}")

        return msgpack_numpy.unpackb(message_body)

    @override
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Close the QUIC connection."""
        with contextlib.suppress(PortalError):
            self._portal.close()
