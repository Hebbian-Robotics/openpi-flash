"""QUIC client policy for direct connections (Docker/EC2).

Connects to a QUIC server at a known host:port via Portal.connect().
No Modal Dict, no STUN, no relay — just a direct UDP connection.

For Modal deployments that use STUN/relay, use QuicClientPolicy instead.
"""

import contextlib
import logging

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from quic_portal import Portal, PortalError, QuicTransportOptions
from typing_extensions import override

from hosting.quic_protocol import recv_data, send_data

logger = logging.getLogger(__name__)


class DirectQuicClientPolicy(_base_policy.BasePolicy):
    """Connects to a QUIC server at a known address via Portal.connect()."""

    def __init__(
        self,
        host: str,
        port: int = 5555,
        local_port: int = 5556,
        transport_options: QuicTransportOptions | None = None,
    ) -> None:
        self._transport_options = transport_options or QuicTransportOptions(
            initial_window=1024 * 1024,
            keep_alive_interval_secs=2,
        )
        self._packer = msgpack_numpy.Packer()

        logger.info("Connecting to %s:%d via QUIC...", host, port)
        self._portal = Portal()
        self._portal.connect(host, port, local_port, self._transport_options)
        logger.info("Connected")

        # First message from server is metadata.
        metadata = recv_data(self._portal, timeout_ms=30_000)
        if metadata is None:
            raise ConnectionError("Timeout waiting for server metadata")
        self._server_metadata = metadata
        logger.info("Received server metadata")

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    @override
    def infer(self, obs: dict) -> dict:
        send_data(self._portal, self._packer.pack(obs))

        response = recv_data(self._portal, timeout_ms=30_000)
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
