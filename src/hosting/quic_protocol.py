"""Shared QUIC protocol types for server and client.

Defines the message framing and shared interfaces used by both
QuicPolicyServer and QuicClientPolicy.
"""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


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
