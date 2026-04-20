"""UDP relay registration helper for NAT traversal fallback.

When direct STUN + UDP hole punching fails (e.g. symmetric NATs), both peers
can register with a shared UDP relay server. The relay forwards datagrams
bidirectionally between the two peers, allowing QUIC to connect through it.

Protocol:
    1. Peer sends ``REG:<session_id>\\n`` to the relay's UDP address.
    2. Relay responds ``ACK\\n`` once registered.
    3. After both peers register for the same session, the relay forwards
       all subsequent UDP datagrams between them transparently.
"""

import socket as socketlib
import threading

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from hosting.quic_protocol import UdpAddr


class _RelayRegistrationTimeout(TimeoutError):
    """Internal signal that a single registration attempt timed out.

    Distinct from the socket's built-in ``TimeoutError`` only so we can log
    the attempt counter cleanly via tenacity's ``before_sleep`` hook. Both
    are recognised as retry-able by tenacity below.
    """


# Re-registration interval to keep the NAT mapping alive.
_KEEPALIVE_INTERVAL_SECS = 3.0


class RelayKeepalive:
    """Maintains a NAT mapping to the relay by periodically re-sending REG packets.

    The keepalive socket uses SO_REUSEPORT so Quinn can bind the same port
    alongside it. Call ``stop()`` once the QUIC connection is established.
    """

    def __init__(self, sock: socketlib.socket, relay_addr: UdpAddr, reg_msg: bytes) -> None:
        self._sock = sock
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._keepalive_loop,
            args=(relay_addr, reg_msg),
            daemon=True,
            name="relay-keepalive",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the keepalive thread and close the socket."""
        self._stop_event.set()
        self._sock.close()
        print("[relay] Keepalive stopped")

    def _keepalive_loop(self, relay_addr: UdpAddr, reg_msg: bytes) -> None:
        keepalive_count = 0
        while not self._stop_event.is_set():
            try:
                self._sock.sendto(reg_msg, relay_addr)
                keepalive_count += 1
                if keepalive_count == 1 or keepalive_count % 10 == 0:
                    print(f"[relay] Keepalive #{keepalive_count} sent")
            except OSError:
                break
            self._stop_event.wait(timeout=_KEEPALIVE_INTERVAL_SECS)


def register_with_relay(
    relay_addr: UdpAddr,
    session_id: str,
    local_port: int,
    max_retries: int = 5,
) -> RelayKeepalive:
    """Register this peer with the UDP relay and start keepalive.

    Returns a ``RelayKeepalive`` handle that continuously re-sends registration
    packets to maintain the NAT mapping. Call ``handle.stop()`` once the QUIC
    connection is established.

    Args:
        relay_addr: ``(host, port)`` of the UDP relay server.
        session_id: Shared session identifier (both peers must use the same one).
        local_port: Local UDP port to bind to (must match the QUIC port).
        max_retries: Number of registration attempts before giving up.

    Returns:
        A ``RelayKeepalive`` handle. Call ``.stop()`` when done.

    Raises:
        ConnectionError: If the relay does not ACK after ``max_retries`` attempts.
    """
    sock = socketlib.socket(socketlib.AF_INET, socketlib.SOCK_DGRAM)
    sock.setsockopt(socketlib.SOL_SOCKET, socketlib.SO_REUSEADDR, 1)
    sock.setsockopt(socketlib.SOL_SOCKET, socketlib.SO_REUSEPORT, 1)
    sock.bind(("0.0.0.0", local_port))
    sock.settimeout(2.0)

    msg = f"REG:{session_id}\n".encode()

    def _log_retry(retry_state: object) -> None:
        attempt_number = getattr(retry_state, "attempt_number", 0)
        print(f"[relay] Registration attempt {attempt_number}/{max_retries} timed out")

    @retry(
        stop=stop_after_attempt(max_retries),
        retry=retry_if_exception_type((TimeoutError, _RelayRegistrationTimeout)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _attempt() -> None:
        sock.sendto(msg, relay_addr)
        # Socket timeout is 2s; recvfrom raises TimeoutError which tenacity retries.
        data, _ = sock.recvfrom(64)
        if data.strip() != b"ACK":
            # Unexpected payload — treat as a failed attempt so we retry rather
            # than accept a partial registration.
            raise _RelayRegistrationTimeout(f"relay returned non-ACK payload: {data!r}")

    try:
        _attempt()
    except (TimeoutError, _RelayRegistrationTimeout, RetryError) as exc:
        sock.close()
        raise ConnectionError(
            f"Failed to register with relay {relay_addr[0]}:{relay_addr[1]} "
            f"after {max_retries} attempts"
        ) from exc

    print(f"[relay] Registered with {relay_addr[0]}:{relay_addr[1]} for session {session_id}")
    return RelayKeepalive(sock, relay_addr, msg)
