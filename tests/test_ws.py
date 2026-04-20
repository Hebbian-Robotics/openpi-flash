"""Smoke test for any WebSocket deployment (EC2, Docker, Modal ASGI).

Sends random ALOHA observations to the server and prints benchmark results.
"""

import urllib.parse
from typing import cast

from openpi_client import websocket_client_policy as _websocket_client_policy

from hosting.benchmark import InferablePolicy, run_benchmark
from tests.helpers import random_observation_aloha, wait_for_server


def run(url: str, mode: str | None = None) -> None:
    parsed_server_url = urllib.parse.urlparse(url)
    if parsed_server_url.scheme not in {"ws", "wss"} or parsed_server_url.hostname is None:
        raise ValueError("Expected a WebSocket URL like ws://host[:port] or wss://host[:port].")

    is_secure = parsed_server_url.scheme == "wss"
    default_port = 443 if is_secure else 8000
    effective_port = parsed_server_url.port or default_port
    websocket_url = f"{parsed_server_url.scheme}://{parsed_server_url.hostname}"

    health_scheme = "https" if is_secure else "http"
    health_url = f"{health_scheme}://{parsed_server_url.hostname}:{effective_port}/healthz"
    wait_for_server(health_url)

    print(f"\nConnecting to {websocket_url}:{effective_port} ...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=websocket_url,
        port=effective_port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    result = run_benchmark(
        cast(InferablePolicy, policy),
        lambda: random_observation_aloha(mode=mode),
    )
    result.print_summary()
