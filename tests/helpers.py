"""Shared utilities for smoke test scripts."""

import time
from typing import Final

import httpx

from hosting.warmup import make_aloha_observation

SERVER_READINESS_TIMEOUT_SECONDS: Final[float] = 300.0
SERVER_READINESS_POLL_INTERVAL_SECONDS: Final[float] = 5.0


def random_observation_aloha() -> dict:
    """Generate a random ALOHA observation for smoke testing."""
    return make_aloha_observation(prompt="do something")


def wait_for_server(
    health_url: str,
    timeout_seconds: float = SERVER_READINESS_TIMEOUT_SECONDS,
) -> None:
    """Poll a health endpoint until the server is ready or the timeout expires."""
    print(f"Health check {health_url} (waiting for server to be ready) ...")
    readiness_deadline = time.monotonic() + timeout_seconds
    last_status_code: int | None = None
    last_error_message: str | None = None

    while True:
        try:
            response = httpx.get(health_url, timeout=30.0)
            if response.is_success:
                print(f"Health check: {response.text.strip()}")
                return
            last_status_code = response.status_code
            last_error_message = None
            print(f"  Server returned {response.status_code}, retrying ...")
        except httpx.HTTPError as exc:
            last_error_message = str(exc)
            print(f"  {exc}, retrying ...")

        if time.monotonic() >= readiness_deadline:
            if last_status_code is not None:
                raise TimeoutError(
                    "Server did not become ready before timeout "
                    f"({timeout_seconds:.0f}s). Last HTTP status: {last_status_code}."
                )
            if last_error_message is not None:
                raise TimeoutError(
                    "Server did not become ready before timeout "
                    f"({timeout_seconds:.0f}s). Last error: {last_error_message}."
                )
            raise TimeoutError(
                f"Server did not become ready before timeout ({timeout_seconds:.0f}s)."
            )

        time.sleep(SERVER_READINESS_POLL_INTERVAL_SECONDS)
