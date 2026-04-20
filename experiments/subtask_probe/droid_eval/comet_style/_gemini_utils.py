"""Shared helpers for Gemini-backed subtask generators.

Split out of generate_subtasks_gemini.py so both the stateless generator and
the Comet-style hierarchical reasoner use the same PNG encoding and 429
retry-with-backoff logic.
"""

from __future__ import annotations

import io
import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_RETRY_DELAY_PATTERN = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")

T = TypeVar("T")


def encode_png(image: np.ndarray) -> bytes:
    """PNG-encode an HxWx3 uint8 RGB image."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def parse_retry_delay_seconds(exc: Exception) -> float | None:
    """Extract the server's suggested retry delay (seconds) from a 429 error string."""
    match = _RETRY_DELAY_PATTERN.search(str(exc))
    if match is None:
        return None
    return float(match.group(1))


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


# Message fragments that indicate a *transient* network problem worth retrying
# (vs. a deterministic API error like 400 or schema validation). Matched against
# str(exc) so we're robust to the specific exception classes google-genai uses
# across versions.
_TRANSIENT_NETWORK_FRAGMENTS = (
    "server disconnected",
    "remote end closed connection",
    "connection reset",
    "connection aborted",
    "read timed out",
    "readtimeout",
    "connecttimeout",
    "503",  # service unavailable
    "502",  # bad gateway
    "504",  # gateway timeout
    "500",  # internal server error
)


def is_transient_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(fragment in message for fragment in _TRANSIENT_NETWORK_FRAGMENTS)


def call_with_retry(
    call: Callable[[], T],
    *,
    max_retries: int,
) -> T:
    """Call a Gemini API function with retry on 429s and transient network errors.

    * 429 / RESOURCE_EXHAUSTED -> sleep for the server-suggested ``retryDelay``
      (or a capped exponential fallback) and try again.
    * Transient network failures (server disconnect, 5xx, read timeout) ->
      exponential backoff and try again, because these previously caused a
      50+ minute hang when they surfaced at the wrong point in the Gemini
      client's internal retry logic.
    * Other exceptions propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            rate_limited = is_rate_limit_error(exc)
            transient = is_transient_network_error(exc)
            if (not rate_limited and not transient) or attempt == max_retries:
                raise
            last_exc = exc
            if rate_limited:
                # Server sends retryDelay like '3s'; add jitter so a stampede
                # of workers doesn't all wake up at the same instant and trip
                # the quota again.
                base_delay = parse_retry_delay_seconds(exc) or min(2**attempt, 30.0)
                sleep_s = base_delay + (0.2 * attempt)
                logger.info(
                    "Rate-limited (attempt %d/%d); sleeping %.1fs before retry",
                    attempt + 1,
                    max_retries,
                    sleep_s,
                )
            else:
                sleep_s = min(2**attempt, 10.0) + (0.2 * attempt)
                logger.warning(
                    "Transient network error (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    sleep_s,
                )
            time.sleep(sleep_s)
    raise RuntimeError("retry loop exited without returning") from last_exc
