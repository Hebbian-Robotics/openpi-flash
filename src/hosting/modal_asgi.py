"""ASGI adapter for serving openpi policies on Modal.

Translates the WebSocket protocol from openpi's WebsocketPolicyServer
into a Starlette ASGI app compatible with Modal's @modal.asgi_app().
"""

import asyncio
import logging
import time
import traceback
from typing import TypedDict

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class OpenPIServerTiming(TypedDict, total=False):
    """Server-side timing for OpenPI inference requests."""

    infer_ms: float
    prev_total_ms: float


def create_openpi_asgi_app(
    policy: _base_policy.BasePolicy,
    metadata: dict | None = None,
) -> Starlette:
    """Create a Starlette ASGI app that serves an openpi policy over WebSocket.

    Same protocol as openpi's WebsocketPolicyServer: msgpack-encoded
    observations in, msgpack-encoded actions out, with server_timing metadata.
    """
    metadata = metadata or {}

    async def healthz(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK\n")

    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()

        packer = msgpack_numpy.Packer()
        await websocket.send_bytes(packer.pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                observation = msgpack_numpy.unpackb(await websocket.receive_bytes())

                infer_time = time.monotonic()
                action = await asyncio.to_thread(policy.infer, observation)
                infer_time = time.monotonic() - infer_time

                timing = OpenPIServerTiming(infer_ms=infer_time * 1000)
                if prev_total_time is not None:
                    timing["prev_total_ms"] = prev_total_time * 1000
                action["server_timing"] = timing

                await websocket.send_bytes(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except WebSocketDisconnect:
                logger.info("Connection closed")
                break
            except Exception:
                await websocket.send_bytes(traceback.format_exc().encode())
                await websocket.close(code=1011, reason="Internal server error.")
                raise

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            WebSocketRoute("/", websocket_handler),
        ],
    )
