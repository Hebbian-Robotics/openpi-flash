"""Hosted inference server entry point.

Wraps the existing WebSocket policy server with:
- Customer authentication via API keys
- Per-request tagging (customer_id, request_id, model_version)
- Concurrency control (semaphore-based, rejects when busy)
- Structured JSON logging for CloudWatch
"""

import asyncio
import json
import logging
import socket
import time
import traceback
import uuid

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from hosting.auth import create_request_handler
from hosting.auth import pop_customer_id
from hosting.config import CustomerId
from hosting.config import ServiceConfig
from hosting.config import load_config

logger = logging.getLogger(__name__)


class HostedPolicyServer:
    """WebSocket policy server with authentication, concurrency control, and request tagging.

    Extends the serving pattern from openpi's WebsocketPolicyServer with hosting features.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        service_config: ServiceConfig,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._config = service_config
        self._metadata = metadata or {}
        self._semaphore = asyncio.Semaphore(service_config.max_concurrent_requests)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        process_request = create_request_handler(self._config)
        async with _server.serve(
            self._handler,
            "0.0.0.0",
            self._config.port,
            compression=None,
            max_size=None,
            process_request=process_request,
        ) as server:
            logger.info(
                "Server listening on port %d (model_version=%s, max_concurrent=%d)",
                self._config.port,
                self._config.model_version,
                self._config.max_concurrent_requests,
            )
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        customer_id: CustomerId = pop_customer_id(websocket) or CustomerId("unknown")
        connection_id = str(uuid.uuid4())[:8]

        logger.info(
            "Connection opened",
            extra={"customer_id": customer_id, "connection_id": connection_id},
        )

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                request_id = str(uuid.uuid4())

                obs = msgpack_numpy.unpackb(await websocket.recv())

                # Acquire the concurrency semaphore. If the server is busy, reject.
                if not self._semaphore._value:  # noqa: SLF001
                    logger.warning(
                        "Server busy, rejecting request",
                        extra={
                            "customer_id": customer_id,
                            "request_id": request_id,
                        },
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.TRY_AGAIN_LATER,
                        reason="Server busy — max concurrent requests reached.",
                    )
                    break

                async with self._semaphore:
                    infer_time = time.monotonic()
                    action = await asyncio.to_thread(self._policy.infer, obs)
                    infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Tag the response with hosting metadata.
                action["hosting"] = {
                    "customer_id": customer_id,
                    "request_id": request_id,
                    "model_version": self._config.model_version,
                }

                logger.info(
                    "Inference complete",
                    extra={
                        "customer_id": customer_id,
                        "request_id": request_id,
                        "model_version": self._config.model_version,
                        "infer_ms": round(infer_time * 1000, 1),
                    },
                )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(
                    "Connection closed",
                    extra={"customer_id": customer_id, "connection_id": connection_id},
                )
                break
            except Exception:
                logger.exception(
                    "Error during inference",
                    extra={"customer_id": customer_id, "connection_id": connection_id},
                )
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# Known extra fields for structured logging. Avoids leaking arbitrary record attributes.
_STRUCTURED_LOG_FIELDS = ("customer_id", "connection_id", "request_id", "model_version", "infer_ms")


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _STRUCTURED_LOG_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging() -> None:
    """Configure structured JSON logging for CloudWatch."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


def main() -> None:
    setup_logging()

    service_config = load_config()
    logger.info(
        "Loading model: config=%s, checkpoint=%s",
        service_config.model_config_name,
        service_config.checkpoint_dir,
    )

    train_config = _config.get_config(service_config.model_config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        service_config.checkpoint_dir,
        default_prompt=service_config.default_prompt,
    )
    policy_metadata = train_config.policy_metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating hosted server (host=%s, ip=%s)", hostname, local_ip)

    server = HostedPolicyServer(
        policy=policy,
        service_config=service_config,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
