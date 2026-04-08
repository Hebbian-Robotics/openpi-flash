"""VLASH hosted inference server with async chunk pre-computation.

Serves VLASH policies over WebSocket with the same auth and logging as the
OpenPI server, but with a VLASH-specific protocol that preserves the async
inference pipeline: the server pre-computes the next action chunk while the
client executes the current one.

Protocol:
    1. Client connects with Bearer auth.
    2. Server sends metadata (msgpack): policy_type, n_action_steps, etc.
    3. Client sends observation (msgpack dict of numpy arrays).
    4. Server returns action chunk + timing metadata.
       - On the first request, inference is synchronous.
       - On subsequent requests, the chunk was pre-computed in the background,
         so the response is near-instant.
    5. After each response, the server starts pre-computing the next chunk
       using future-state conditioning (last action of the returned chunk
       as the predicted robot state).
"""

import asyncio
from copy import copy
import logging
import time
import traceback
from typing import Protocol
import uuid

import numpy as np
from openpi_client import msgpack_numpy
import torch
from vlash.utils import prepare_observation_for_inference
import websockets.asyncio.server as _server
import websockets.frames

from hosting.auth import create_request_handler
from hosting.auth import pop_customer_id
from hosting.config import CustomerId
from hosting.vlash_config import VlashServiceConfig

logger = logging.getLogger(__name__)


class VlashPolicy(Protocol):
    """Protocol for VLASH policies that support action chunk prediction."""

    def predict_action_chunk(self, observation: dict) -> torch.Tensor: ...


class VlashPolicyServer:
    """WebSocket server for VLASH policies with async chunk pre-computation.

    Each connection maintains its own inference pipeline state: the last
    returned chunk (for future-state conditioning) and a pre-computed next
    chunk future.
    """

    def __init__(
        self,
        policy: VlashPolicy,
        device: torch.device,
        service_config: VlashServiceConfig,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._device = device
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
                "VLASH server listening on port %d (model_version=%s, max_concurrent=%d)",
                self._config.port,
                self._config.model_version,
                self._config.max_concurrent_requests,
            )
            await server.serve_forever()

    def _run_inference(
        self, observation: dict[str, np.ndarray], future_state: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        """Run VLASH inference synchronously (called via asyncio.to_thread).

        Args:
            observation: Observation dict with numpy arrays.
            future_state: If provided, replaces observation.state with the predicted
                future state (last action of the previous chunk) for future-state
                conditioning.

        Returns:
            Tuple of (action_chunk as numpy array [n_steps, action_dim], inference_time_ms).
        """
        observation = copy(observation)

        if future_state is not None:
            observation["observation.state"] = future_state

        start = time.monotonic()
        with torch.inference_mode():
            prepared = prepare_observation_for_inference(
                observation,
                self._device,
                self._config.task,
                self._config.robot_type,
            )
            action_chunk = self._policy.predict_action_chunk(prepared)

        action_chunk_numpy = action_chunk.squeeze(0).cpu().numpy()
        inference_time_ms = (time.monotonic() - start) * 1000
        return action_chunk_numpy, inference_time_ms

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        customer_id: CustomerId = pop_customer_id(websocket) or CustomerId("unknown")
        connection_id = str(uuid.uuid4())[:8]

        logger.info(
            "Connection opened",
            extra={"customer_id": customer_id, "connection_id": connection_id},
        )

        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        # Per-connection async state.
        precomputed_future: asyncio.Task | None = None

        while True:
            try:
                request_start = time.monotonic()
                request_id = str(uuid.uuid4())

                observation = msgpack_numpy.unpackb(await websocket.recv())

                # Concurrency check.
                if not self._semaphore._value:  # noqa: SLF001
                    logger.warning(
                        "Server busy, rejecting request",
                        extra={"customer_id": customer_id, "request_id": request_id},
                    )
                    await websocket.close(
                        code=websockets.frames.CloseCode.TRY_AGAIN_LATER,
                        reason="Server busy — max concurrent requests reached.",
                    )
                    break

                async with self._semaphore:
                    if precomputed_future is not None:
                        # We have a pre-computed chunk — await it (should be fast).
                        action_chunk, inference_time_ms = await precomputed_future
                        precomputed = True
                    else:
                        # First request: compute synchronously.
                        action_chunk, inference_time_ms = await asyncio.to_thread(
                            self._run_inference, observation, None
                        )
                        precomputed = False

                    # Start pre-computing the next chunk in the background.
                    # Use future-state conditioning: the last action of the current
                    # chunk predicts where the robot will be when this chunk finishes.
                    future_state = action_chunk[-1]
                    precomputed_future = asyncio.create_task(
                        asyncio.to_thread(self._run_inference, observation, future_state)
                    )

                wait_ms = (time.monotonic() - request_start) * 1000

                response = {
                    "actions": action_chunk,
                    "precomputed": precomputed,
                    "server_timing": {
                        "infer_ms": round(inference_time_ms, 1),
                        "wait_ms": round(wait_ms, 1),
                    },
                    "hosting": {
                        "customer_id": customer_id,
                        "request_id": request_id,
                        "model_version": self._config.model_version,
                    },
                }

                logger.info(
                    "Inference complete",
                    extra={
                        "customer_id": customer_id,
                        "request_id": request_id,
                        "model_version": self._config.model_version,
                        "infer_ms": round(inference_time_ms, 1),
                        "precomputed": precomputed,
                    },
                )

                await websocket.send(packer.pack(response))

            except websockets.ConnectionClosed:
                logger.info(
                    "Connection closed",
                    extra={"customer_id": customer_id, "connection_id": connection_id},
                )
                # Cancel any pending pre-computation.
                if precomputed_future is not None and not precomputed_future.done():
                    precomputed_future.cancel()
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
                # Cancel any pending pre-computation.
                if precomputed_future is not None and not precomputed_future.done():
                    precomputed_future.cancel()
                raise
