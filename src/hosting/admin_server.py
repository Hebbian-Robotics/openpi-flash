"""Lightweight HTTP admin endpoint for runtime configuration.

Runs alongside the inference servers (WebSocket, QUIC) on a separate port.
Only started when the planner slot is loaded — the admin endpoint's only
current purpose is mutating the planner's decode prompt at runtime.

Routes:
    GET   /health          — health check
    GET   /config          — returns current runtime config as JSON
    PATCH /config          — partial-update runtime config (JSON body)
    GET   /docs            — auto-generated Swagger UI (FastAPI)

Example:
    curl http://localhost:8001/config
    curl -X PATCH http://localhost:8001/config \\
         -d '{"generation_prompt_format": "Task: {task}. Subtask: 1"}'
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, fields
from typing import Annotated

import uvicorn
from fastapi import FastAPI
from pydantic import AfterValidator, BaseModel, ConfigDict

from hosting.config import DEFAULT_GENERATION_PROMPT_FORMAT, require_task_placeholder

logger = logging.getLogger(__name__)

DEFAULT_ADMIN_PORT = 8001


class RuntimeConfigUpdate(BaseModel):
    """Partial-update schema for ``PATCH /config`` request bodies.

    All fields are ``Optional`` — only provided keys are applied to
    ``RuntimeConfig``. Unknown fields are rejected so typos fail loudly
    rather than silently no-op. The ``{task}`` placeholder rule is shared
    with ``PlannerConfig`` via ``hosting.config.require_task_placeholder``,
    so the invariant has a single source of truth.

    FastAPI auto-parses the request body against this model. Validation
    failures come back to the client as HTTP 422 with the standard
    ``{"detail": [...]}`` shape — no hand-rolled error plumbing needed.
    """

    model_config = ConfigDict(extra="forbid")

    generation_prompt_format: Annotated[str, AfterValidator(require_task_placeholder)] | None = None


@dataclass
class RuntimeConfig:
    """Mutable runtime state for the planner.

    Thread-safe: reads are atomic (GIL), writes go through ``apply()`` which
    holds a lock. Only instantiated when the planner slot is loaded;
    ``SubtaskGenerator.generate()`` re-reads each field on every call to
    pick up admin edits without needing a restart.
    """

    generation_prompt_format: str = DEFAULT_GENERATION_PROMPT_FORMAT

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def apply(self, update: RuntimeConfigUpdate) -> None:
        """Atomically apply a validated partial update.

        Only keys the client explicitly sent (``exclude_unset=True``) are
        applied, so a ``PATCH`` with ``{"generation_prompt_format": "..."}``
        doesn't clobber fields that weren't mentioned.
        """
        changes = update.model_dump(exclude_unset=True)
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)
                logger.info("Runtime config updated: %s = %r", key, value)

    def to_dict(self) -> dict[str, object]:
        # Build manually — asdict() recurses into _lock (threading.Lock) and
        # fails before we can pop it. Only expose non-underscore fields.
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}


def build_admin_app(runtime_config: RuntimeConfig) -> FastAPI:
    """Build the FastAPI app for the admin endpoint.

    Exposed separately from ``start_admin_server`` so tests can mount the
    same routing stack against ``fastapi.testclient.TestClient`` without
    starting a real socket.
    """
    app = FastAPI(
        title="openpi-flash admin",
        description="Runtime configuration for the planner slot.",
        version="1",
    )

    @app.get("/health")
    def get_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> dict[str, object]:
        return runtime_config.to_dict()

    @app.patch("/config")
    def patch_config(update: RuntimeConfigUpdate) -> dict[str, object]:
        # FastAPI already parsed the body into a validated ``update``.
        # On bad input it short-circuits with a 422 before this runs.
        runtime_config.apply(update)
        return runtime_config.to_dict()

    return app


def start_admin_server(
    runtime_config: RuntimeConfig,
    port: int = DEFAULT_ADMIN_PORT,
) -> None:
    """Start the admin HTTP server in a daemon thread.

    uvicorn owns its own asyncio event loop inside the thread so we don't
    have to coordinate with the main thread's loop (which is synchronous).
    """
    app = build_admin_app(runtime_config)
    config = uvicorn.Config(
        app,
        # Bind to 0.0.0.0 inside the container; docker/terraform publishes
        # the admin port on 127.0.0.1 only, so it's never internet-reachable.
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="admin-server", daemon=True)
    thread.start()
    logger.info("Admin HTTP server started on port %d", port)
