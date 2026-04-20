"""Service configuration for the openpi-flash inference engine.

Two optional component slots — ``action`` (PyTorch or JAX action policy) and
``planner`` (JAX subtask generator). Mode is derived from which slots are set:

- action only  → current production default: serves actions, no planner loaded.
- planner only → serves subtask text only; no PyTorch action model loaded.
- both         → "combined" two-phase mode; clients can hit either endpoint.

Each active slot gets its own transport triple (websocket, QUIC, unix socket),
so the two endpoints are fully independent on the wire.

Configuration sources, in order of precedence (highest wins):

1. Environment variables: ``OPENPI_<FIELD>[__<NESTED_FIELD>...]``.
   Example: ``OPENPI_ACTION__CHECKPOINT_DIR=/tmp/x`` overrides the action
   slot's checkpoint without touching the JSON file. Useful for one-off
   checkpoint swaps on a deployed box.
2. JSON file at ``INFERENCE_CONFIG_PATH`` (or the ``--config`` CLI flag).
3. Pydantic defaults (transports, prompts, ``max_generation_tokens``).
"""

import json
import os
import pathlib
from typing import Self, TypeVar

from openpi.training import config as _openpi_config
from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

_T = TypeVar("_T", bound=BaseModel)


# Shared defaults and validators
# ------------------------------------------------------------------
# Defined at module scope (not inside a model class) so the admin HTTP
# endpoint can re-use the same invariants without duplicating them.

DEFAULT_GENERATION_PROMPT_FORMAT = "Task: {task}. Subtask: "
DEFAULT_ACTION_PROMPT_TEMPLATE = "{task}. Subtask: {subtask}"


def require_task_placeholder(value: str) -> str:
    """Enforce that a generation prompt contains the ``{task}`` placeholder."""
    if "{task}" not in value:
        raise ValueError("must contain '{task}' placeholder")
    return value


def require_task_and_subtask_placeholders(value: str) -> str:
    """Enforce that an action-prompt template contains both placeholders."""
    if "{task}" not in value or "{subtask}" not in value:
        raise ValueError("must contain both '{task}' and '{subtask}' placeholders")
    return value


def load_json_config(config_cls: type[_T], config_path: str | None = None) -> _T:
    """Load and parse a Pydantic config from a JSON file.

    Uses INFERENCE_CONFIG_PATH env var if config_path is not provided.
    Returns a validated config instance or raises on invalid input.
    """
    config_path = config_path or os.environ.get("INFERENCE_CONFIG_PATH")
    if not config_path:
        raise ValueError(
            "No config path provided. Set INFERENCE_CONFIG_PATH env var or pass config_path argument."
        )
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return config_cls(**data)


class ActionConfig(BaseModel):
    """Configuration for the action generation slot.

    The backend (PyTorch vs JAX) is auto-detected by
    ``openpi.policies.policy_config.create_trained_policy`` based on whether
    ``model.safetensors`` exists in ``checkpoint_dir``. Point at a PyTorch
    checkpoint to serve PyTorch actions, or a JAX Orbax checkpoint to serve
    JAX actions — no server code change needed.
    """

    # Kept as ``str`` because valid names are defined by the openpi registry,
    # which is an external runtime dependency — not a finite set we can encode
    # as a Literal.
    model_config_name: str
    checkpoint_dir: str  # local path or gs:// URI
    default_prompt: str | None = None

    @field_validator("model_config_name")
    @classmethod
    def validate_model_config_name(cls, value: str) -> str:
        """Validate that the config name exists in openpi's registry at parse time."""
        try:
            _openpi_config.get_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        return value


class PlannerConfig(BaseModel):
    """Configuration for the JAX subtask planner slot (pi0.5 two-phase inference).

    The planner is JAX-only for now

    Two prompt fields live here because both are planner-specific:

    - ``generation_prompt_format``: prompt used during the subtask decode.
      Typically ``"Task: {task}. Subtask: "``. Must contain ``{task}``.
      Mutable at runtime via the admin HTTP endpoint.
    - ``action_prompt_template``: template used to splice generated subtask
      text back into the action prompt when combined mode is active. Must
      contain both ``{task}`` and ``{subtask}``. Only consulted when the
      action slot is also loaded.
    """

    checkpoint_dir: str  # Orbax checkpoint path (local or gs://)
    max_generation_tokens: int = 20
    generation_prompt_format: str = DEFAULT_GENERATION_PROMPT_FORMAT
    action_prompt_template: str = DEFAULT_ACTION_PROMPT_TEMPLATE

    @field_validator("generation_prompt_format")
    @classmethod
    def _check_generation_prompt_format(cls, value: str) -> str:
        return require_task_placeholder(value)

    @field_validator("action_prompt_template")
    @classmethod
    def _check_action_prompt_template(cls, value: str) -> str:
        return require_task_and_subtask_placeholders(value)


class SlotTransportConfig(BaseModel):
    """Transport listeners for one slot: TCP websocket + QUIC UDP + unix socket."""

    websocket_port: int
    quic_port: int
    unix_socket_path: str


_DEFAULT_ACTION_TRANSPORT = SlotTransportConfig(
    websocket_port=8000,
    quic_port=5555,
    unix_socket_path="/tmp/openpi-action.sock",
)

_DEFAULT_PLANNER_TRANSPORT = SlotTransportConfig(
    websocket_port=8002,
    quic_port=5556,
    unix_socket_path="/tmp/openpi-planner.sock",
)


class ServiceConfig(BaseSettings):
    """Top-level configuration for the openpi-flash inference engine.

    At least one of ``action`` or ``planner`` must be set. When both are set,
    the server runs in "combined" mode and clients can hit either endpoint;
    the shared ``SubtaskGenerator`` is loaded once and used by both.

    Environment variables prefixed with ``OPENPI_`` override JSON values
    (nested fields use ``__`` as the delimiter — e.g.
    ``OPENPI_ACTION__CHECKPOINT_DIR``).
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENPI_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    action: ActionConfig | None = None
    planner: PlannerConfig | None = None

    action_transport: SlotTransportConfig = _DEFAULT_ACTION_TRANSPORT
    planner_transport: SlotTransportConfig = _DEFAULT_PLANNER_TRANSPORT

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: env > init (JSON) > dotenv > secrets. The default is
        # init > env, which is wrong for us — we want env vars to override
        # the JSON on deployed boxes, not the other way around.
        del settings_cls  # unused; retained to match BaseSettings signature
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)

    @model_validator(mode="after")
    def validate_at_least_one_slot(self) -> Self:
        if self.action is None and self.planner is None:
            raise ValueError(
                "ServiceConfig requires at least one of 'action' or 'planner' to be configured"
            )
        return self


def load_config(config_path: str | None = None) -> ServiceConfig:
    """Load and parse service config from a JSON file.

    Env vars (``OPENPI_*``) are merged on top — see the module docstring.
    """
    return load_json_config(ServiceConfig, config_path)
