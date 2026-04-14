"""Service configuration for the openpi-flash inference engine."""

import json
import os
import pathlib
from typing import TypeVar

from openpi.training import config as _openpi_config
from pydantic import BaseModel, field_validator

_T = TypeVar("_T", bound=BaseModel)


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


class ServiceConfig(BaseModel):
    """Top-level configuration for the openpi-flash inference engine."""

    # Model settings
    # Exception: kept as str because valid names are defined by the openpi registry,
    # which is an external runtime dependency — not a finite set we can encode as a Literal.
    model_config_name: str
    checkpoint_dir: str  # local path or gs:// URI
    default_prompt: str | None = None

    # Server settings
    port: int = 8000
    quic_port: int = 5555

    @field_validator("model_config_name")
    @classmethod
    def validate_model_config_name(cls, value: str) -> str:
        """Validate that the config name exists in openpi's registry at parse time."""
        try:
            _openpi_config.get_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        return value


def load_config(config_path: str | None = None) -> ServiceConfig:
    """Load and parse service config from a JSON file."""
    return load_json_config(ServiceConfig, config_path)
