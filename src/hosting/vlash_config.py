"""VLASH-specific service configuration for the hosted inference server."""

import json
import os
import pathlib
from typing import Literal

from pydantic import BaseModel

PolicyType = Literal["pi0", "pi05"]


class VlashServiceConfig(BaseModel):
    """Top-level configuration for the VLASH hosted inference service."""

    # VLASH model settings.
    policy_type: PolicyType
    pretrained_path: str  # HuggingFace hub name or local path to checkpoint
    model_version: str
    task: str | None = None  # Language prompt / task description
    robot_type: str | None = None  # Robot type string for observation preprocessing
    compile_model: bool = False

    # Server settings.
    port: int = 8000
    max_concurrent_requests: int = 1


def load_vlash_config(config_path: str | None = None) -> VlashServiceConfig:
    """Load and parse VLASH service config from a JSON file.

    Uses INFERENCE_CONFIG_PATH env var if config_path is not provided.
    """
    config_path = config_path or os.environ.get("INFERENCE_CONFIG_PATH")
    if not config_path:
        raise ValueError(
            "No config path provided. Set INFERENCE_CONFIG_PATH env var or pass config_path argument."
        )
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    return VlashServiceConfig(**data)
