"""VLASH-specific service configuration for the hosted inference server."""

from typing import Literal

from pydantic import BaseModel

from hosting.config import load_json_config

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
    """Load and parse VLASH service config from a JSON file."""
    return load_json_config(VlashServiceConfig, config_path)
