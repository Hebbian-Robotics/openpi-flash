"""VLASH-specific service configuration for the hosted inference server."""

import json
import os
import pathlib

from pydantic import BaseModel
from pydantic import field_validator

from hosting.config import CustomerConfig
from hosting.config import ModelVersion

# Re-export for convenience.
__all__ = ["VlashServiceConfig", "load_vlash_config"]


class VlashServiceConfig(BaseModel):
    """Top-level configuration for the VLASH hosted inference service."""

    # VLASH model settings.
    policy_type: str  # "pi0" or "pi05"
    pretrained_path: str  # HuggingFace hub name or local path to checkpoint
    model_version: ModelVersion
    task: str | None = None  # Language prompt / task description
    robot_type: str | None = None  # Robot type string for observation preprocessing
    compile_model: bool = False

    # Server settings.
    port: int = 8000
    max_concurrent_requests: int = 1

    # Customer settings (shared with OpenPI config).
    customers: list[CustomerConfig]

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, value: str) -> str:
        if value not in ("pi0", "pi05"):
            raise ValueError(f"policy_type must be 'pi0' or 'pi05', got '{value}'")
        return value

    def lookup_api_key(self, api_key: str) -> CustomerConfig | None:
        """Find the customer for a given API key, or None if not found."""
        for customer in self.customers:
            if customer.api_key == api_key:
                return customer
        return None


def load_vlash_config(config_path: str | None = None) -> VlashServiceConfig:
    """Load and parse VLASH service config from a JSON file.

    Uses INFERENCE_CONFIG_PATH env var if config_path is not provided.
    """
    config_path = config_path or os.environ.get("INFERENCE_CONFIG_PATH")
    if not config_path:
        raise ValueError("No config path provided. Set INFERENCE_CONFIG_PATH env var or pass config_path argument.")
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return VlashServiceConfig(**data)
