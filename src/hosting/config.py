"""Service configuration for the hosted inference server."""

import json
import os
import pathlib
from typing import Annotated, NewType, Protocol, runtime_checkable

from openpi.training import config as _openpi_config
from pydantic import BaseModel
from pydantic import field_validator

# Refined types — parsed at the boundary, trusted downstream.
ApiKey = NewType("ApiKey", str)
CustomerId = NewType("CustomerId", str)
ModelVersion = NewType("ModelVersion", str)


@runtime_checkable
class HasAuth(Protocol):
    """Minimal interface consumed by create_request_handler and HostedPolicyServer.

    Both ServiceConfig and VlashServiceConfig satisfy this protocol.
    """

    customers: list  # list[CustomerConfig]

    def lookup_api_key(self, api_key: str) -> "CustomerConfig | None": ...


class CustomerConfig(BaseModel):
    """A single customer's authentication and limits."""

    customer_id: CustomerId
    api_key: ApiKey
    rate_limit: int | None = None  # requests per minute, optional


class ServiceConfig(BaseModel):
    """Top-level configuration for the hosted inference service."""

    # Model settings
    model_config_name: Annotated[str, "Must match a registered openpi training config name"]
    checkpoint_dir: str  # local path or gs:// URI
    model_version: ModelVersion
    default_prompt: str | None = None

    # Server settings
    port: int = 8000
    max_concurrent_requests: int = 1

    # Customer settings
    customers: list[CustomerConfig]

    @field_validator("model_config_name")
    @classmethod
    def validate_model_config_name(cls, value: str) -> str:
        """Validate that the config name exists in openpi's registry at parse time."""
        try:
            _openpi_config.get_config(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        return value

    def lookup_api_key(self, api_key: str) -> CustomerConfig | None:
        """Find the customer for a given API key, or None if not found."""
        for customer in self.customers:
            if customer.api_key == api_key:
                return customer
        return None


def load_config(config_path: str | None = None) -> ServiceConfig:
    """Load and parse service config from a JSON file.

    Uses INFERENCE_CONFIG_PATH env var if config_path is not provided.
    Returns a validated ServiceConfig or raises on invalid input.
    """
    config_path = config_path or os.environ.get("INFERENCE_CONFIG_PATH")
    if not config_path:
        raise ValueError("No config path provided. Set INFERENCE_CONFIG_PATH env var or pass config_path argument.")
    path = pathlib.Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return ServiceConfig(**data)
