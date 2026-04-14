"""Embodiment-aware warmup observation factories for model compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AlohaWarmupObservationSpec:
    """Warmup input spec for ALOHA-style embodiments."""

    prompt: str = "warmup"


@dataclass(frozen=True)
class DroidWarmupObservationSpec:
    """Warmup input spec for DROID-style embodiments."""

    prompt: str = "warmup"


WarmupObservationSpec = AlohaWarmupObservationSpec | DroidWarmupObservationSpec


def make_aloha_observation(prompt: str) -> dict[str, Any]:
    """Create a dummy ALOHA observation matching the expected model input shape."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def make_droid_observation(prompt: str) -> dict[str, Any]:
    """Create a dummy DROID observation matching the expected model input shape."""
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256,
            size=(224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/wrist_image_left": np.random.randint(
            256,
            size=(224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": prompt,
    }


def get_warmup_observation_spec(train_config: Any) -> WarmupObservationSpec:
    """Derive the correct warmup input spec from the parsed OpenPI train config."""
    data_config = getattr(train_config, "data", None)
    data_config_type_name = type(data_config).__name__
    asset_id = getattr(getattr(data_config, "assets", None), "asset_id", None)
    config_name = getattr(train_config, "name", "<unknown>")

    if data_config_type_name == "LeRobotAlohaDataConfig":
        return AlohaWarmupObservationSpec()
    if data_config_type_name == "SimpleDataConfig" and asset_id == "droid":
        return DroidWarmupObservationSpec()

    if asset_id == "trossen":
        return AlohaWarmupObservationSpec()
    if asset_id == "droid":
        return DroidWarmupObservationSpec()

    raise ValueError(
        "No warmup observation generator is registered for "
        f"config={config_name!r} data_config_type={data_config_type_name!r} asset_id={asset_id!r}."
    )


def make_warmup_observation(train_config: Any) -> dict[str, Any]:
    """Create a valid warmup observation for the given parsed OpenPI config."""
    warmup_observation_spec = get_warmup_observation_spec(train_config)

    match warmup_observation_spec:
        case AlohaWarmupObservationSpec(prompt=prompt):
            return make_aloha_observation(prompt=prompt)
        case DroidWarmupObservationSpec(prompt=prompt):
            return make_droid_observation(prompt=prompt)
