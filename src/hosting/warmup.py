"""Embodiment-aware warmup observation factories for model compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np

# Dtypes the transport accepts for image preprocessing. Kept as a Literal so
# serializing `ImageSpec` to msgpack cannot smuggle in an invalid value
# (e.g. "float64") that the Rust side would reject at runtime.
ImageDtypeName = Literal["uint8", "float32"]


class ImageSpec(TypedDict):
    """Per-field image preprocessing rule advertised to the transport.

    Serialized to msgpack as part of server metadata and re-parsed in Rust
    (`flash-transport/src/metadata.rs`). The TypedDict shape is the contract
    both sides agree on; the Python type-checker flags misspelled keys and
    wrong value types at authoring time rather than at handshake time.
    """

    path: list[str]
    target_shape: list[int]
    dtype: ImageDtypeName


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


# Image specs the server advertises in metadata so the client transport
# can resize raw camera frames before they hit the QUIC wire. See
# `flash-transport/src/image_preprocess.rs` for the consumer side.
_ALOHA_IMAGE_SPECS: list[ImageSpec] = [
    ImageSpec(path=["images", camera_name], target_shape=[3, 224, 224], dtype="uint8")
    for camera_name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
]
_DROID_IMAGE_SPECS: list[ImageSpec] = [
    ImageSpec(
        path=["observation/exterior_image_1_left"],
        target_shape=[224, 224, 3],
        dtype="uint8",
    ),
    ImageSpec(
        path=["observation/wrist_image_left"],
        target_shape=[224, 224, 3],
        dtype="uint8",
    ),
]


def make_image_specs(train_config: Any) -> list[ImageSpec]:
    """Return the per-field image preprocessing specs for this embodiment,
    or an empty list for embodiments we don't recognize (client just no-ops
    on preprocessing in that case).
    """
    warmup_observation_spec = get_warmup_observation_spec(train_config)
    match warmup_observation_spec:
        case AlohaWarmupObservationSpec():
            return _ALOHA_IMAGE_SPECS
        case DroidWarmupObservationSpec():
            return _DROID_IMAGE_SPECS


def get_action_horizon(train_config: Any) -> int | None:
    """Return the model's action chunk length (the transport advertises it
    to clients for action chunking).

    Reads ``train_config.model.action_horizon`` if present. Returns ``None``
    for configs that don't expose it; the client treats that as "chunking
    disabled" and just passes server responses through whole.
    """
    model = getattr(train_config, "model", None)
    if model is None:
        return None
    horizon = getattr(model, "action_horizon", None)
    if isinstance(horizon, int) and horizon > 0:
        return horizon
    return None
