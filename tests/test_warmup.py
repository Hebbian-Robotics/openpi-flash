from types import SimpleNamespace

import pytest

from hosting.warmup import (
    AlohaWarmupObservationSpec,
    DroidWarmupObservationSpec,
    get_warmup_observation_spec,
    make_warmup_observation,
)


def _make_train_config(asset_id: str, config_name: str = "test-config") -> SimpleNamespace:
    return SimpleNamespace(
        name=config_name,
        data=SimpleNamespace(assets=SimpleNamespace(asset_id=asset_id)),
    )


def test_make_warmup_observation_uses_aloha_shape_for_trossen_configs() -> None:
    train_config = _make_train_config("trossen", config_name="pi05_aloha")

    warmup_observation_spec = get_warmup_observation_spec(train_config)
    warmup_observation = make_warmup_observation(train_config)

    assert warmup_observation_spec == AlohaWarmupObservationSpec()
    assert warmup_observation["state"].shape == (14,)
    assert set(warmup_observation["images"]) == {
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    }


def test_make_warmup_observation_uses_aloha_shape_for_aloha_data_configs_without_asset_id() -> None:
    train_config = SimpleNamespace(
        name="pi0_aloha_sim",
        data=type("LeRobotAlohaDataConfig", (), {"assets": SimpleNamespace(asset_id=None)})(),
    )

    warmup_observation_spec = get_warmup_observation_spec(train_config)
    warmup_observation = make_warmup_observation(train_config)

    assert warmup_observation_spec == AlohaWarmupObservationSpec()
    assert warmup_observation["state"].shape == (14,)
    assert "images" in warmup_observation


def test_make_warmup_observation_uses_droid_shape_for_droid_configs() -> None:
    train_config = _make_train_config("droid", config_name="pi05_droid")

    warmup_observation_spec = get_warmup_observation_spec(train_config)
    warmup_observation = make_warmup_observation(train_config)

    assert warmup_observation_spec == DroidWarmupObservationSpec()
    assert warmup_observation["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert warmup_observation["observation/wrist_image_left"].shape == (224, 224, 3)
    assert warmup_observation["observation/joint_position"].shape == (7,)
    assert warmup_observation["observation/gripper_position"].shape == (1,)


def test_make_warmup_observation_rejects_unknown_embodiments() -> None:
    train_config = _make_train_config("custom-hand", config_name="custom-hand-policy")

    with pytest.raises(ValueError, match="No warmup observation generator is registered"):
        make_warmup_observation(train_config)
