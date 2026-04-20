"""Roundtrip correctness tests for YOR embodiment mappings.

YOR has 6-DOF per arm (same as Galaxea R1 Lite), so the ALOHA mapping
is a native fit with no joint dropping required.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shared.mappings import AlohaMapping, DroidDualArmMapping
from shared.types import MappingName
from yor_client import MAPPINGS

# ---------------------------------------------------------------------------
# Fixtures — reusable sensor data with distinct values per joint (6-DOF)
# ---------------------------------------------------------------------------

LEFT_ARM = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
RIGHT_ARM = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
LEFT_GRIPPER = 0.8
RIGHT_GRIPPER = 0.9


def _make_dummy_images(
    camera_names: list[str],
    height: int = 224,
    width: int = 224,
) -> dict[str, np.ndarray]:
    """Create HWC uint8 images with a known red pixel at (0, 0)."""
    images: dict[str, np.ndarray] = {}
    for name in camera_names:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[0, 0] = [255, 0, 0]
        images[name] = image
    return images


# ---------------------------------------------------------------------------
# AlohaMapping — Native 6-DOF Fit (no joint dropping)
# ---------------------------------------------------------------------------


class TestAlohaMapping:
    mapping = MAPPINGS[MappingName.ALOHA]

    def _camera_names(self) -> list[str]:
        return list(self.mapping.camera_names)

    def test_joints_pass_through_without_dropping(self) -> None:
        """YOR's 6-DOF is a native ALOHA fit — no joints should be dropped."""
        obs = self.mapping.build_observation(
            left_arm_positions=LEFT_ARM,
            right_arm_positions=RIGHT_ARM,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(self._camera_names()),
            prompt="test",
        )

        state = obs["state"]
        assert state.shape == (14,), f"ALOHA state should be 14-dim, got {state.shape}"

        expected_state = np.concatenate([LEFT_ARM, [LEFT_GRIPPER], RIGHT_ARM, [RIGHT_GRIPPER]])
        assert_array_almost_equal(state, expected_state)

    def test_roundtrip_preserves_all_joints(self) -> None:
        """No joints are lost in the YOR roundtrip (6-DOF is native ALOHA)."""
        obs = self.mapping.build_observation(
            left_arm_positions=LEFT_ARM,
            right_arm_positions=RIGHT_ARM,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(self._camera_names()),
            prompt="test",
        )

        command = self.mapping.unpack_actions({"actions": obs["state"]})

        assert_array_almost_equal(command.left_arm_joint_positions, LEFT_ARM)
        assert_array_almost_equal(command.right_arm_joint_positions, RIGHT_ARM)
        assert command.left_gripper_position == pytest.approx(LEFT_GRIPPER)
        assert command.right_gripper_position == pytest.approx(RIGHT_GRIPPER)

    def test_action_indexing_14dim(self) -> None:
        actions_14 = np.arange(14, dtype=np.float64)
        command = self.mapping.unpack_actions({"actions": actions_14})

        assert_array_equal(command.left_arm_joint_positions, [0, 1, 2, 3, 4, 5])
        assert command.left_gripper_position == 6.0

        assert_array_equal(command.right_arm_joint_positions, [7, 8, 9, 10, 11, 12])
        assert command.right_gripper_position == 13.0

    def test_unpack_handles_batched_actions(self) -> None:
        actions_14 = np.arange(14, dtype=np.float64)

        command_unbatched = self.mapping.unpack_actions({"actions": actions_14})

        batched = np.tile(actions_14, (5, 1))
        batched[1:] = 999.0
        command_batched = self.mapping.unpack_actions({"actions": batched})

        assert_array_equal(
            command_unbatched.left_arm_joint_positions,
            command_batched.left_arm_joint_positions,
        )

    def test_images_forwarded_as_native_hwc(self) -> None:
        """ALOHA images flow through as native HWC uint8; the Rust client
        sidecar transposes to CHW and resizes to 3x224x224 before the wire."""
        obs = self.mapping.build_observation(
            left_arm_positions=LEFT_ARM,
            right_arm_positions=RIGHT_ARM,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(self._camera_names()),
            prompt="test",
        )

        for camera_name, image in obs["images"].items():
            if camera_name == "cam_low":
                continue  # dummy black image
            assert image.ndim == 3, f"{camera_name} should be 3-D but got {image.shape}"
            assert image.shape[-1] == 3, f"{camera_name} should be HWC but got {image.shape}"
            assert_array_equal(image[0, 0], [255, 0, 0])

    def test_dummy_cam_low_injected(self) -> None:
        """YOR's default ALOHA mapping uses use_dummy_cam_low=True."""
        obs = self.mapping.build_observation(
            left_arm_positions=LEFT_ARM,
            right_arm_positions=RIGHT_ARM,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(self._camera_names()),
            prompt="test",
        )

        assert "cam_low" in obs["images"]
        cam_low = obs["images"]["cam_low"]
        assert cam_low.ndim == 3
        assert cam_low.shape[-1] == 3, f"dummy cam_low should be HWC but got {cam_low.shape}"
        assert np.all(cam_low == 0)


# ---------------------------------------------------------------------------
# DroidDualArmMapping — Joint Routing Roundtrip (6-DOF)
# ---------------------------------------------------------------------------


class TestDroidDualArmMapping:
    mapping = MAPPINGS[MappingName.DROID_DUAL_ARM]

    def _build_observation(self) -> dict:
        camera_names = list(self.mapping.camera_names)
        return self.mapping.build_observation(
            left_arm_positions=LEFT_ARM,
            right_arm_positions=RIGHT_ARM,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(camera_names),
            prompt="test prompt",
        )

    def test_joint_values_route_to_correct_actuators(self) -> None:
        obs = self._build_observation()

        assert_array_almost_equal(obs["observation/joint_position"], LEFT_ARM)
        assert_array_almost_equal(obs["observation/joint_position_right"], RIGHT_ARM)
        assert_array_almost_equal(obs["observation/gripper_position"], [LEFT_GRIPPER])
        assert_array_almost_equal(obs["observation/gripper_position_right"], [RIGHT_GRIPPER])

        actions_14 = np.concatenate([LEFT_ARM, [LEFT_GRIPPER], RIGHT_ARM, [RIGHT_GRIPPER]])
        command = self.mapping.unpack_actions({"actions": actions_14})

        assert_array_almost_equal(command.left_arm_joint_positions, LEFT_ARM)
        assert_array_almost_equal(command.right_arm_joint_positions, RIGHT_ARM)
        assert command.left_gripper_position == pytest.approx(LEFT_GRIPPER)
        assert command.right_gripper_position == pytest.approx(RIGHT_GRIPPER)

    def test_action_slicing_with_index_values(self) -> None:
        actions_14 = np.arange(14, dtype=np.float64)
        command = self.mapping.unpack_actions({"actions": actions_14})

        assert_array_equal(command.left_arm_joint_positions, [0, 1, 2, 3, 4, 5])
        assert command.left_gripper_position == 6.0
        assert_array_equal(command.right_arm_joint_positions, [7, 8, 9, 10, 11, 12])
        assert command.right_gripper_position == 13.0

    def test_images_forwarded_as_native_hwc(self) -> None:
        """Images are forwarded as HWC uint8 at whatever resolution the robot
        provides — the client transport handles resize to 224x224."""
        obs = self._build_observation()

        for key in ("observation/exterior_image_1_left",):
            image = obs[key]
            assert image.ndim == 3, f"{key} should be 3-D but got {image.shape}"
            assert image.shape[-1] == 3, f"{key} should be HWC but got {image.shape}"
            assert_array_equal(image[0, 0], [255, 0, 0])

    def test_prompt_passes_through(self) -> None:
        obs = self._build_observation()
        assert obs["prompt"] == "test prompt"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestMappingRegistry:
    def test_all_enum_variants_registered(self) -> None:
        for name in MappingName:
            assert name in MAPPINGS, f"MappingName.{name} not in YOR mappings"

    def test_registry_types(self) -> None:
        assert isinstance(MAPPINGS[MappingName.DROID_DUAL_ARM], DroidDualArmMapping)
        assert isinstance(MAPPINGS[MappingName.ALOHA], AlohaMapping)
