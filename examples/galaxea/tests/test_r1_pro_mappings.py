"""Roundtrip correctness tests for R1 Pro embodiment mappings.

Every test feeds known, distinct values through the mapping pipeline and
verifies they arrive at the correct destination.  A failure here means the
robot would move the wrong joint or the model would see garbled images.
"""

from __future__ import annotations

import numpy as np
import pytest
from _common import MappingName, RobotModel
from embodiment_mappings import CAMERA_ROS_TOPICS, MAPPINGS_BY_ROBOT
from numpy.testing import assert_array_almost_equal, assert_array_equal
from shared.mappings import AlohaMapping, DroidDualArmMapping

# ---------------------------------------------------------------------------
# Fixtures — reusable sensor data with distinct values per joint (7-DOF)
# ---------------------------------------------------------------------------

R1_PRO_MAPPINGS = MAPPINGS_BY_ROBOT[RobotModel.R1_PRO]

LEFT_ARM = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
RIGHT_ARM = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
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
# DroidDualArmMapping — Joint Routing Roundtrip (7-DOF)
# ---------------------------------------------------------------------------


class TestDroidDualArmMapping:
    mapping = R1_PRO_MAPPINGS[MappingName.DROID_DUAL_ARM]

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

        actions_16 = np.concatenate([LEFT_ARM, [LEFT_GRIPPER], RIGHT_ARM, [RIGHT_GRIPPER]])
        command = self.mapping.unpack_actions({"actions": actions_16})

        assert_array_almost_equal(command.left_arm_joint_positions, LEFT_ARM)
        assert_array_almost_equal(command.right_arm_joint_positions, RIGHT_ARM)
        assert command.left_gripper_position == pytest.approx(LEFT_GRIPPER)
        assert command.right_gripper_position == pytest.approx(RIGHT_GRIPPER)

    def test_action_slicing_with_index_values(self) -> None:
        """Each dim set to its own index — makes off-by-one errors immediately visible."""
        actions_16 = np.arange(16, dtype=np.float64)
        command = self.mapping.unpack_actions({"actions": actions_16})

        assert_array_equal(command.left_arm_joint_positions, [0, 1, 2, 3, 4, 5, 6])
        assert command.left_gripper_position == 7.0
        assert_array_equal(command.right_arm_joint_positions, [8, 9, 10, 11, 12, 13, 14])
        assert command.right_gripper_position == 15.0

    def test_unpack_handles_batched_and_unbatched_actions(self) -> None:
        actions_16 = np.arange(16, dtype=np.float64)

        command_unbatched = self.mapping.unpack_actions({"actions": actions_16})

        batched = np.tile(actions_16, (5, 1))
        batched[1:] = 999.0
        command_batched = self.mapping.unpack_actions({"actions": batched})

        assert_array_equal(
            command_unbatched.left_arm_joint_positions,
            command_batched.left_arm_joint_positions,
        )
        assert_array_equal(
            command_unbatched.right_arm_joint_positions,
            command_batched.right_arm_joint_positions,
        )
        assert command_unbatched.left_gripper_position == command_batched.left_gripper_position
        assert command_unbatched.right_gripper_position == command_batched.right_gripper_position

    def test_images_forwarded_as_native_hwc(self) -> None:
        """Images are forwarded as HWC uint8 at whatever resolution the robot
        provides — the client transport handles resize to 224x224."""
        obs = self._build_observation()

        for key in (
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
            "observation/wrist_image_right",
        ):
            image = obs[key]
            assert image.ndim == 3, f"{key} should be 3-D but got {image.shape}"
            assert image.shape[-1] == 3, f"{key} should be HWC but got {image.shape}"
            assert_array_equal(image[0, 0], [255, 0, 0])

    def test_prompt_passes_through(self) -> None:
        obs = self._build_observation()
        assert obs["prompt"] == "test prompt"


# ---------------------------------------------------------------------------
# AlohaMapping — Drop/Restore Roundtrip (7-DOF → 6-DOF → 7-DOF)
# ---------------------------------------------------------------------------


class TestAlohaMapping:
    mapping = R1_PRO_MAPPINGS[MappingName.ALOHA]

    def _camera_names(self) -> list[str]:
        return list(self.mapping.camera_names)

    def test_drops_correct_joint_and_restores_it(self) -> None:
        left_arm = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        right_arm = np.array([110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0])

        obs = self.mapping.build_observation(
            left_arm_positions=left_arm,
            right_arm_positions=right_arm,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(self._camera_names()),
            prompt="test",
        )

        state = obs["state"]
        assert state.shape == (14,), f"ALOHA state should be 14-dim, got {state.shape}"

        assert 10.0 not in state, "Left joint 0 should be dropped"
        assert 110.0 not in state, "Right joint 0 should be dropped"

        expected_state = np.array(
            [20, 30, 40, 50, 60, 70, LEFT_GRIPPER, 120, 130, 140, 150, 160, 170, RIGHT_GRIPPER]
        )
        assert_array_almost_equal(state, expected_state)

        command = self.mapping.unpack_actions({"actions": state})

        assert command.left_arm_joint_positions[0] == 0.0
        assert command.right_arm_joint_positions[0] == 0.0

        assert_array_almost_equal(command.left_arm_joint_positions[1:], [20, 30, 40, 50, 60, 70])
        assert_array_almost_equal(
            command.right_arm_joint_positions[1:], [120, 130, 140, 150, 160, 170]
        )
        assert command.left_gripper_position == pytest.approx(LEFT_GRIPPER)
        assert command.right_gripper_position == pytest.approx(RIGHT_GRIPPER)

    def test_custom_drop_index_routes_correctly(self) -> None:
        """Construct an AlohaMapping with a non-default dropped joint index."""
        mapping = AlohaMapping(
            camera_names_config=self.mapping.camera_names,
            dropped_joint_index=3,
        )
        left_arm = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        right_arm = np.array([110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0])

        obs = mapping.build_observation(
            left_arm_positions=left_arm,
            right_arm_positions=right_arm,
            left_gripper_position=LEFT_GRIPPER,
            right_gripper_position=RIGHT_GRIPPER,
            images=_make_dummy_images(list(mapping.camera_names)),
            prompt="test",
        )

        state = obs["state"]
        assert 40.0 not in state, "Left joint 3 should be dropped"
        assert 140.0 not in state, "Right joint 3 should be dropped"

        command = mapping.unpack_actions({"actions": state})
        assert command.left_arm_joint_positions[3] == 0.0
        assert command.right_arm_joint_positions[3] == 0.0

    def test_action_indexing_14dim(self) -> None:
        actions_14 = np.arange(14, dtype=np.float64)
        command = self.mapping.unpack_actions({"actions": actions_14})

        assert command.left_arm_joint_positions[0] == 0.0  # restored
        assert_array_equal(command.left_arm_joint_positions[1:], [0, 1, 2, 3, 4, 5])
        assert command.left_gripper_position == 6.0

        assert command.right_arm_joint_positions[0] == 0.0  # restored
        assert_array_equal(command.right_arm_joint_positions[1:], [7, 8, 9, 10, 11, 12])
        assert command.right_gripper_position == 13.0

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
            assert image.ndim == 3, f"{camera_name} should be 3-D but got {image.shape}"
            assert image.shape[-1] == 3, f"{camera_name} should be HWC but got {image.shape}"
            assert_array_equal(image[0, 0], [255, 0, 0])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestMappingRegistry:
    def test_all_enum_variants_registered(self) -> None:
        for name in MappingName:
            assert name in R1_PRO_MAPPINGS, f"MappingName.{name} not in R1 Pro mappings"

    def test_registry_types(self) -> None:
        assert isinstance(R1_PRO_MAPPINGS[MappingName.DROID_DUAL_ARM], DroidDualArmMapping)
        assert isinstance(R1_PRO_MAPPINGS[MappingName.ALOHA], AlohaMapping)

    def test_camera_ros_topics_registered(self) -> None:
        """Every robot+mapping combo should have corresponding camera ROS topics."""
        for robot in RobotModel:
            for mapping_name in MappingName:
                key = (robot, mapping_name)
                assert key in CAMERA_ROS_TOPICS, f"{key} not in CAMERA_ROS_TOPICS"
                mapping = MAPPINGS_BY_ROBOT[robot][mapping_name]
                ros_camera_names = set(CAMERA_ROS_TOPICS[key].keys())
                mapping_camera_names = set(mapping.camera_names)
                assert ros_camera_names == mapping_camera_names, (
                    f"Camera name mismatch for {key}: "
                    f"ROS topics have {ros_camera_names}, mapping expects {mapping_camera_names}"
                )
