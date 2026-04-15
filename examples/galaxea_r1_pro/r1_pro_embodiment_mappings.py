"""Pluggable embodiment mappings for the Galaxea R1 Pro -> OpenPI observation format.

Each mapping converts raw R1 Pro sensor data (joint positions, camera images) into
the observation dict expected by a specific OpenPI model config and converts the
returned action array back into per-actuator commands.

To add a new mapping, subclass ``EmbodimentMapping`` and register it in
``AVAILABLE_MAPPINGS`` at the bottom of this file.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# R1 Pro arm constants — single source of truth for array indexing
# ---------------------------------------------------------------------------
R1_PRO_JOINTS_PER_ARM = 7
R1_PRO_LEFT_ARM_SLICE = slice(0, R1_PRO_JOINTS_PER_ARM)  # dims 0-6
R1_PRO_LEFT_GRIPPER_INDEX = R1_PRO_JOINTS_PER_ARM  # dim 7
R1_PRO_RIGHT_ARM_SLICE = slice(
    R1_PRO_JOINTS_PER_ARM + 1,
    2 * R1_PRO_JOINTS_PER_ARM + 1,
)  # dims 8-14
R1_PRO_RIGHT_GRIPPER_INDEX = 2 * R1_PRO_JOINTS_PER_ARM + 1  # dim 15
R1_PRO_TOTAL_STATE_DIMS = 2 * (R1_PRO_JOINTS_PER_ARM + 1)  # 16

ALOHA_JOINTS_PER_ARM = 6
ALOHA_TOTAL_STATE_DIMS = 2 * (ALOHA_JOINTS_PER_ARM + 1)  # 14


# ---------------------------------------------------------------------------
# Mapping name — finite value set as an enum
# ---------------------------------------------------------------------------
class MappingName(StrEnum):
    DROID_DUAL_ARM = "droid-dual-arm"
    ALOHA = "aloha"


# ---------------------------------------------------------------------------
# Result container returned by ``unpack_actions``
# ---------------------------------------------------------------------------
class R1ProActionCommand(NamedTuple):
    """Unpacked action command ready to publish on R1 Pro ROS topics."""

    left_arm_joint_positions: np.ndarray  # (7,) radians
    right_arm_joint_positions: np.ndarray  # (7,) radians
    left_gripper_position: float  # 0.0 = open, 1.0 = closed
    right_gripper_position: float  # 0.0 = open, 1.0 = closed


# ---------------------------------------------------------------------------
# ROS topic configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class R1ProTopicConfig:
    """All ROS 2 topic names needed by a mapping — cameras, joints, and commands."""

    camera_topics: dict[str, str]
    left_arm_feedback_topic: str = "/hdas/feedback_arm_left"
    right_arm_feedback_topic: str = "/hdas/feedback_arm_right"
    left_gripper_feedback_topic: str = "/hdas/feedback_gripper_left"
    right_gripper_feedback_topic: str = "/hdas/feedback_gripper_right"
    left_arm_command_topic: str = "/motion_target/target_joint_state_arm_left"
    right_arm_command_topic: str = "/motion_target/target_joint_state_arm_right"
    left_gripper_command_topic: str = "/motion_control/position_control_gripper_left"
    right_gripper_command_topic: str = "/motion_control/position_control_gripper_right"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbodimentMapping(abc.ABC):
    """Converts between R1 Pro sensor data and an OpenPI observation format."""

    @abc.abstractmethod
    def topic_config(self) -> R1ProTopicConfig:
        """Return the full ROS topic configuration for this mapping."""

    @abc.abstractmethod
    def image_size(self) -> tuple[int, int]:
        """Target ``(height, width)`` for resized camera images."""

    @abc.abstractmethod
    def build_observation(
        self,
        *,
        left_arm_positions: np.ndarray,
        right_arm_positions: np.ndarray,
        left_gripper_position: float,
        right_gripper_position: float,
        images: dict[str, np.ndarray],
        prompt: str,
    ) -> dict:
        """Build the observation dict that gets sent to the OpenPI server.

        ``images`` keys match the camera name keys from ``topic_config().camera_topics``
        and are already resized to ``image_size()``.
        """

    @abc.abstractmethod
    def unpack_actions(self, server_response: dict) -> R1ProActionCommand:
        """Extract per-actuator commands from the server's action output."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HWC uint8 image to (height, width, 3)."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Convert (H, W, C) uint8 image to (C, H, W)."""
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def _first_action_step(server_response: dict) -> np.ndarray:
    """Extract the first action step from a possibly batched action horizon."""
    actions = np.asarray(server_response["actions"])
    if actions.ndim == 2:
        return actions[0]
    return actions


# ---------------------------------------------------------------------------
# DROID-style dual-arm mapping (recommended for R1 Pro)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DroidDualArmMapping(EmbodimentMapping):
    """Maps R1 Pro's dual 7-DOF arms into the DROID observation format.

    State layout (16 dims):
        dims 0-6:   left arm joint positions (7 radians)
        dim 7:      left gripper (0.0 = open, 1.0 = closed)
        dims 8-14:  right arm joint positions (7 radians)
        dim 15:     right gripper (0.0 = open, 1.0 = closed)

    Camera mapping (3 model views):
        base_0_rgb        <- head camera
        left_wrist_0_rgb  <- left wrist camera
        right_wrist_0_rgb <- right wrist camera

    Use with ``pi05_droid`` or ``pi0_droid`` server config. The server's
    ``DroidOutputs`` only slices the first 8 dims so you will need a server
    config that returns all 16 dims. Alternatively you can use ``pi05_base``
    and handle normalization on the client.
    """

    def topic_config(self) -> R1ProTopicConfig:
        return R1ProTopicConfig(
            camera_topics={
                "cam_head": "/hdas/camera_head/left_raw/image_raw_color/compressed",
                "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
                "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
            },
        )

    def image_size(self) -> tuple[int, int]:
        return (224, 224)

    def build_observation(
        self,
        *,
        left_arm_positions: np.ndarray,
        right_arm_positions: np.ndarray,
        left_gripper_position: float,
        right_gripper_position: float,
        images: dict[str, np.ndarray],
        prompt: str,
    ) -> dict:
        height, width = self.image_size()
        head_image = _resize_image(images["cam_head"], height, width)
        left_wrist_image = _resize_image(images["cam_left_wrist"], height, width)
        right_wrist_image = _resize_image(images["cam_right_wrist"], height, width)

        return {
            "observation/exterior_image_1_left": head_image,
            "observation/wrist_image_left": left_wrist_image,
            "observation/wrist_image_right": right_wrist_image,
            "observation/joint_position": np.asarray(left_arm_positions),
            "observation/joint_position_right": np.asarray(right_arm_positions),
            "observation/gripper_position": np.array([left_gripper_position]),
            "observation/gripper_position_right": np.array([right_gripper_position]),
            "prompt": prompt,
        }

    def unpack_actions(self, server_response: dict) -> R1ProActionCommand:
        actions = _first_action_step(server_response)

        return R1ProActionCommand(
            left_arm_joint_positions=actions[R1_PRO_LEFT_ARM_SLICE],
            right_arm_joint_positions=actions[R1_PRO_RIGHT_ARM_SLICE],
            left_gripper_position=float(actions[R1_PRO_LEFT_GRIPPER_INDEX]),
            right_gripper_position=float(actions[R1_PRO_RIGHT_GRIPPER_INDEX]),
        )


# ---------------------------------------------------------------------------
# ALOHA-style mapping (alternative — drops one joint per arm)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlohaMapping(EmbodimentMapping):
    """Maps R1 Pro into the ALOHA observation format by dropping joint 0 per arm.

    ALOHA expects 6-DOF per arm (14 dims total). The R1 Pro has 7 joints per
    arm, so this mapping drops the first joint (shoulder rotation) from each arm
    and maps the remaining 6 joints + gripper into the ALOHA layout.

    State layout (14 dims):
        dims 0-5:   left arm joints 1-6 (skipping joint 0)
        dim 6:      left gripper (0.0 = open, 1.0 = closed)
        dims 7-12:  right arm joints 1-6 (skipping joint 0)
        dim 13:     right gripper (0.0 = open, 1.0 = closed)

    Camera mapping (4 views):
        cam_high       <- head camera
        cam_low        <- front chassis camera
        cam_left_wrist <- left wrist camera
        cam_right_wrist <- right wrist camera

    WARNING: Dropping a joint means the returned actions will also be missing
    one joint per arm. You must hold joint 0 at a fixed position or interpolate.
    This mapping is provided for experimentation; prefer DroidDualArmMapping.
    """

    # Which R1 Pro joint index to drop per arm (0-indexed).
    dropped_joint_index: int = 0

    def topic_config(self) -> R1ProTopicConfig:
        return R1ProTopicConfig(
            camera_topics={
                "cam_high": "/hdas/camera_head/left_raw/image_raw_color/compressed",
                "cam_low": "/hdas/camera_chassis_front_left/rgb/compressed",
                "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
                "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
            },
        )

    def image_size(self) -> tuple[int, int]:
        return (224, 224)

    def _drop_joint(self, positions: np.ndarray) -> np.ndarray:
        """Drop one joint from a 7-element array to produce 6 elements."""
        return np.delete(positions, self.dropped_joint_index)

    def _restore_joint(self, positions_6dof: np.ndarray, held_value: float = 0.0) -> np.ndarray:
        """Insert a held joint value back into a 6-element array to produce 7."""
        return np.insert(positions_6dof, self.dropped_joint_index, held_value)

    def build_observation(
        self,
        *,
        left_arm_positions: np.ndarray,
        right_arm_positions: np.ndarray,
        left_gripper_position: float,
        right_gripper_position: float,
        images: dict[str, np.ndarray],
        prompt: str,
    ) -> dict:
        left_6dof = self._drop_joint(left_arm_positions)
        right_6dof = self._drop_joint(right_arm_positions)

        # ALOHA state: [left_6j, left_grip, right_6j, right_grip] = 14 dims
        state = np.concatenate(
            [
                left_6dof,
                np.array([left_gripper_position]),
                right_6dof,
                np.array([right_gripper_position]),
            ]
        )

        height, width = self.image_size()
        cam_images = {
            name: _hwc_to_chw(_resize_image(images[name], height, width))
            for name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
        }

        return {
            "state": state,
            "images": cam_images,
            "prompt": prompt,
        }

    def unpack_actions(self, server_response: dict) -> R1ProActionCommand:
        actions = _first_action_step(server_response)
        actions_14 = actions[:ALOHA_TOTAL_STATE_DIMS]

        left_6dof = actions_14[:ALOHA_JOINTS_PER_ARM]
        left_gripper = float(actions_14[ALOHA_JOINTS_PER_ARM])
        right_6dof = actions_14[ALOHA_JOINTS_PER_ARM + 1 : ALOHA_TOTAL_STATE_DIMS - 1]
        right_gripper = float(actions_14[ALOHA_TOTAL_STATE_DIMS - 1])

        return R1ProActionCommand(
            left_arm_joint_positions=self._restore_joint(left_6dof),
            right_arm_joint_positions=self._restore_joint(right_6dof),
            left_gripper_position=left_gripper,
            right_gripper_position=right_gripper,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
AVAILABLE_MAPPINGS: dict[MappingName, EmbodimentMapping] = {
    MappingName.DROID_DUAL_ARM: DroidDualArmMapping(),
    MappingName.ALOHA: AlohaMapping(),
}
