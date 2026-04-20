"""Pluggable embodiment mappings for bimanual robots -> OpenPI.

Each mapping converts raw sensor data (joint positions, camera images) into
the observation dict expected by a specific OpenPI model config and converts
the returned action array back into per-actuator commands.

Robot-specific differences (joints per arm, camera layout) are captured as
constructor parameters — the mapping classes themselves are shared across
all supported robots (Galaxea R1, YOR, etc.).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from .types import (
    ALOHA_JOINTS_PER_ARM,
    ALOHA_TOTAL_STATE_DIMS,
    ActionCommand,
    first_action_step,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbodimentMapping(abc.ABC):
    """Converts between bimanual robot sensor data and an OpenPI observation format."""

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

        ``images`` keys match camera name identifiers and values are HWC uint8
        arrays already captured from the robot's cameras.
        """

    @abc.abstractmethod
    def unpack_actions(self, server_response: dict) -> ActionCommand:
        """Extract per-actuator commands from the server's action output."""

    @property
    @abc.abstractmethod
    def camera_names(self) -> tuple[str, ...]:
        """Return the camera name identifiers expected by this mapping."""


# ---------------------------------------------------------------------------
# DROID-style dual-arm mapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DroidDualArmMapping(EmbodimentMapping):
    """Maps a bimanual robot into the DROID observation format.

    State layout (2 * (joints_per_arm + 1) dims):
        dims 0..N-1:       left arm joint positions (N radians)
        dim N:             left gripper (0.0 = open, 1.0 = closed)
        dims N+1..2N:      right arm joint positions (N radians)
        dim 2N+1:          right gripper (0.0 = open, 1.0 = closed)

    Camera images are forwarded as native-resolution HWC uint8 arrays and
    placed into the observation dict under the keys specified by
    ``camera_observation_keys``. The client transport consumes them at
    native shape and applies aspect-preserving ``resize_with_pad`` to
    224x224 HWC uint8 before the QUIC wire, driven by the ``image_specs``
    the server advertises at handshake.
    """

    joints_per_arm: int
    camera_observation_keys: dict[str, str]
    """Maps camera name -> observation dict key (e.g. ``cam_head`` -> ``observation/exterior_image_1_left``)."""

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.camera_observation_keys.keys())

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
        obs: dict = {}
        for camera_name, observation_key in self.camera_observation_keys.items():
            obs[observation_key] = images[camera_name]

        obs["observation/joint_position"] = np.asarray(left_arm_positions)
        obs["observation/joint_position_right"] = np.asarray(right_arm_positions)
        obs["observation/gripper_position"] = np.array([left_gripper_position])
        obs["observation/gripper_position_right"] = np.array([right_gripper_position])
        obs["prompt"] = prompt
        return obs

    def unpack_actions(self, server_response: dict) -> ActionCommand:
        actions = first_action_step(server_response)
        n = self.joints_per_arm

        return ActionCommand(
            left_arm_joint_positions=actions[0:n],
            right_arm_joint_positions=actions[n + 1 : 2 * n + 1],
            left_gripper_position=float(actions[n]),
            right_gripper_position=float(actions[2 * n + 1]),
        )


# ---------------------------------------------------------------------------
# ALOHA-style dual-arm mapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlohaMapping(EmbodimentMapping):
    """Maps a bimanual robot into the ALOHA observation format.

    ALOHA expects 6-DOF per arm (14 dims total).  Robots with more than 6
    joints per arm (e.g. R1 Pro with 7) must drop one joint; set
    ``dropped_joint_index`` to the index to drop.  Robots that are already
    6-DOF (e.g. R1 Lite, YOR) leave ``dropped_joint_index`` as ``None``.

    State layout (14 dims):
        dims 0-5:   left arm joint positions (6 radians)
        dim 6:      left gripper (0.0 = open, 1.0 = closed)
        dims 7-12:  right arm joint positions (6 radians)
        dim 13:     right gripper (0.0 = open, 1.0 = closed)

    Camera images are forwarded as native-resolution HWC uint8 arrays under
    ``obs["images"][camera_name]``. The client transport resizes them with
    ``resize_with_pad`` and transposes to the CHW 3x224x224 layout the
    ALOHA model expects, driven by the ``image_specs`` the server advertises
    at handshake. If the robot lacks a ``cam_low`` topic (e.g. R1 Lite has
    no chassis camera), set ``use_dummy_cam_low=True`` to send a black
    placeholder for servers that require 4 cameras.
    """

    camera_names_config: tuple[str, ...]
    """Camera name identifiers expected by this mapping (e.g. ``("cam_high", "cam_left_wrist", ...)``).
    """

    dropped_joint_index: int | None = None
    use_dummy_cam_low: bool = False

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self.camera_names_config

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
        if self.dropped_joint_index is not None:
            left_joints = np.delete(left_arm_positions, self.dropped_joint_index)
            right_joints = np.delete(right_arm_positions, self.dropped_joint_index)
        else:
            left_joints = np.asarray(left_arm_positions)
            right_joints = np.asarray(right_arm_positions)

        state = np.concatenate(
            [
                left_joints,
                np.array([left_gripper_position]),
                right_joints,
                np.array([right_gripper_position]),
            ]
        )

        cam_images: dict[str, np.ndarray] = {
            name: images[name] for name in self.camera_names_config
        }

        if self.use_dummy_cam_low:
            # The transport resizes to the server-advertised target shape,
            # so a small black HWC placeholder is enough — it'll be padded to
            # whatever the model wants.
            cam_images["cam_low"] = np.zeros((224, 224, 3), dtype=np.uint8)

        return {
            "state": state,
            "images": cam_images,
            "prompt": prompt,
        }

    def unpack_actions(self, server_response: dict) -> ActionCommand:
        actions = first_action_step(server_response)
        actions_14 = actions[:ALOHA_TOTAL_STATE_DIMS]

        left_6dof = actions_14[:ALOHA_JOINTS_PER_ARM]
        left_gripper = float(actions_14[ALOHA_JOINTS_PER_ARM])
        right_6dof = actions_14[ALOHA_JOINTS_PER_ARM + 1 : ALOHA_TOTAL_STATE_DIMS - 1]
        right_gripper = float(actions_14[ALOHA_TOTAL_STATE_DIMS - 1])

        if self.dropped_joint_index is not None:
            left_arm = np.insert(left_6dof, self.dropped_joint_index, 0.0)
            right_arm = np.insert(right_6dof, self.dropped_joint_index, 0.0)
        else:
            left_arm = left_6dof
            right_arm = right_6dof

        return ActionCommand(
            left_arm_joint_positions=left_arm,
            right_arm_joint_positions=right_arm,
            left_gripper_position=left_gripper,
            right_gripper_position=right_gripper,
        )
