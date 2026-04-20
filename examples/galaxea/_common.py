"""Galaxea-specific types and configuration for Galaxea R1 -> OpenPI mapping.

This module re-exports the shared types used across all robots and defines
Galaxea-specific configuration (robot models, ROS 2 topic names, joint counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Re-export shared types so existing ``from _common import ...`` statements
# in Galaxea code continue to work without changes.
from shared.types import (  # noqa: F401
    ALOHA_JOINTS_PER_ARM,
    ALOHA_TOTAL_STATE_DIMS,
    ActionCommand,
    MappingName,
    first_action_step,
)


# ---------------------------------------------------------------------------
# Robot model — determines joints-per-arm and which mappings are available
# ---------------------------------------------------------------------------
class RobotModel(StrEnum):
    R1_PRO = "r1-pro"
    R1_LITE = "r1-lite"


JOINTS_PER_ARM: dict[RobotModel, int] = {
    RobotModel.R1_PRO: 7,
    RobotModel.R1_LITE: 6,
}


# ---------------------------------------------------------------------------
# ROS topic configuration — identical topic names across R1 Pro and R1 Lite
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GalaxeaTopicConfig:
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
