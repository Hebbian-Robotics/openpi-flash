"""Shared types and helpers for bimanual robot -> OpenPI embodiment mappings.

This module contains the data containers, constants, and utility functions
shared across all robot variants (Galaxea R1, YOR, etc.).
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Shared ALOHA constants (6-DOF is ALOHA's native format)
# ---------------------------------------------------------------------------
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
class ActionCommand(NamedTuple):
    """Unpacked action command ready to send to a bimanual robot.

    The length of ``left_arm_joint_positions`` and ``right_arm_joint_positions``
    matches the robot's DOF per arm (e.g. 7 for R1 Pro, 6 for R1 Lite / YOR).
    """

    left_arm_joint_positions: np.ndarray  # (N,) radians — N = joints per arm
    right_arm_joint_positions: np.ndarray  # (N,) radians
    left_gripper_position: float  # 0.0 = open, 1.0 = closed
    right_gripper_position: float  # 0.0 = open, 1.0 = closed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def first_action_step(server_response: dict) -> np.ndarray:
    """Extract the first action step from a possibly batched action horizon."""
    actions = np.asarray(server_response["actions"])
    if actions.ndim == 2:
        return actions[0]
    return actions
