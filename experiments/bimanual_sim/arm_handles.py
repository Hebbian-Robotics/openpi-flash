"""Piper-specific arm accessors.

`get_arm_handles` looks up the fully-qualified joint, actuator, body, and
equality IDs for one prefixed Piper instance in a compiled model. The result
is a lightweight dataclass the runner/IK/weld code passes around instead of
re-doing name lookups every tick.

This helper assumes the Piper MJCF's naming: `{prefix}joint1..8`,
`{prefix}gripper`, `{prefix}link6`, `{prefix}grasp_cube{i}`. A scene using a
different arm would need its own handles builder.

Arm identity is modelled as a `StrEnum` whose members' values are the MuJoCo
name prefixes ("left_", "right_"). Because `StrEnum` inherits from `str`,
f-string concatenation (`f"{side}joint1"`) yields the expected body names
without any `.value` boilerplate, while downstream code that used to take
`prefix: str` now rejects arbitrary strings at type-check time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np


class ArmSide(StrEnum):
    """Bimanual arm identity; value is the MuJoCo body-name prefix."""

    LEFT = "left_"
    RIGHT = "right_"


@dataclass
class ArmHandles:
    side: ArmSide
    qpos_idx: np.ndarray  # 8 qpos indices: joint1..6, finger_l (joint7), finger_r (joint8)
    dof_idx: np.ndarray  # 8 dof indices, same ordering
    arm_dof_idx: np.ndarray  # first 6 dof indices (arm joints only)
    jnt_ids: np.ndarray  # joint ids (length 8)
    act_arm_ids: np.ndarray  # actuator ids for joint1..6 position actuators
    act_gripper_id: int  # actuator id for the gripper
    link6_id: int  # body id at end of arm (parent of fingers)
    tcp_site_id: int  # site id at the grip center (link6 + 0.14 m along +z)
    gripper_open: float  # ctrl value that opens the gripper
    gripper_closed: float  # ctrl value that closes the gripper
    weld_ids: np.ndarray  # equality ids, one per cube (may be empty if n_cubes=0)

    @property
    def arm_qpos_idx(self) -> np.ndarray:
        return self.qpos_idx[:6]

    @property
    def tcp_site_name(self) -> str:
        return f"{self.side}tcp"


def get_arm_handles(model: mujoco.MjModel, side: ArmSide, n_cubes: int) -> ArmHandles:
    jnt_names = [f"{side}joint{i}" for i in range(1, 9)]
    jnt_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in jnt_names])
    qpos_idx = np.array([model.jnt_qposadr[j] for j in jnt_ids])
    dof_idx = np.array([model.jnt_dofadr[j] for j in jnt_ids])
    act_arm_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}joint{i}")
            for i in range(1, 7)
        ]
    )
    act_gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}gripper")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}link6")
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}tcp")

    # Gripper actuator controls joint7 (range 0..0.035). Higher ctrl = more open.
    gripper_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}joint7")
    lo, hi = model.jnt_range[gripper_jnt]

    weld_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}grasp_cube{i}")
            for i in range(n_cubes)
        ],
        dtype=np.int64,
    )

    return ArmHandles(
        side=side,
        qpos_idx=qpos_idx,
        dof_idx=dof_idx,
        arm_dof_idx=dof_idx[:6],
        jnt_ids=jnt_ids,
        act_arm_ids=act_arm_ids,
        act_gripper_id=act_gripper_id,
        link6_id=link6_id,
        tcp_site_id=tcp_site_id,
        gripper_open=float(hi),
        gripper_closed=float(lo),
        weld_ids=weld_ids,
    )
