"""Per-arm handle resolution, dispatched by robot kind.

`get_arm_handles(model, side, n_cubes, robot_kind)` returns an
`ArmHandles` carrying the qpos/dof/actuator/body indices the runner +
IK code use to drive one prefixed arm. Two robot families are supported:

* `"piper"` — AgileX Piper 6-DoF arm + parallel-jaw with two
  tendon-coupled finger slide joints. `qpos_idx` / `dof_idx` are
  length-8 (joints 1..8).
* `"ur10e"` — Universal Robots UR10e + Robotiq 2F-85. The 2F-85's
  4-bar linkage is tendon-driven by a single `fingers_actuator`
  (ctrl 0..255); finger qpos is NOT puppet-written — the actuator
  pushes the tendon equality and the linkage settles.
  `qpos_idx` / `dof_idx` are length-6 (no finger entries).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import mujoco
import numpy as np

RobotKind = Literal["piper", "ur10e"]


class ArmSide(StrEnum):
    """Bimanual arm identity; value is the MuJoCo body-name prefix.

    The trailing `/` is dm_control.mjcf's namespace separator: when a
    sub-MJCF (Piper or UR10e) is attached with `model="left"`, every
    body/joint/actuator inside it is renamed `left/<original>`.
    f-string concatenation (`f"{side}link6"` /
    `f"{side}wrist_3_link"`) naturally produces the slash-namespaced
    compiled name.
    """

    LEFT = "left/"
    RIGHT = "right/"


_PIPER_ARM_JOINT_SUFFIXES: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 9))
"""Piper joints 1..8 — first 6 are arm DoFs, last 2 are tendon-coupled
finger slides."""

_UR10E_ARM_JOINT_SUFFIXES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
"""UR10e arm DoFs in canonical chain order. 2F-85 finger joints are
tendon-driven and not addressed through `qpos_idx` / `dof_idx`."""


def arm_joint_suffixes(robot_kind: RobotKind) -> tuple[str, ...]:
    """Return canonical-order arm joint suffixes for a robot kind.

    Single source of truth shared by `get_arm_handles`, the runner's
    rerun scalar logging, and teleop's per-joint slider labels.
    Callers must NOT redefine this list locally.
    """
    if robot_kind == "piper":
        return _PIPER_ARM_JOINT_SUFFIXES[:6]
    if robot_kind == "ur10e":
        return _UR10E_ARM_JOINT_SUFFIXES
    raise ValueError(f"unknown robot_kind: {robot_kind!r}")


@dataclass
class ArmHandles:
    side: ArmSide
    robot_kind: RobotKind
    # piper=8 (joints 1..8), ur10e=6.
    qpos_idx: np.ndarray
    dof_idx: np.ndarray
    jnt_ids: np.ndarray
    arm_dof_idx: np.ndarray
    # Position-actuator ids for the 6 arm DoFs. The runner mirrors
    # puppet qpos into ctrl so position servos don't fight back with
    # stale targets.
    act_arm_ids: np.ndarray
    act_gripper_id: int
    # Grasp-weld parent. Piper: link6. UR10e: wrist_3_link (the 2F-85
    # attachment frame).
    link6_id: int
    tcp_site_id: int
    gripper_open: float
    gripper_closed: float
    weld_ids: np.ndarray  # equality ids, one per cube (may be empty)

    @property
    def arm_qpos_idx(self) -> np.ndarray:
        """qpos indices for the 6 arm DoFs only (drops Piper's gripper slides)."""
        return self.qpos_idx[:6]

    @property
    def tcp_site_name(self) -> str:
        return f"{self.side}tcp"


def _resolve_id(model: mujoco.MjModel, obj_type: int, name: str, kind: str) -> int:
    """Look up an MJCF element id by name; raise with context if missing."""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(
            f"{kind} {name!r} not found in compiled model. "
            "Check the scene module's `ROBOT_KIND` matches the actually-loaded robot."
        )
    return obj_id


def get_arm_handles(
    model: mujoco.MjModel,
    side: ArmSide,
    n_cubes: int,
    robot_kind: RobotKind = "piper",
) -> ArmHandles:
    """Resolve all per-arm handles. The `robot_kind` argument selects
    which joint-name convention + gripper layout to use. Scenes
    declare this via a module-level `ROBOT_KIND` attribute that the
    runner reads with `getattr`."""
    if robot_kind == "piper":
        joint_suffixes = _PIPER_ARM_JOINT_SUFFIXES
        gripper_actuator_suffix = "gripper"
        wrist_body_suffix = "link6"
    elif robot_kind == "ur10e":
        joint_suffixes = _UR10E_ARM_JOINT_SUFFIXES
        # 2F-85 mounts under a `gripper/` sub-namescope (set by
        # `robots/ur10e.py` to dodge a `base` body-name collision with
        # the UR).
        gripper_actuator_suffix = "gripper/fingers_actuator"
        wrist_body_suffix = "wrist_3_link"
    else:  # pragma: no cover — Literal exhausts at type-check time
        raise ValueError(f"unknown robot_kind: {robot_kind!r}")

    jnt_names = [f"{side}{suffix}" for suffix in joint_suffixes]
    jnt_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n, "joint") for n in jnt_names]
    )
    qpos_idx = np.array([model.jnt_qposadr[j] for j in jnt_ids])
    dof_idx = np.array([model.jnt_dofadr[j] for j in jnt_ids])

    # UR10e actuators drop the `_joint` suffix per upstream Menagerie naming.
    if robot_kind == "piper":
        act_arm_names = [f"{side}joint{i}" for i in range(1, 7)]
    else:
        act_arm_names = [
            f"{side}shoulder_pan",
            f"{side}shoulder_lift",
            f"{side}elbow",
            f"{side}wrist_1",
            f"{side}wrist_2",
            f"{side}wrist_3",
        ]
    act_arm_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n, "actuator") for n in act_arm_names]
    )
    act_gripper_id = _resolve_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"{side}{gripper_actuator_suffix}",
        "gripper actuator",
    )
    link6_id = _resolve_id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{side}{wrist_body_suffix}", "wrist body"
    )
    tcp_site_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}tcp", "TCP site")

    if robot_kind == "piper":
        # Piper joint7 is a slide (range 0..0.035 m); joint8 mirrors via
        # tendon. Read the range rather than hardcoding numerics.
        gripper_jnt_id = _resolve_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}joint7", "Piper joint7"
        )
        lo, hi = model.jnt_range[gripper_jnt_id]
        gripper_open = float(hi)
        gripper_closed = float(lo)
    else:
        # Robotiq 2F-85 ctrlrange: 0 fully open, 255 fully closed.
        gripper_open = 0.0
        gripper_closed = 255.0

    weld_ids = np.array(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                f"{side.replace('/', '_')}grasp_cube{i}",
            )
            for i in range(n_cubes)
        ],
        dtype=np.int64,
    )

    return ArmHandles(
        side=side,
        robot_kind=robot_kind,
        qpos_idx=qpos_idx,
        dof_idx=dof_idx,
        jnt_ids=jnt_ids,
        arm_dof_idx=dof_idx[:6],
        act_arm_ids=act_arm_ids,
        act_gripper_id=act_gripper_id,
        link6_id=link6_id,
        tcp_site_id=tcp_site_id,
        gripper_open=gripper_open,
        gripper_closed=gripper_closed,
        weld_ids=weld_ids,
    )
