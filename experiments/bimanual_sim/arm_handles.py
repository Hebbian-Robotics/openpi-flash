"""Per-arm handle resolution, dispatched by robot kind.

`get_arm_handles(model, side, n_cubes, robot_kind)` returns an
`ArmHandles` carrying the qpos/dof/actuator/body indices the runner +
IK code use to drive one prefixed arm. Two robot families are supported:

* `"piper"` — AgileX Piper 6-DoF arm + parallel-jaw with two
  tendon-coupled finger slide joints (`{side}joint7` / `{side}joint8`).
  `qpos_idx` / `dof_idx` are length-8 arrays (joints 1..8).
* `"ur10e"` — Universal Robots UR10e + Robotiq 2F-85 grippper. UR has
  6 named revolute joints (`{side}shoulder_pan_joint` …
  `{side}wrist_3_joint`); the 2F-85's tendon-driven 4-bar linkage is
  driven by a single `{side}fingers_actuator` (ctrl 0..255). Finger
  joint qpos is NOT puppet-written — the actuator pushes the tendon
  equality and the linkage settles. `qpos_idx` / `dof_idx` are
  length-6 arrays (no finger entries).

Per-robot constants (joint name lists, gripper ranges) live here so
the scene module's only obligation is declaring `ROBOT_KIND`.
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
"""UR10e arm DoFs in canonical chain order. The 2F-85 mounted at the
wrist has its own joints but they're tendon-driven and not addressed
through `qpos_idx` / `dof_idx`."""


def arm_joint_suffixes(robot_kind: RobotKind) -> tuple[str, ...]:
    """Return the canonical-order arm joint suffixes for a robot kind.

    Single source of truth shared by `get_arm_handles` (resolves IDs
    from these), the runner (per-tick rerun scalar logging), and
    teleop (per-joint slider labels). Callers must NOT redefine this
    list locally — fixing one place + missing the others is the
    classic drift bug, and these enums are short enough that the
    type system can't fully prevent transposition.
    """
    if robot_kind == "piper":
        # Drop the two finger slides for piper — only the 6 arm DoFs
        # are user-relevant for IK / per-joint sliders.
        return _PIPER_ARM_JOINT_SUFFIXES[:6]
    if robot_kind == "ur10e":
        return _UR10E_ARM_JOINT_SUFFIXES
    raise ValueError(f"unknown robot_kind: {robot_kind!r}")


@dataclass
class ArmHandles:
    side: ArmSide
    robot_kind: RobotKind
    # qpos/dof indices for the per-tick puppet writes. Length depends on
    # robot_kind: piper=8 (joints 1..8), ur10e=6 (UR's 6 DoFs).
    qpos_idx: np.ndarray
    dof_idx: np.ndarray
    # Joint ids and arm-DoF subset (always the first 6 entries — UR has
    # exactly 6, Piper's first 6 are the arm joints, the trailing 2
    # are gripper fingers).
    jnt_ids: np.ndarray
    arm_dof_idx: np.ndarray
    # Position-actuator ids for the 6 arm DoFs (used by the runner to
    # mirror puppet qpos into ctrl so the position servos don't fight
    # back with stale targets).
    act_arm_ids: np.ndarray
    # Single gripper actuator id. Piper: `{side}gripper`. UR10e + 2F-85:
    # `{side}fingers_actuator`.
    act_gripper_id: int
    # Body the grasp weld attaches to. Piper: link6 (parent of fingers).
    # UR10e: wrist_3_link (parent of the 2F-85 attachment frame).
    link6_id: int
    tcp_site_id: int
    gripper_open: float  # ctrl/qpos value for "open"
    gripper_closed: float  # ctrl/qpos value for "closed"
    weld_ids: np.ndarray  # equality ids, one per cube (may be empty)

    @property
    def arm_qpos_idx(self) -> np.ndarray:
        """qpos indices for the 6 arm DoFs only (drops Piper's gripper
        slides; same as `qpos_idx` for UR10e since it's already 6-long)."""
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
        # `robots/ur10e.py` to avoid a `base` body-name collision with
        # the UR), so the driving actuator's compiled name is
        # `<side>/gripper/fingers_actuator`.
        gripper_actuator_suffix = "gripper/fingers_actuator"
        # UR10e's tool flange — the 2F-85 mounts here, and the grasp
        # weld pins held objects to this body.
        wrist_body_suffix = "wrist_3_link"
    else:  # pragma: no cover — Literal exhausts at type-check time
        raise ValueError(f"unknown robot_kind: {robot_kind!r}")

    jnt_names = [f"{side}{suffix}" for suffix in joint_suffixes]
    jnt_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n, "joint") for n in jnt_names]
    )
    qpos_idx = np.array([model.jnt_qposadr[j] for j in jnt_ids])
    dof_idx = np.array([model.jnt_dofadr[j] for j in jnt_ids])

    # Arm position actuators — one per arm DoF (Piper: joint1..joint6;
    # UR10e: shoulder_pan/shoulder_lift/elbow/wrist_1/wrist_2/wrist_3,
    # which drop the `_joint` suffix per the upstream Menagerie naming).
    if robot_kind == "piper":
        act_arm_names = [f"{side}joint{i}" for i in range(1, 7)]
    else:  # ur10e
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

    # Gripper open/closed values.
    if robot_kind == "piper":
        # Piper joint7 is a slide (range 0..0.035 m). Joint8 mirrors it
        # via tendon. We read the joint range to derive open/closed
        # rather than hardcoding numerics.
        gripper_jnt_id = _resolve_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}joint7", "Piper joint7"
        )
        lo, hi = model.jnt_range[gripper_jnt_id]
        gripper_open = float(hi)
        gripper_closed = float(lo)
    else:  # ur10e + 2F-85
        # Robotiq 2F-85 driving actuator: ctrlrange [0, 255] — 0 fully
        # open, 255 fully closed. Tendon equality propagates the
        # actuator force to the 4-bar finger linkage.
        gripper_open = 0.0
        gripper_closed = 255.0

    # Weld equalities for grasp interactions (one per cube). Names
    # follow the same `<side>_grasp_cube<i>` convention regardless of
    # robot kind — the grasp-weld registry is robot-agnostic.
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
