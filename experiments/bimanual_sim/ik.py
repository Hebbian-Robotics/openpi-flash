"""Differential IK over a Piper arm's 6-DOF chain, driven by mink.

`mink.solve_ik` is a QP (ProxQP-backed by default, switchable via `solver=...`)
over joint velocities, with joint limits and optional tasks. We iterate it to
convergence on a target frame — the `{prefix}tcp` site declared by the scene.

ProxQP (via `proxsuite`) is typically 2-5x faster than DAQP on small 6-DoF QPs
and pays back the planning-phase wait (~100 QPs per waypoint x ~13 waypoints
x 2 arms at scene startup). Override with `solver="daqp"` to fall back.

Three orientation modes encoded as a discriminated union so the three
combinations are mutually exclusive at the type level:

  * `PositionOnly()` — position-only; gripper orientation is free.
  * `AlignGripperDown()` — keep TCP +z pointing world -z; rotation about
    that axis is free.
  * `FullPose(target_quat_wxyz)` — full pose tracking.

Compared to the old hand-rolled DLS loop, joint limits are enforced as hard
QP constraints instead of post-step clamping — which is what used to park
the solver at Piper's ±70° j5 limit on top-down targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
import viser.transforms as vtf

from arm_handles import ArmHandles
from scene_base import JointConfig, Position3, QuatWxyz


@dataclass(frozen=True)
class PositionOnly:
    """Match TCP position; orientation free."""


@dataclass(frozen=True)
class AlignGripperDown:
    """Match TCP position AND keep TCP +z → world -z; in-axis rotation free."""


@dataclass(frozen=True)
class FullPose:
    """Match full TCP pose (wxyz quaternion)."""

    target_quat_wxyz: QuatWxyz


OrientationMode = PositionOnly | AlignGripperDown | FullPose


def _target_pose(target_pos: Position3, orientation: OrientationMode) -> mink.SE3:
    match orientation:
        case PositionOnly():
            # Orientation is weighted 0 in the task cost; quat is a placeholder.
            rot = vtf.SO3.identity()
        case AlignGripperDown():
            # 180° about world +x maps site +z (the gripper axis) → world -z.
            rot = vtf.SO3.from_x_radians(np.pi)
        case FullPose(target_quat_wxyz):
            rot = vtf.SO3(wxyz=np.asarray(target_quat_wxyz, dtype=float))

    mat4 = np.eye(4)
    mat4[:3, :3] = rot.as_matrix()
    mat4[:3, 3] = np.asarray(target_pos, dtype=float)
    return mink.SE3.from_matrix(mat4)


def _costs(orientation: OrientationMode) -> tuple[float, float]:
    match orientation:
        case PositionOnly():
            return 1.0, 0.0
        case AlignGripperDown():
            return 1.0, 0.5
        case FullPose():
            return 1.0, 1.0


_MAX_JOINT_VEL_RAD_S = np.pi  # 180°/s — sane manipulation cap
_LOCKED_JOINT_VEL = 1e-6  # effectively frozen


def _velocity_limit(
    model: mujoco.MjModel,
    locked_joint_names: tuple[str, ...] = (),
) -> mink.VelocityLimit:
    """Build a VelocityLimit over every single-DOF joint.

    Hinge joints get the default manipulation cap; any joint whose name is in
    `locked_joint_names` is clamped to ~0 — use this to freeze scene-owned
    DOFs (a lift prismatic, a second arm's joints that shouldn't move during
    this IK call) so the QP can't "solve" them to compensate.

    Without this cap at all, mink's QP happily returns 100+ rad/s when the
    position error is large; the integrator then overshoots wildly.
    """
    limits: dict[str, float] = {}
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        limits[name] = _MAX_JOINT_VEL_RAD_S
    for name in locked_joint_names:
        limits[name] = _LOCKED_JOINT_VEL
    return mink.VelocityLimit(model, limits)


# Keep the old name working for sink_bimanual (no locked joints needed there).
def _velocity_limit_for_all_hinges(model: mujoco.MjModel) -> mink.VelocityLimit:
    return _velocity_limit(model, locked_joint_names=())


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
    target_pos: Position3,
    *,
    orientation: OrientationMode = PositionOnly(),
    seed_q: JointConfig | None = None,
    max_iters: int = 400,
    rate_dt: float = 0.02,
    pos_tol: float = 1.5e-3,
    rot_tol: float = 2e-2,
    solver: str = "proxqp",
    damping: float = 1e-5,
    locked_joint_names: tuple[str, ...] = (),
) -> tuple[JointConfig, float]:
    """Place the `{prefix}tcp` site at the target; return (solved_arm_q, err).

    Only the target arm's joints are free to move — the other arm's DOFs don't
    appear in the FrameTask's Jacobian, so with a small velocity damping they
    stay at zero velocity throughout the solve.
    """
    if seed_q is not None:
        for i, idx in enumerate(arm.arm_qpos_idx):
            data.qpos[idx] = seed_q[i]

    configuration = mink.Configuration(model, q=data.qpos.copy())

    position_cost, orientation_cost = _costs(orientation)
    frame_task = mink.FrameTask(
        frame_name=arm.tcp_site_name,
        frame_type="site",
        position_cost=position_cost,
        orientation_cost=orientation_cost,
    )
    frame_task.set_target(_target_pose(np.asarray(target_pos, dtype=float), orientation))

    limits: list[mink.Limit] = [
        mink.ConfigurationLimit(model),
        _velocity_limit(model, locked_joint_names),
    ]

    # Track "best" using the same weighting as the task cost — otherwise a
    # PositionOnly solve would score the seed as best (its rot_err is small
    # while the solved config's rot_err has drifted to ~pi), and the function
    # would return the unchanged seed even though the arm reached the target.
    def scored_err(pos: float, rot: float) -> float:
        return position_cost * pos + orientation_cost * rot

    best_score = np.inf
    best_q = np.array([configuration.q[i] for i in arm.arm_qpos_idx])
    best_pos_err = np.inf
    best_rot_err = np.inf

    for _ in range(max_iters):
        err = frame_task.compute_error(configuration)  # [err_pos(3), err_rot(3)]
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))
        score = scored_err(pos_err, rot_err)

        if score < best_score:
            best_score = score
            best_pos_err = pos_err
            best_rot_err = rot_err
            best_q = np.array([configuration.q[i] for i in arm.arm_qpos_idx])

        if pos_err < pos_tol and (orientation_cost == 0.0 or rot_err < rot_tol):
            break

        velocity = mink.solve_ik(
            configuration,
            [frame_task],
            rate_dt,
            solver,
            limits=limits,
            damping=damping,
        )
        configuration.integrate_inplace(velocity, rate_dt)

    # Commit best arm config back to the caller's data.
    for i, idx in enumerate(arm.arm_qpos_idx):
        data.qpos[idx] = best_q[i]
    mujoco.mj_kinematics(model, data)
    # Report the position error explicitly — callers care about "did the TCP
    # reach the target", not the scored combination.
    reported = (
        best_pos_err if orientation_cost == 0.0 else float(np.hypot(best_pos_err, best_rot_err))
    )
    return best_q, reported
