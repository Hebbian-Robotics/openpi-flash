"""Weld-equality "cheat" grasp.

MuJoCo parallel-jaw grasping with simulated friction is notoriously unreliable
— cubes slip out under lateral acceleration, or require painstaking contact
tuning. Instead, every (arm, cube) pair has a pre-declared `mjEQ_WELD`
equality whose `active` flag we flip on and off.

At activation we teleport the cube onto the TCP site and freeze the current
relative pose by writing into `model.eq_data[eq_id]`. MuJoCo then holds the
cube rigidly to link6. Releasing is just clearing the active flag.

Rotation math goes through `viser.transforms.SO3` (already a viser transitive
dep) so we can avoid the verbose `mju_mat2Quat`/`mju_mulQuat` dance.
"""

from __future__ import annotations

import mujoco
import numpy as np
import viser.transforms as vtf

WorldPose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
"""(world_xyz, quat_wxyz) — explicit pose target for `activate_attachment_weld`."""


def activate_grasp_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq_id: int,
    hand_body_id: int,
    cube_body_id: int,
    tcp_site_id: int,
) -> None:
    """Teleport the cube onto the TCP site, then enable the weld locked at
    that pose."""
    jnt_id = int(model.body_jntadr[cube_body_id])
    qpos_start = int(model.jnt_qposadr[jnt_id])
    qvel_start = int(model.jnt_dofadr[jnt_id])

    tcp_pos = data.site_xpos[tcp_site_id].copy()
    tcp_mat = data.site_xmat[tcp_site_id].reshape(3, 3).copy()
    tcp_rot = vtf.SO3.from_matrix(tcp_mat)

    # Cube's freejoint qpos layout: [x, y, z, qw, qx, qy, qz].
    data.qpos[qpos_start : qpos_start + 3] = tcp_pos
    data.qpos[qpos_start + 3 : qpos_start + 7] = tcp_rot.wxyz
    data.qvel[qvel_start : qvel_start + 6] = 0.0
    mujoco.mj_forward(model, data)

    # Relative pose of the cube in the hand (link6) frame.
    hand_pos = data.xpos[hand_body_id].copy()
    hand_mat = data.xmat[hand_body_id].reshape(3, 3).copy()
    hand_rot = vtf.SO3.from_matrix(hand_mat)
    cube_pos = data.xpos[cube_body_id].copy()
    cube_rot = vtf.SO3.from_matrix(data.xmat[cube_body_id].reshape(3, 3).copy())

    rel_pos = hand_mat.T @ (cube_pos - hand_pos)
    rel_rot = hand_rot.inverse() @ cube_rot

    # mjEQ_WELD data layout: anchor(3), relpose pos(3), relpose quat(4), torquescale(1).
    # `eq_data` lives on model (static layout); poking it at runtime is legal.
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_rot.wxyz
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1


def deactivate_grasp_weld(data: mujoco.MjData, eq_id: int) -> None:
    data.eq_active[eq_id] = 0


def activate_attachment_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq_id: int,
    body_a_id: int,
    body_b_id: int,
    *,
    target_world_pose: WorldPose | None = None,
) -> None:
    """Activate a body↔body weld equality.

    Two modes:

    * **target_world_pose=None** (legacy): freeze the *current* relative
      pose between body_a and body_b. Both bodies stay where they are.

    * **target_world_pose=((x, y, z), (qw, qx, qy, qz))**: target is the
      desired world pose of **body_a** (the movable body in the
      attachment — convention: scene declares welds with the moving
      object as `name1`/`obj1`, the static parent as `name2`/`obj2`).
      We compute the relpose using body_b's actual current world pose
      so the weld, once active, snaps body_a to the requested world
      pose while body_b stays put.
    """
    if target_world_pose is None:
        a_pos = data.xpos[body_a_id].copy()
        a_mat = data.xmat[body_a_id].reshape(3, 3).copy()
        a_rot = vtf.SO3.from_matrix(a_mat)
    else:
        target_xyz, target_quat_wxyz = target_world_pose
        a_pos = np.asarray(target_xyz, dtype=float)
        a_rot = vtf.SO3(wxyz=np.asarray(target_quat_wxyz, dtype=float))
        a_mat = a_rot.as_matrix()

    b_pos = data.xpos[body_b_id].copy()
    b_rot = vtf.SO3.from_matrix(data.xmat[body_b_id].reshape(3, 3).copy())

    rel_pos = a_mat.T @ (b_pos - a_pos)
    rel_rot = a_rot.inverse() @ b_rot

    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_rot.wxyz
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1


def deactivate_weld(data: mujoco.MjData, eq_id: int) -> None:
    """Alias for deactivate_grasp_weld — same action, different semantic name."""
    data.eq_active[eq_id] = 0
