"""Viser camera frustum widgets.

Scenes declare cameras as `(mujoco_name, CameraRole)` pairs; role picks
intrinsics (FOV / aspect / widget scale). `CameraRole` is a `StrEnum`
rather than a name-substring heuristic — substring matching silently
mishandled e.g. `topology_cam` matching `CameraRole.TOP`.

The runner calls `add_frustum_widgets(...)` once at startup and
`update_frustum_widgets(...)` each frame so the frustum tracks cameras
attached to moving bodies (lift carriage, wrist links).

Per-frame updates use the same fast path as `viser_render.update_viser`:
`mujoco.mju_mat2Quat` (not `vtf.SO3.from_matrix`), direct `buffer.push(...)`
instead of the property setter, and one `server.atomic()` around the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np
import viser
from viser._messages import SetOrientationMessage, SetPositionMessage


class CameraRole(StrEnum):
    """Classifies a scene camera so the right default intrinsics apply."""

    TOP = "top"
    WRIST = "wrist"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Viser frustum-widget intrinsics. `scale_m` is the visual size in metres."""

    fov_deg: float
    aspect: float  # width / height
    scale_m: float


_INTRINSICS: dict[CameraRole, CameraIntrinsics] = {
    CameraRole.TOP: CameraIntrinsics(fov_deg=60.0, aspect=16.0 / 9.0, scale_m=0.10),
    CameraRole.WRIST: CameraIntrinsics(fov_deg=87.0, aspect=4.0 / 3.0, scale_m=0.05),
}


@dataclass
class _FrustumHandle:
    camera_id: int
    handle: viser.CameraFrustumHandle


_MJ_TO_VISER_FLIP_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
"""180° rotation about x. MuJoCo cameras look down -z with +y up (OpenGL);
viser's `add_camera_frustum` opens along +z with -y up. Right-multiplying
the MuJoCo camera quat by this flip aligns frustum direction with view
direction — without it, every frustum points 180° away from the lens."""


def _camera_quat(data: mujoco.MjData, camera_id: int, out: np.ndarray) -> None:
    """Fill `out` with the wxyz quaternion of the camera's world orientation,
    in viser frustum convention. Uses C kernels rather than
    `vtf.SO3.from_matrix` (whose `np.allclose` branch scans dominate the
    profile when called per camera per frame)."""
    mj_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(mj_quat, data.cam_xmat[camera_id])
    mujoco.mju_mulQuat(out, mj_quat, _MJ_TO_VISER_FLIP_QUAT)


def add_frustum_widgets(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: tuple[tuple[str, CameraRole], ...],
) -> list[_FrustumHandle]:
    """Publish a Viser frustum for each declared camera; return handles to update later."""
    out: list[_FrustumHandle] = []
    quat_buf = np.empty(4, dtype=np.float64)
    for name, role in cameras:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise ValueError(f"camera {name!r} not found in compiled model")
        intrinsics = _INTRINSICS[role]
        pos = data.cam_xpos[cam_id]
        _camera_quat(data, cam_id, quat_buf)
        handle = server.scene.add_camera_frustum(
            f"/cameras/{name}",
            fov=np.deg2rad(intrinsics.fov_deg),
            aspect=intrinsics.aspect,
            scale=intrinsics.scale_m,
            color=(0.9, 0.5, 0.2),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            wxyz=(
                float(quat_buf[0]),
                float(quat_buf[1]),
                float(quat_buf[2]),
                float(quat_buf[3]),
            ),
        )
        out.append(_FrustumHandle(camera_id=cam_id, handle=handle))
    return out


def update_frustum_widgets(
    server: viser.ViserServer,
    data: mujoco.MjData,
    handles: list[_FrustumHandle],
) -> None:
    """Push world pose of each frustum widget using the same fast path as
    `viser_render.update_viser`: cache `buffer.push`, bypass the property
    setter's `np.allclose` diff, wrap the whole loop in `server.atomic()`."""
    if not handles:
        return

    wsi = handles[0].handle._impl.api._websock_interface
    buffer_push = wsi.get_message_buffer().push

    set_pos_msg = SetPositionMessage
    set_ori_msg = SetOrientationMessage
    mat2quat = mujoco.mju_mat2Quat
    cam_xpos = data.cam_xpos
    cam_xmat = data.cam_xmat

    mul_quat = mujoco.mju_mulQuat
    mj_quat_buf = np.empty(4, dtype=np.float64)
    quat_buf = np.empty(4, dtype=np.float64)
    with server.atomic():
        for fh in handles:
            cid = fh.camera_id
            pos = cam_xpos[cid]
            mat2quat(mj_quat_buf, cam_xmat[cid])
            mul_quat(quat_buf, mj_quat_buf, _MJ_TO_VISER_FLIP_QUAT)
            pos_tuple = (float(pos[0]), float(pos[1]), float(pos[2]))
            wxyz_tuple = (
                float(quat_buf[0]),
                float(quat_buf[1]),
                float(quat_buf[2]),
                float(quat_buf[3]),
            )
            impl = fh.handle._impl
            impl.position[:] = pos_tuple
            impl.wxyz[:] = wxyz_tuple
            name = impl.name
            buffer_push(set_pos_msg(name, pos_tuple))
            buffer_push(set_ori_msg(name, wxyz_tuple))
