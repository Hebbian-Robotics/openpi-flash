"""Viser camera frustum widgets (and a place for future EGL image capture).

A scene that wants cameras declares::

    CAMERAS: tuple[tuple[str, CameraRole], ...] = (
        ("top_cam", CameraRole.TOP),
        ("left_wrist", CameraRole.WRIST),
        ("right_wrist", CameraRole.WRIST),
    )

Each entry pairs the MuJoCo camera name with the role that picks its
intrinsics (FOV / aspect / widget scale). Role is an explicit `StrEnum`
rather than a substring inferred from the camera name — the earlier
substring logic would happily match `CameraRole.TOP` against a camera
named `topology_cam`, and the iteration order decided ties silently.

The runner calls `add_frustum_widgets(...)` once at startup and
`update_frustum_widgets(...)` each frame so the frustum tracks the
camera's world pose (important for cameras attached to moving bodies
like the lift carriage or wrist links).

Per-frame updates take the same fast path as `viser_render.update_viser`:
matrix→quaternion via `mujoco.mju_mat2Quat` (not `vtf.SO3.from_matrix`),
direct `buffer.push(...)` instead of the property setter, and one
`server.atomic()` block around the loop. Camera frustums are few (a
handful per scene) so the absolute CPU savings are modest, but they kept
the old hot-path shapes alive in the profile — this closes that gap.

Future extension: `CameraRenderer` (sketched at the bottom, not implemented)
will use `mujoco.Renderer` with EGL to produce actual images and push them
as Viser image widgets. Keeping that concern here means the scene modules
don't need to change — they just declare camera names.
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
Viser's `add_camera_frustum` opens the frustum along +z with -y up. Applying
this rotation to the MuJoCo camera's world-frame quaternion (right-multiply)
flips the -z/+y axes to +z/-y so the frustum points the way the camera looks.
Without it, every frustum appears 180° about the camera's side axis — which
is the symptom the user saw ("cameras facing the wrong way")."""


def _camera_quat(data: mujoco.MjData, camera_id: int, out: np.ndarray) -> None:
    """Fill `out` (shape (4,), float64) with the wxyz quaternion of the named
    MuJoCo camera's world-frame orientation, converted to the viser frustum
    convention. Uses `mujoco.mju_mat2Quat` + `mujoco.mju_mulQuat` (C kernels)
    rather than `vtf.SO3.from_matrix`, which runs multiple `np.allclose` scans
    to pick among four quaternion branches."""
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

    wsi = handles[0].handle._impl.api._websock_interface  # noqa: SLF001
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
            impl = fh.handle._impl  # noqa: SLF001 — viser-internal fast path
            impl.position[:] = pos_tuple
            impl.wxyz[:] = wxyz_tuple
            name = impl.name
            buffer_push(set_pos_msg(name, pos_tuple))
            buffer_push(set_ori_msg(name, wxyz_tuple))


# ---------------------------------------------------------------------------
# Future: EGL image capture.
#
# class CameraRenderer:
#     def __init__(self, model: mujoco.MjModel, camera_name: str,
#                  width: int = 320, height: int = 240) -> None:
#         self.renderer = mujoco.Renderer(model, height=height, width=width)
#         self.camera_id = mujoco.mj_name2id(
#             model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
#
#     def render(self, data: mujoco.MjData) -> Float[np.ndarray, "h w 3"]:
#         self.renderer.update_scene(data, camera=self.camera_id)
#         return self.renderer.render()  # HxWx3 uint8
#
# The runner would build one per camera in CAMERAS, throttle to ~10 Hz, and
# push via `server.gui.add_image(...)` or an equivalent viser widget. Adding
# this requires no scene changes — only cameras.py + runner.py wiring.
