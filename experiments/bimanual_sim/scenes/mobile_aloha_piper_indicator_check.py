"""Indicator-check scene — Mobile ALOHA + bimanual Piper.

The robot drives into the aisle between two rows of 5 racks (each filled
with 21 x 2U servers, each with a green indicator light), turns left to
face the 4th rack, both arms reach toward the alert server (whose light
starts red), holds for 1 s while the light flips green via runtime RGBA
update, then arms retract. No manipulation, no welds, no grippables.

Task / phase decomposition uses the same `JointSetStatic` invariants as
the live server-swap scene: each phase moves either base OR arms, never
both at once. The one wrinkle is `ALIGN_TO_TARGET`: it has two base-only
sub-steps (yaw at chassis_Y=0 where the swept-corner radius fits, then
translate +Y to the click pose). Both sub-steps are still base-only, so
the per-phase invariant `_ARMS_STATIC` holds for the entire phase.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import mujoco
import numpy as np
from dm_control import mjcf

from arm_handles import ArmHandles, ArmSide, arm_joint_suffixes
from cameras import CameraRole
from paths import D405_MESH_STL, D435I_XML
from robots.mobile_aloha import (
    BASE_X_JOINT_NAME,
    BASE_Y_JOINT_NAME,
    BASE_YAW_JOINT_NAME,
    LEFT_ARM_MOUNT_SITE,
    RIGHT_ARM_MOUNT_SITE,
    load_mobile_aloha,
)
from robots.piper import load_piper
from scene_base import (
    GripperState,
    JointSetStatic,
    PhaseContract,
    PhaseState,
    QaccSentinel,
    QuatWxyz,
    Step,
    TaskPhase,
)
from scene_check import (
    AttachmentConstraint,
    CameraInvariant,
    FixedCameraInvariant,
)
from scenes.mobile_aloha_piper_indicator_check_layout import (
    ALERT_LIGHT_GEOM_NAME,
    BASE_HOME_POSE,
    HOME_ARM_Q_BY_SIDE,
    LAYOUT,
)

NAME = "mobile_aloha_piper_indicator_check"
ROBOT_KIND = "piper"
IK_LOCKED_JOINT_NAMES: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)
GRIPPABLES: tuple[str, ...] = ()
N_CUBES = 0
ATTACHMENTS: tuple[AttachmentConstraint, ...] = ()


class IndicatorAux(StrEnum):
    """Scene-owned planar-base actuators addressable via Step.aux_ctrl."""

    BASE_X = BASE_X_JOINT_NAME
    BASE_Y = BASE_Y_JOINT_NAME
    BASE_YAW = BASE_YAW_JOINT_NAME


class RotationAxis(StrEnum):
    """Finite set of axes accepted by `_axis_angle_quat`."""

    X = "x"
    Y = "y"
    Z = "z"


AUX_ACTUATOR_NAMES: tuple[str, ...] = tuple(m.value for m in IndicatorAux)


def _axis_angle_quat(axis: RotationAxis, angle_rad: float) -> QuatWxyz:
    """Return a unit quaternion for one axis-angle rotation."""
    half_angle_rad = angle_rad / 2.0
    cos_half_angle = math.cos(half_angle_rad)
    sin_half_angle = math.sin(half_angle_rad)
    match axis:
        case RotationAxis.X:
            values = (cos_half_angle, sin_half_angle, 0.0, 0.0)
        case RotationAxis.Y:
            values = (cos_half_angle, 0.0, sin_half_angle, 0.0)
        case RotationAxis.Z:
            values = (cos_half_angle, 0.0, 0.0, sin_half_angle)
    return np.asarray(values, dtype=float)


def _compose_quat(left: QuatWxyz, right: QuatWxyz) -> QuatWxyz:
    """Hamilton product `left * right` for MuJoCo `(w, x, y, z)` quats."""
    left_w, left_x, left_y, left_z = np.asarray(left, dtype=float)
    right_w, right_x, right_y, right_z = np.asarray(right, dtype=float)
    return np.asarray(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ),
        dtype=float,
    )


def _mjcf_quat(quat: QuatWxyz) -> list[float]:
    """Convert a typed numpy quat into the plain list dm_control.mjcf expects."""
    return [float(value) for value in quat]


def base_aux_targets(*, x: float, y: float, yaw: float) -> tuple[tuple[str, float], ...]:
    """Canonical phase-boundary representation for planar base targets."""
    return (
        (IndicatorAux.BASE_X, x),
        (IndicatorAux.BASE_Y, y),
        (IndicatorAux.BASE_YAW, yaw),
    )


CAMERAS: tuple[tuple[str, CameraRole], ...] = (
    ("forward_cam", CameraRole.TOP),
    ("left/wrist_d405_cam", CameraRole.WRIST),
    ("right/wrist_d405_cam", CameraRole.WRIST),
)

# `forward_cam` is rigidly attached to base_link with optical axis along
# chassis +x, so the view always frames whatever the robot is facing — no
# yaw-driven swing. The d435i mesh sits beside the camera, oriented to match.
# Wrist cameras live on each Piper's link6 and look along the gripper's +z
# axis (the tool-use direction), giving two POV-style views that pair well
# with the static top view for video rendering.
_ALERT_RACK_BODY_NAME = f"rack_{LAYOUT.alert.row}_r{LAYOUT.alert.rack_index}"

CAMERA_INVARIANTS: tuple[CameraInvariant, ...] = (
    FixedCameraInvariant(name="forward_cam", parent_body="base_link"),
    FixedCameraInvariant(name="left/wrist_d405_cam", parent_body="left/link6"),
    FixedCameraInvariant(name="right/wrist_d405_cam", parent_body="right/link6"),
)

# All static geometry sits inside one `data_center` body (racks + servers +
# lights), so scene_check's same-body skip handles within-body overlaps.
# No cross-body overlaps are expected — the chassis stays clear of rack
# panels at every phase boundary by construction (see layout invariants).
ALLOWED_STATIC_OVERLAPS: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Camera + visual tunables
# ---------------------------------------------------------------------------
# Lifted out of `build_spec` so changes here don't drift away from the
# prose in surrounding comments. Each constant documents *what* it
# represents; the value can change without invalidating the docstring.

# Forward (top) D435i camera, mounted on base_link.
_TOP_CAM_HEIGHT_M = 1.40
"""Height of the top-cam mount in chassis-body z. The shared
`load_mobile_aloha` chassis was authored for z=1.6; lowering to 1.4
brings the workspace + arms inside the frame at the click pose."""

_TOP_CAM_X_M = 0.076
"""Forward offset (chassis +x) of the top cam mount, matching the camera
pole position on the Mobile-ALOHA chassis."""

_TOP_CAM_TILT_DOWN_DEG = 25.0
"""Downward pitch of the top cam, in degrees. Combined with the lower
mount and a 42° vertical FOV (real D435i), this centres the workspace
at the click pose — the look axis hits world (5.10, ~1.06, 1.0) within
1 cm of the alert."""

_TOP_CAM_FOVY_DEG = 42.0
"""Real Intel D435i color-stream vertical FOV. Horizontal is 69° but
MuJoCo's `<camera fovy>` takes the vertical."""

# Camera-stand override: load_mobile_aloha's pole is built for z=1.6, so
# we resize the existing `top_cam_stand` cylinder to end at the new
# height. The base z (1.0) sits on the chassis top.
_CAM_STAND_BASE_Z = 1.0

# Wrist D405 cameras, one per Piper, mounted on link6 (the wrist roll
# body). Offsets are in link6's local frame:
#   +z is the gripper / tool axis (fingers attach at z=+0.135).
#   -x is the dorsal "top" of the wrist at home pose, where a real
#       D405 mounts on ALOHA-style Piper rigs.
# Verified by interactive sweep at sim t=22 (gripper touching alert).
_WRIST_MESH_OFFSET_X_M = -0.05
"""Perpendicular offset of the visible D405 mesh body (link6 -x)."""

_WRIST_MESH_OFFSET_Z_M = 0.04
"""Forward offset of the mesh along the gripper axis. Sits ahead of
link6 origin so the mesh visually reads as "mounted on top of the
wrist" rather than inside it."""

_WRIST_CAM_OFFSET_X_M = -0.14
"""Perpendicular offset of the wrist camera (link6 -x). Pulled well
clear of the wrist housing so the framing isn't dominated by link6's
own geometry; chosen via interactive sweep at sim t=22."""

_WRIST_CAM_OFFSET_Z_M = -0.03
"""Backward offset of the cam along the gripper axis. Negative pulls
the cam behind link6 origin so the bezel doesn't fill the frame at
the touch moment."""

_WRIST_CAM_TILT_TOWARD_GRIPPER_DEG = -20.0
"""Tilt about the cam's local +x axis. Negative tips the look vector
toward +x_link6 (toward the gripper tip) — compensates for pulling
the cam outward in -x. Sign verified empirically: positive tilts away."""

_WRIST_CAM_FOVY_DEG = 58.0
"""Real Intel D405 color-stream vertical FOV. Horizontal is 87°."""

_WRIST_MESH_RGBA: tuple[float, float, float, float] = (0.12, 0.12, 0.14, 1.0)
"""Dark grey for the d405 visual mesh (matches a real RealSense body)."""

_FINGER_RGBA: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
"""Opaque grey override on link7 / link8 visual meshes. Replaces the
menagerie's gray_mat material whose specular component exposed
duplicate-shell artefacts on opposite-diagonal corners under headless
EGL render."""


# ---------------------------------------------------------------------------
# Phase invariants
# ---------------------------------------------------------------------------

_QACC_SENTINEL = QaccSentinel(max_increase=0)
_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side.value}{suffix}" for side in ARM_PREFIXES for suffix in arm_joint_suffixes(ROBOT_KIND)
)
_ARMS_STATIC = JointSetStatic(joint_names=_ARM_JOINT_NAMES, label="arms")
_BASE_STATIC = JointSetStatic(joint_names=IK_LOCKED_JOINT_NAMES, label="base")


# ---------------------------------------------------------------------------
# Phase contracts
# ---------------------------------------------------------------------------

# Base waypoints. `_CLICK_X` / `_CLICK_Y` come from the layout — chassis
# nose flush with the alert row's rack front when yawed +π/2. The yaw
# itself happens at chassis_Y=0 (TRAVERSE_END) so the swept-corner radius
# (~1.024 m) fits between the two rows; the +Y nudge to the click pose
# is a translation-only motion at fixed yaw, which slides without sweeping.
_CLICK_X, _CLICK_Y = LAYOUT.click_chassis_xy
_CLICK_YAW = math.pi / 2.0 if LAYOUT.alert.row == "left" else -math.pi / 2.0

_BASE_ORIGIN = base_aux_targets(x=0.0, y=0.0, yaw=0.0)
_BASE_AT_TRAVERSE_END = base_aux_targets(x=_CLICK_X, y=0.0, yaw=0.0)
_BASE_AT_CLICK = base_aux_targets(x=_CLICK_X, y=_CLICK_Y, yaw=_CLICK_YAW)


PHASE_CONTRACTS: tuple[PhaseContract, ...] = (
    PhaseContract(
        phase=TaskPhase.SETUP,
        starts=PhaseState(
            description="Robot at world origin, all indicator lights set.",
            base_aux=_BASE_ORIGIN,
        ),
        ends=PhaseState(
            description="Hold complete; ready to drive into aisle.",
            base_aux=_BASE_ORIGIN,
        ),
        invariants=(_QACC_SENTINEL, _ARMS_STATIC, _BASE_STATIC),
    ),
    # Base-only: drive +X to align with the alert rack column. Aisle Y stays
    # at 0 — the +Y nudge to the click pose happens after the in-place yaw.
    PhaseContract(
        legal_predecessors=(TaskPhase.SETUP,),
        phase=TaskPhase.TRAVERSE_INTO_AISLE,
        starts=PhaseState(
            description="Robot at origin facing +X.",
            base_aux=_BASE_ORIGIN,
        ),
        ends=PhaseState(
            description="Robot in aisle aligned with alert rack column; still facing +X.",
            base_aux=_BASE_AT_TRAVERSE_END,
        ),
        invariants=(_QACC_SENTINEL, _ARMS_STATIC),
    ),
    # Base-only with two sub-steps: yaw in place at chassis_Y=0 (where the
    # swept-corner radius clears both rows), then translate +Y to bring the
    # chassis nose flush with the alert row's rack front.
    PhaseContract(
        legal_predecessors=(TaskPhase.TRAVERSE_INTO_AISLE,),
        phase=TaskPhase.ALIGN_TO_TARGET,
        starts=PhaseState(
            description="Robot in aisle, facing +X.",
            base_aux=_BASE_AT_TRAVERSE_END,
        ),
        ends=PhaseState(
            description="Robot facing the alert row, chassis nose at rack front.",
            base_aux=_BASE_AT_CLICK,
        ),
        invariants=(_QACC_SENTINEL, _ARMS_STATIC),
    ),
    # Arm-only: both arms extend from home toward the alert server's bezel.
    PhaseContract(
        legal_predecessors=(TaskPhase.ALIGN_TO_TARGET,),
        phase=TaskPhase.REACH_TO_SERVER,
        starts=PhaseState(
            description="Arms at home, robot at click pose.",
            base_aux=_BASE_AT_CLICK,
        ),
        ends=PhaseState(
            description="Arms extended near the alert server's bezel.",
            base_aux=_BASE_AT_CLICK,
        ),
        invariants=(_QACC_SENTINEL, _BASE_STATIC),
    ),
    # Hold + colour flip: arms freeze in the reach pose for 1 s; the alert
    # indicator's RGBA flips green on this phase's first tick via
    # `Step.set_geom_rgba`. Both arms and base must stay still.
    PhaseContract(
        legal_predecessors=(TaskPhase.REACH_TO_SERVER,),
        phase=TaskPhase.WAIT_AT_SERVER,
        starts=PhaseState(
            description="Both arms at the alert server; light still red.",
            base_aux=_BASE_AT_CLICK,
        ),
        ends=PhaseState(
            description="Light flipped green; demo logically complete.",
            base_aux=_BASE_AT_CLICK,
        ),
        invariants=(_QACC_SENTINEL, _ARMS_STATIC, _BASE_STATIC),
    ),
    # Arm-only: arms back to home pose.
    PhaseContract(
        legal_predecessors=(TaskPhase.WAIT_AT_SERVER,),
        phase=TaskPhase.RETRACT,
        starts=PhaseState(
            description="Arms extended at the alert server.",
            base_aux=_BASE_AT_CLICK,
        ),
        ends=PhaseState(
            description="Arms at home; demo complete.",
            base_aux=_BASE_AT_CLICK,
        ),
        invariants=(_QACC_SENTINEL, _BASE_STATIC),
    ),
)


# ---------------------------------------------------------------------------
# Spec construction
# ---------------------------------------------------------------------------


def _add_data_center(root: mjcf.RootElement) -> None:
    """Build the static data-center body holding every rack, server, and
    indicator light. All geoms live on a single body so scene_check's
    same-body skip handles intra-body overlaps automatically — no entries
    needed in `ALLOWED_STATIC_OVERLAPS` for the 200+ adjacent server pairs."""
    rack_cfg = LAYOUT.rack
    server_cfg = LAYOUT.server
    light_cfg = LAYOUT.light
    aisle_cfg = LAYOUT.aisle
    servers_cfg = LAYOUT.servers

    panel_rgba = [0.10, 0.10, 0.13, 1.0]
    rhx, rhy, rhz = rack_cfg.half
    wt = rack_cfg.wall_thickness

    # Rack bodies — each rack a separate body so the top-camera TARGETBODY
    # can name a specific rack. Servers + lights all collapse into one
    # parent `data_center` body for overlap-check efficiency.
    for row in ("left", "right"):
        row_y = LAYOUT.row_centre_y(row)
        # Right-row racks rotated 180° about Z so the rack's "rear" panel
        # ends up at the row's outer edge (away from the aisle), matching
        # the left row's geometry by symmetry.
        rack_quat = [0.0, 0.0, 0.0, 1.0] if row == "right" else [1.0, 0.0, 0.0, 0.0]
        for i, x in enumerate(aisle_cfg.rack_x_centres):
            rack_body = root.worldbody.add(
                "body",
                name=f"rack_{row}_r{i}",
                pos=[x, row_y, aisle_cfg.rack_centre_z],
                quat=rack_quat,
            )
            # Rear panel — at body +Y face, away from the aisle.
            rack_body.add(
                "geom",
                dclass="visual",
                type="box",
                pos=[0.0, +rhy - wt, 0.0],
                size=[rhx, wt, rhz],
                rgba=panel_rgba,
                name=f"rack_{row}_r{i}_rear",
            )
            # Side panels — at body ±X faces, full rack depth/height.
            rack_body.add(
                "geom",
                dclass="visual",
                type="box",
                pos=[-rhx + wt, 0.0, 0.0],
                size=[wt, rhy, rhz],
                rgba=panel_rgba,
                name=f"rack_{row}_r{i}_side_neg_x",
            )
            rack_body.add(
                "geom",
                dclass="visual",
                type="box",
                pos=[+rhx - wt, 0.0, 0.0],
                size=[wt, rhy, rhz],
                rgba=panel_rgba,
                name=f"rack_{row}_r{i}_side_pos_x",
            )
            # Top + bottom panels.
            rack_body.add(
                "geom",
                dclass="visual",
                type="box",
                pos=[0.0, 0.0, +rhz - wt],
                size=[rhx, rhy, wt],
                rgba=panel_rgba,
                name=f"rack_{row}_r{i}_top",
            )
            rack_body.add(
                "geom",
                dclass="visual",
                type="box",
                pos=[0.0, 0.0, -rhz + wt],
                size=[rhx, rhy, wt],
                rgba=panel_rgba,
                name=f"rack_{row}_r{i}_bottom",
            )

    # All 210 servers + 210 lights inside a single body so scene_check's
    # same-body skip absorbs the within-body adjacency overlaps.
    dc = root.worldbody.add("body", name="data_center", pos=[0.0, 0.0, 0.0])
    # Cylinder default axis is local +Z; rotate +90° about X to put the
    # axis along world +Y so each light pokes out of the bezel face.
    light_quat = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]

    # Alternate two shades per slot so adjacent servers read as distinct
    # in the headless render. MuJoCo's GL fills two co-planar same-rgba
    # boxes as a single continuous surface (no edge highlighting), so
    # 21 stacked dark-grey boxes would blend into one undifferentiated
    # column. Striping odd slots a few percent lighter creates a visible
    # horizontal break per U without changing geometry. Viser's WebGL
    # shader did this implicitly; the offline renderer doesn't.
    rgba_chassis_dark = list(server_cfg.rgba_chassis)
    rgba_chassis_light = [min(1.0, c + 0.06) for c in server_cfg.rgba_chassis[:3]] + [
        server_cfg.rgba_chassis[3]
    ]
    for row in ("left", "right"):
        for rack_index in range(len(aisle_cfg.rack_x_centres)):
            for slot_index in range(servers_cfg.n_per_rack):
                server_x = LAYOUT.rack_centre_x(rack_index)
                server_y = LAYOUT.server_centre_y(row)
                server_z = LAYOUT.server_centre_z(slot_index)
                rgba_chassis = rgba_chassis_dark if slot_index % 2 == 0 else rgba_chassis_light
                dc.add(
                    "geom",
                    dclass="visual",
                    type="box",
                    pos=[server_x, server_y, server_z],
                    size=list(server_cfg.half),
                    rgba=rgba_chassis,
                    name=f"server_{row}_r{rack_index}_s{slot_index:02d}",
                )
                light_pos = LAYOUT.light_world_pos(row, rack_index, slot_index)
                is_alert = (
                    row == LAYOUT.alert.row
                    and rack_index == LAYOUT.alert.rack_index
                    and slot_index == LAYOUT.alert.slot_index
                )
                rgba = light_cfg.rgba_red if is_alert else light_cfg.rgba_green
                dc.add(
                    "geom",
                    dclass="visual",
                    type="cylinder",
                    pos=light_pos.tolist(),
                    quat=light_quat,
                    size=[light_cfg.radius, light_cfg.half_length],
                    rgba=list(rgba),
                    name=LAYOUT.light_geom_name(row, rack_index, slot_index),
                )


def build_spec() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Assemble the scene with dm_control.mjcf, compile via mujoco, return
    plain `(model, data)` so downstream runtime code uses the standard API."""
    root = load_mobile_aloha()

    root.option.integrator = "implicitfast"
    root.option.cone = "elliptic"
    root.option.impratio = 10.0
    root.option.timestep = 0.002
    # Disable contacts: every body is held by direct qpos writes (puppet
    # mode). Contact detection just adds work and risks spurious QACC.
    root.option.flag.contact = "disable"
    root.option.gravity = [0.0, 0.0, 0.0]

    # `global` is a Python keyword — access via getattr.
    visual_global = getattr(root.visual, "global")
    visual_global.offwidth = 1920
    visual_global.offheight = 1080
    # Slight bump to the camera near-clip plane (default ~0.005 m).
    # Combined with pulling the wrist cam outward on -x, this trims
    # the partial-clipping artifact on the gripper plate corners
    # without entering link6's interior cavity.
    root.visual.map.znear = 0.010
    # 8x MSAA on the offscreen framebuffer (default is 4). Sharpens
    # rack panels, gripper plates, and per-server boundaries in the
    # headless render at the cost of a bit more GPU work per frame.
    # Lives on `<visual><quality offsamples=…/>`, NOT on `<global>`.
    root.visual.quality.offsamples = 8

    visual = root.default.add("default", dclass="visual")
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.mass = 0.0

    wb = root.worldbody

    # Lighting — toned down vs. the live server-swap scene. With ambient
    # 0.4 + headlight 0.6 + key 0.5 + fill 0.3 stacking on a 0.78-grey
    # floor, every additive contribution clipped to white when the cam
    # framed the aisle floor head-on (e.g. forward_cam at chassis home).
    # Cutting ambient to 0.25, headlight to 0.4, dropping the fill, and
    # darkening the floor rgba keeps the same lit feel without burning
    # out the highlights at headless render time.
    root.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    root.visual.headlight.ambient = [0.25, 0.25, 0.25]
    root.visual.headlight.specular = [0.0, 0.0, 0.0]
    wb.add(
        "light",
        name="key_directional",
        pos=[3.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        directional="true",
        diffuse=[0.4, 0.4, 0.4],
        specular=[0.05, 0.05, 0.05],
    )

    wb.add(
        "geom",
        type="plane",
        size=[8.0, 4.0, 0.1],
        pos=[3.0, 0.0, 0.0],
        rgba=[0.45, 0.45, 0.48, 1.0],
    )

    # Racks + servers + indicator lights.
    _add_data_center(root)

    # Shared D405 mesh asset, referenced by both pipers' wrist visuals.
    # Declared on the parent root once (not per-arm) so the compiled model
    # has a single mesh asset instead of two redundant copies — mjcf
    # resolves the cross-namescope reference by Element identity.
    d405_mesh = root.asset.add("mesh", name="d405", file=str(D405_MESH_STL))

    # Bimanual Piper arms on Mobile ALOHA's front mount sites. Each arm
    # carries a TCP site, a D405 visual mesh, and a wrist camera on link6
    # — the wrist cam looks along link6 +z (the gripper / tool axis) so
    # the view always frames whatever the arm is reaching for.
    arm_mount_sites = {
        ArmSide.LEFT: LEFT_ARM_MOUNT_SITE,
        ArmSide.RIGHT: RIGHT_ARM_MOUNT_SITE,
    }
    # See "Camera + visual tunables" at module top for all the values
    # below. link6 itself rotates with joint6, so both mesh and cam
    # follow the wrist roll — whichever side they start on stays their
    # side after every rotation.
    wrist_mesh_pos = [_WRIST_MESH_OFFSET_X_M, 0.0, _WRIST_MESH_OFFSET_Z_M]
    wrist_cam_pos = [_WRIST_CAM_OFFSET_X_M, 0.0, _WRIST_CAM_OFFSET_Z_M]
    # Wrist-cam orientation: a base layout (-90° about z_link6) ∘
    # (180° about x_link6) that maps -z_cam onto +z_link6 (look down
    # the gripper axis) and reads upright at link6's home rotation,
    # composed with a downward tilt about cam-local +x to tip the look
    # vector toward the gripper tip.
    sqrt_half = 0.5**0.5
    wrist_base_quat: QuatWxyz = np.asarray((0.0, sqrt_half, -sqrt_half, 0.0), dtype=float)
    wrist_cam_quat = _mjcf_quat(
        _compose_quat(
            wrist_base_quat,
            _axis_angle_quat(RotationAxis.X, math.radians(_WRIST_CAM_TILT_TOWARD_GRIPPER_DEG)),
        )
    )
    for side in ARM_PREFIXES:
        piper = load_piper(side)
        link6 = piper.find("body", "link6")
        if link6 is None:
            raise RuntimeError(f"piper {side!r} missing 'link6' after load")
        # Hide the gripper-finger collision boxes from the offline
        # render. The menagerie Piper ships each finger (link7, link8)
        # with two `class="collision"` boxes carrying `rgba="1 0 0 .2"`
        # / `rgba="0 0 1 .2"` — translucent helpers for contact pads
        # that read as "see-through chunks of the gripper" in the
        # rendered video. Our scene disables contacts entirely
        # (`flag.contact = "disable"`), so these geoms exist purely as
        # visual clutter. Move them to geom group 3, which the render
        # script's default `MjvOption.geomgroup = [1,1,1,0,0,0]` skips,
        # while leaving viser's live view untouched (viser_render
        # ignores geomgroup).
        for geom in piper.find_all("geom"):
            if geom.dclass is not None and geom.dclass.dclass == "collision":
                geom.group = 3
        # Force opaque rgba (alpha=1) on the finger visual meshes,
        # overriding the default `material="gray_mat"` reference. The
        # menagerie material has `rgba="0.59 0.59 0.59 1"` (already
        # alpha 1), but its reflectance/specular components combined
        # with the headless EGL renderer expose duplicate-shell
        # artefacts in `link7.stl` / `link8.stl` — opposite-diagonal
        # corner panels render with apparent transparency. Replacing
        # the material with a flat rgba on these specific geoms gives
        # uniform shading and hides the artefact.
        for finger_body_name in ("link7", "link8"):
            finger_body = piper.find("body", finger_body_name)
            if finger_body is None:
                continue
            for geom in finger_body.find_all("geom"):
                if geom.dclass is not None and geom.dclass.dclass == "visual":
                    geom.material = None
                    geom.rgba = list(_FINGER_RGBA)
        # TCP site at 14 cm forward of link6 along the gripper axis. IK /
        # weld code addresses this by `f"{side}tcp"` — inside the namescope
        # the local name is `tcp`.
        link6.add(
            "site",
            name="tcp",
            pos=[0.0, 0.0, 0.14],
            size=[0.006, 0.006, 0.006],
        )
        # D405 visual mesh — sits forward of the cam so the body reads
        # as a real D405 mounted at the wrist top. Mesh quat matches
        # the cam quat (no extra roll: with q_reach[…][5] = 0 the wrist
        # stays level when reaching, so the cam doesn't need the 180°
        # roll-compensation that the earlier asymmetric pose required).
        link6.add(
            "geom",
            dclass="visual",
            type="mesh",
            mesh=d405_mesh,
            pos=wrist_mesh_pos,
            quat=wrist_cam_quat,
            rgba=list(_WRIST_MESH_RGBA),
        )
        link6.add(
            "camera",
            name="wrist_d405_cam",
            pos=wrist_cam_pos,
            quat=wrist_cam_quat,
            mode="fixed",
            fovy=_WRIST_CAM_FOVY_DEG,
        )
        mount_site_name = arm_mount_sites[side]
        mount_site = root.find("site", mount_site_name)
        if mount_site is None:
            raise RuntimeError(
                f"mobile-aloha mount site {mount_site_name!r} not found — "
                "did robots/mobile_aloha.py change site names?"
            )
        mount_site.attach(piper)

    # Top-camera + d435i mesh, both attached directly to base_link via a
    # scene-local site. We deliberately bypass the shared `top_cam_mount`
    # site (whose +90°-x quat was authored for the legacy live scene's
    # camera convention) so this scene can specify the orientation
    # explicitly without touching robots/mobile_aloha.py.
    base_link_body = root.find("body", "base_link")
    if base_link_body is None:
        raise RuntimeError("Mobile ALOHA chassis body 'base_link' not found")
    # Shorten the load_mobile_aloha camera stand to end at this scene's
    # `_TOP_CAM_HEIGHT_M`, so the d435i body sits flush on the post.
    cam_stand = root.find("geom", "top_cam_stand")
    if cam_stand is not None:
        stand_half_height = (_TOP_CAM_HEIGHT_M - _CAM_STAND_BASE_Z) / 2.0
        stand_centre_z = (_TOP_CAM_HEIGHT_M + _CAM_STAND_BASE_Z) / 2.0
        cam_stand.size = [0.020, stand_half_height]
        cam_stand.pos = [_TOP_CAM_X_M, 0.0, stand_centre_z]
    # Top-cam orientation. See the `_TOP_CAM_*` constants for tunables.
    _tilt_rad = math.radians(_TOP_CAM_TILT_DOWN_DEG)
    _cos_tilt, _sin_tilt = math.cos(_tilt_rad), math.sin(_tilt_rad)
    # Site quat composes (in order):
    #   q_yaw180   - 180° about base_link z, flips the d435i mesh body so
    #                its lens face points forward (+x_body) instead of
    #                backward like the bare attach orientation does
    #   q_y_neg25  - tilts the body -25° about base_link y to match the
    #                actual camera pitch (so visual + view direction agree)
    #   q_base     - the (+90° x) ∘ (-90° z) layout that puts the d435i
    #                body horizontal with its long axis along world -y
    # Pre-composed once offline (Hamilton product, unit-checked) so MJCF
    # only sees the final 4-vector. The camera itself is a separate
    # `<camera>` element with explicit xyaxes — flipping the mesh here
    # doesn't affect what the camera renders.
    top_cam_base_quat: QuatWxyz = np.asarray((0.5, 0.5, -0.5, -0.5), dtype=float)
    top_cam_mount_quat = _mjcf_quat(
        _compose_quat(
            _axis_angle_quat(RotationAxis.Z, math.pi),
            _compose_quat(
                _axis_angle_quat(RotationAxis.Y, -_tilt_rad),
                top_cam_base_quat,
            ),
        )
    )
    indicator_top_cam_mount = base_link_body.add(
        "site",
        name="indicator_top_cam_mount",
        pos=[_TOP_CAM_X_M, 0.0, _TOP_CAM_HEIGHT_M],
        quat=top_cam_mount_quat,
        size=[0.001, 0.001, 0.001],
    )
    top_d435i = mjcf.from_path(str(D435I_XML))
    top_d435i.model = "top"
    indicator_top_cam_mount.attach(top_d435i)

    # `forward_cam` uses xyaxes so the tilt is spelled out in base_link's
    # frame: cam +x = -y_body (camera right, perpendicular to look + up),
    # cam +y = (+sin(tilt), 0, +cos(tilt)) (camera up, tilted forward by
    # `_TOP_CAM_TILT_DOWN_DEG`), -z_cam = (cos(tilt), 0, -sin(tilt))
    # (look direction = forward + tilt downward). The camera yaws with
    # the chassis, so the framing stays the same regardless of base
    # orientation.
    base_link_body.add(
        "camera",
        name="forward_cam",
        pos=[_TOP_CAM_X_M, 0.0, _TOP_CAM_HEIGHT_M],
        xyaxes=[0.0, -1.0, 0.0, _sin_tilt, 0.0, _cos_tilt],
        mode="fixed",
        fovy=_TOP_CAM_FOVY_DEG,
    )

    # Planar-base actuators — same pattern as the live scene.
    root.actuator.add(
        "position",
        name=IndicatorAux.BASE_X,
        joint=IndicatorAux.BASE_X,
        kp=20000.0,
        kv=400.0,
        ctrllimited="true",
        ctrlrange=[-1.0, 7.0],  # extended +X range — robot drives ~5 m into the aisle
    )
    root.actuator.add(
        "position",
        name=IndicatorAux.BASE_Y,
        joint=IndicatorAux.BASE_Y,
        kp=20000.0,
        kv=400.0,
        ctrllimited="true",
        ctrlrange=[-1.0, 1.0],
    )
    root.actuator.add(
        "position",
        name=IndicatorAux.BASE_YAW,
        joint=IndicatorAux.BASE_YAW,
        kp=8000.0,
        kv=200.0,
        ctrllimited="true",
        ctrlrange=[-3.5, 3.5],
    )

    # No equality constraints (no welds, no grasps).

    xml_str = root.to_xml_string()
    assets = dict(root.get_assets())
    model = mujoco.MjModel.from_xml_string(xml_str, assets)
    data = mujoco.MjData(model)
    return model, data


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def apply_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
    *,
    start_phase: TaskPhase | None = None,
) -> None:
    """Reset to scene home (arms at `HOME_ARM_Q_BY_SIDE`, base at origin),
    or seed the chassis pose from a specific phase's contract `starts.base_aux`
    when `start_phase` is set. Arms always reset to home — the user is
    expected to teleop them into the desired pose for whatever phase
    they're authoring.
    """
    del cube_body_ids
    base_pose = BASE_HOME_POSE
    if start_phase is not None and start_phase is not TaskPhase.SETUP:
        contract = next((c for c in PHASE_CONTRACTS if c.phase is start_phase), None)
        if contract is None:
            print(
                f"[apply_initial_state] no contract for {start_phase.value!r}; "
                "booting at scene home."
            )
        else:
            base_aux_dict = {str(name): float(value) for name, value in contract.starts.base_aux}
            base_pose = (
                base_aux_dict.get(BASE_X_JOINT_NAME, 0.0),
                base_aux_dict.get(BASE_Y_JOINT_NAME, 0.0),
                base_aux_dict.get(BASE_YAW_JOINT_NAME, 0.0),
            )

    mujoco.mj_resetData(model, data)
    for arm in arms.values():
        home_q = HOME_ARM_Q_BY_SIDE[arm.side]
        for i, idx in enumerate(arm.arm_qpos_idx):
            data.qpos[idx] = home_q[i]
        data.ctrl[arm.act_arm_ids] = home_q
        # Piper gripper open at home — no payload, so visual default of
        # open jaws reads as "ready to interact".
        data.ctrl[arm.act_gripper_id] = arm.gripper_open
        # Mirror the gripper open value into the tendon-coupled finger
        # slides (Piper has two finger qpos that mirror each other).
        if arm.robot_kind == "piper":
            data.qpos[arm.qpos_idx[6]] = arm.gripper_open
            data.qpos[arm.qpos_idx[7]] = -arm.gripper_open
            data.qvel[arm.dof_idx[6]] = 0.0
            data.qvel[arm.dof_idx[7]] = 0.0
    for jname, value in zip(
        (IndicatorAux.BASE_X, IndicatorAux.BASE_Y, IndicatorAux.BASE_YAW),
        base_pose,
        strict=True,
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0.0
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid >= 0:
            data.ctrl[aid] = value

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Task plan
# ---------------------------------------------------------------------------


def make_task_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
) -> dict[ArmSide, list[Step]]:
    """Scripted indicator-check choreography: drive in → align → reach →
    wait + flip → retract."""
    scripts: dict[ArmSide, list[Step]] = {side: [] for side in ARM_PREFIXES}

    def push_both(
        label: str,
        duration: float,
        phase: TaskPhase,
        aux: tuple[tuple[str, float], ...] | dict[str, float] | None = None,
        *,
        gripper: GripperState = "open",
        set_geom_rgba: tuple[tuple[str, tuple[float, float, float, float]], ...] = (),
    ) -> None:
        aux_dict = dict(aux) if aux is not None else None
        for side in ARM_PREFIXES:
            scripts[side].append(
                Step(
                    label=label,
                    arm_q=HOME_ARM_Q_BY_SIDE[side].copy(),
                    gripper=gripper,
                    duration=duration,
                    phase=phase,
                    aux_ctrl=aux_dict,
                    set_geom_rgba=set_geom_rgba,
                )
            )

    def push_arms(
        label: str,
        duration: float,
        phase: TaskPhase,
        arm_q_by_side: dict[ArmSide, np.ndarray],
        aux: tuple[tuple[str, float], ...] | dict[str, float],
        *,
        gripper: GripperState = "open",
    ) -> None:
        aux_dict = dict(aux)
        for side in ARM_PREFIXES:
            scripts[side].append(
                Step(
                    label=f"{label} {side.rstrip('_/')}",
                    arm_q=arm_q_by_side[side].copy(),
                    gripper=gripper,
                    duration=duration,
                    phase=phase,
                    aux_ctrl=aux_dict,
                )
            )

    aux_origin = dict(_BASE_ORIGIN)
    aux_traverse_end = dict(_BASE_AT_TRAVERSE_END)
    aux_yaw_only = dict(base_aux_targets(x=_CLICK_X, y=0.0, yaw=_CLICK_YAW))
    aux_click = dict(_BASE_AT_CLICK)

    # === SETUP ============================================================
    push_both("home", 1.0, TaskPhase.SETUP, aux=aux_origin)
    push_both("settle", 0.5, TaskPhase.SETUP, aux=aux_origin)

    # === TRAVERSE_INTO_AISLE =============================================
    # ~5 m drive at ~0.5 m/s — realistic indoor mobile-robot cruise.
    push_both(
        "drive into aisle",
        10.0,
        TaskPhase.TRAVERSE_INTO_AISLE,
        aux=aux_traverse_end,
    )

    # === ALIGN_TO_TARGET =================================================
    # Two base-only sub-steps within one phase: yaw in place at chassis_Y=0
    # (where the swept-corner radius fits), then translate +Y to the click
    # pose (no rotation, just a slide). Yaw at ~22°/s and translate at
    # ~6 cm/s — slower than physically necessary but reads as "deliberate
    # robot navigation" rather than "cinematic snap".
    push_both(
        "yaw to face alert row",
        4.0,
        TaskPhase.ALIGN_TO_TARGET,
        aux=aux_yaw_only,
    )
    push_both(
        "step toward rack front",
        2.0,
        TaskPhase.ALIGN_TO_TARGET,
        aux=aux_click,
    )

    # === REACH_TO_SERVER =================================================
    # Hardcoded "touching the server" pose captured via teleop with the
    # chassis seeded at the click pose (`--start-phase reach_to_server`).
    # Re-capture and overwrite if the click geometry ever shifts (rack X
    # spacing, server depth, indicator inset, etc.) — the IK helpers are
    # gone, so the click pose is whatever is pasted here.
    q_reach: dict[ArmSide, np.ndarray] = {
        # joint6 (wrist roll) zeroed vs the captured pose so the gripper
        # plates stay level when reaching — TCP position is unchanged
        # since joint6 only rolls about the gripper axis. With both
        # wrists level the wrist cams stay upright by default and the
        # render frames don't need a 180° flip.
        ArmSide.LEFT: np.array([-0.15, 2.8, -1.95, 0.05, -0.55, 0.0]),
        ArmSide.RIGHT: np.array([0.25, 2.9, -2.1, 0.6, -0.65, 0.0]),
    }
    push_arms(
        "reach to server",
        3.0,
        TaskPhase.REACH_TO_SERVER,
        q_reach,
        aux_click,
    )

    # === WAIT_AT_SERVER ==================================================
    # Two sub-steps within one phase so the viewer sees an "examining"
    # pause BEFORE the indicator flips. The pre-flip beat alternates
    # gripper state across the two arms — when L is squeezing, R is
    # releasing, and vice versa. Reads as a coordinated "tapping at the
    # server" gesture rather than synchronised symmetric clenching.
    # `push_arms` writes the same gripper to both, so we append per-side
    # Steps directly here.
    for cycle_index in range(3):
        # L closes while R stays / opens
        scripts[ArmSide.LEFT].append(
            Step(
                label=f"L close #{cycle_index}",
                arm_q=q_reach[ArmSide.LEFT].copy(),
                gripper="closed",
                duration=0.5,
                phase=TaskPhase.WAIT_AT_SERVER,
                aux_ctrl=aux_click,
            )
        )
        scripts[ArmSide.RIGHT].append(
            Step(
                label=f"R open #{cycle_index}",
                arm_q=q_reach[ArmSide.RIGHT].copy(),
                gripper="open",
                duration=0.5,
                phase=TaskPhase.WAIT_AT_SERVER,
                aux_ctrl=aux_click,
            )
        )
        # L opens while R closes
        scripts[ArmSide.LEFT].append(
            Step(
                label=f"L open #{cycle_index}",
                arm_q=q_reach[ArmSide.LEFT].copy(),
                gripper="open",
                duration=0.5,
                phase=TaskPhase.WAIT_AT_SERVER,
                aux_ctrl=aux_click,
            )
        )
        scripts[ArmSide.RIGHT].append(
            Step(
                label=f"R close #{cycle_index}",
                arm_q=q_reach[ArmSide.RIGHT].copy(),
                gripper="closed",
                duration=0.5,
                phase=TaskPhase.WAIT_AT_SERVER,
                aux_ctrl=aux_click,
            )
        )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                label=f"flip indicator {side.rstrip('_/')}",
                arm_q=q_reach[side].copy(),
                gripper="open",
                duration=1.5,
                phase=TaskPhase.WAIT_AT_SERVER,
                aux_ctrl=aux_click,
                # Only the LEFT arm's step carries the rgba flip — the
                # mechanism is global, doesn't matter which arm's step
                # triggers it. Putting it on one side avoids a redundant
                # second remove+re-add when both arms' first_tick fire.
                set_geom_rgba=(
                    (
                        ALERT_LIGHT_GEOM_NAME,
                        LAYOUT.light.rgba_green,
                    ),
                )
                if side is ArmSide.LEFT
                else (),
            )
        )

    # === RETRACT =========================================================
    push_both(
        "arms back to home",
        2.0,
        TaskPhase.RETRACT,
        aux=aux_click,
    )

    for side in ARM_PREFIXES:
        print(f"  [{side}] {len(scripts[side])} steps planned")

    apply_initial_state(model, data, arms, cube_body_ids)
    return scripts


# Suppress unused-import warnings — `Sequence` and `Any` are kept available
# for future extension (e.g. typed aux_ctrl, scene-specific helpers).
_ = (Sequence, Any)
