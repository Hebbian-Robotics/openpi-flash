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
from paths import D435I_XML
from robots.mobile_aloha import (
    BASE_X_JOINT_NAME,
    BASE_Y_JOINT_NAME,
    BASE_YAW_JOINT_NAME,
    LEFT_ARM_MOUNT_SITE,
    RIGHT_ARM_MOUNT_SITE,
    TOP_CAM_MOUNT_SITE,
    load_mobile_aloha,
)
from robots.piper import load_piper
from scene_base import (
    GripperState,
    JointSetStatic,
    PhaseContract,
    PhaseState,
    QaccSentinel,
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


AUX_ACTUATOR_NAMES: tuple[str, ...] = tuple(m.value for m in IndicatorAux)


def base_aux_targets(*, x: float, y: float, yaw: float) -> tuple[tuple[str, float], ...]:
    """Canonical phase-boundary representation for planar base targets."""
    return (
        (IndicatorAux.BASE_X, x),
        (IndicatorAux.BASE_Y, y),
        (IndicatorAux.BASE_YAW, yaw),
    )


CAMERAS: tuple[tuple[str, CameraRole], ...] = (("forward_cam", CameraRole.TOP),)

# Top cam is rigidly attached to base_link with optical axis along chassis
# +x, so the view always frames whatever the robot is facing — no yaw-driven
# swing. The d435i mesh body is still parented to the pole site for visual
# fidelity, but the camera itself bypasses it to keep the frame story simple.
_ALERT_RACK_BODY_NAME = f"rack_{LAYOUT.alert.row}_r{LAYOUT.alert.rack_index}"

CAMERA_INVARIANTS: tuple[CameraInvariant, ...] = (
    FixedCameraInvariant(name="forward_cam", parent_body="base_link"),
)

# All static geometry sits inside one `data_center` body (racks + servers +
# lights), so scene_check's same-body skip handles within-body overlaps.
# No cross-body overlaps are expected — the chassis stays clear of rack
# panels at every phase boundary by construction (see layout invariants).
ALLOWED_STATIC_OVERLAPS: tuple[tuple[str, str], ...] = ()


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

    for row in ("left", "right"):
        for rack_index in range(len(aisle_cfg.rack_x_centres)):
            for slot_index in range(servers_cfg.n_per_rack):
                server_x = LAYOUT.rack_centre_x(rack_index)
                server_y = LAYOUT.server_centre_y(row)
                server_z = LAYOUT.server_centre_z(slot_index)
                dc.add(
                    "geom",
                    dclass="visual",
                    type="box",
                    pos=[server_x, server_y, server_z],
                    size=list(server_cfg.half),
                    rgba=list(server_cfg.rgba_chassis),
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

    visual = root.default.add("default", dclass="visual")
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.mass = 0.0

    wb = root.worldbody

    # Lighting — same setup as the live server-swap scene.
    root.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    root.visual.headlight.ambient = [0.4, 0.4, 0.4]
    root.visual.headlight.specular = [0.0, 0.0, 0.0]
    wb.add(
        "light",
        name="key_directional",
        pos=[3.0, 0.0, 4.0],
        dir=[0.0, 0.0, -1.0],
        directional="true",
        diffuse=[0.5, 0.5, 0.5],
        specular=[0.1, 0.1, 0.1],
    )
    wb.add(
        "light",
        name="aisle_fill",
        pos=[3.0, 0.0, 2.5],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.30, 0.30, 0.32],
        specular=[0.05, 0.05, 0.05],
        castshadow="false",
    )

    wb.add(
        "geom",
        type="plane",
        size=[8.0, 4.0, 0.1],
        pos=[3.0, 0.0, 0.0],
        rgba=[0.78, 0.78, 0.80, 1.0],
    )

    # Racks + servers + indicator lights.
    _add_data_center(root)

    # Bimanual Piper arms on Mobile ALOHA's front mount sites. Mirrors the
    # legacy tiago_piper scene's per-arm subtree-prep (TCP site on link6,
    # then attach), but with no wrist camera (Piper's wrist orientation
    # convention differs from UR10e and the legacy scene didn't wire
    # wrist cams either).
    arm_mount_sites = {
        ArmSide.LEFT: LEFT_ARM_MOUNT_SITE,
        ArmSide.RIGHT: RIGHT_ARM_MOUNT_SITE,
    }
    for side in ARM_PREFIXES:
        piper = load_piper(side)
        link6 = piper.find("body", "link6")
        if link6 is None:
            raise RuntimeError(f"piper {side!r} missing 'link6' after load")
        # TCP site at 14 cm forward of link6 along the gripper axis. IK /
        # weld code addresses this by `f"{side}tcp"` — inside the namescope
        # the local name is `tcp`.
        link6.add(
            "site",
            name="tcp",
            pos=[0.0, 0.0, 0.14],
            size=[0.006, 0.006, 0.006],
        )
        mount_site_name = arm_mount_sites[side]
        mount_site = root.find("site", mount_site_name)
        if mount_site is None:
            raise RuntimeError(
                f"mobile-aloha mount site {mount_site_name!r} not found — "
                "did robots/mobile_aloha.py change site names?"
            )
        mount_site.attach(piper)

    # D435i mesh body on the camera pole (purely visual — gives the camera
    # something to look like). The actual camera is attached separately to
    # `base_link` below so we have direct control over its frame.
    top_d435i = mjcf.from_path(str(D435I_XML))
    top_d435i.model = "top"
    top_cam_mount = root.find("site", TOP_CAM_MOUNT_SITE)
    if top_cam_mount is None:
        raise RuntimeError(f"mobile-aloha mount site {TOP_CAM_MOUNT_SITE!r} not found")
    top_cam_mount.attach(top_d435i)

    # Forward-facing top camera, rigidly attached to the chassis. Bypass the
    # d435i body so the camera frame doesn't depend on the menagerie XML's
    # internal orientation — this gives a predictable, always-faces-forward
    # cinematic. `quat = (0.7071, 0, -0.7071, 0)` is -π/2 about base_link's
    # +y axis, which rotates the camera's default look direction (-z_cam) so
    # it points along base_link's +x axis (chassis forward). The camera
    # rotates with the chassis through the planar yaw joint, so the view
    # always frames whatever direction the robot is heading.
    base_link_body = root.find("body", "base_link")
    if base_link_body is None:
        raise RuntimeError("Mobile ALOHA chassis body 'base_link' not found")
    base_link_body.add(
        "camera",
        name="forward_cam",
        pos=[0.076, 0.0, 1.6],  # at the camera-pole top, in chassis local frame
        quat=[0.7071067811865476, 0.0, -0.7071067811865476, 0.0],
        mode="fixed",
        fovy=69.0,
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
        ArmSide.LEFT: np.array([-0.15, 2.8, -1.95, 0.05, -0.55, -1.35]),
        ArmSide.RIGHT: np.array([0.25, 2.9, -2.1, 0.6, -0.65, -1.5]),
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
