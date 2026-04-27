"""Data-center server-swap scene — Mobile ALOHA base + UR10e + 2F-85.

A pair of UR10e arms with Robotiq 2F-85 grippers rides on a Mobile-ALOHA-
style chassis parked in front of a 19" 42U rack. Per
`experiments/bimanual_sim/NEW_LAYOUT.md`:

    A. Extract  — grip bezel handles, pull straight out, base reverses,
                  hold the server level 30 cm in front of the slot.
    B. Place    — base relocates to the cart's right side facing +Y;
                  lower the old server onto the trolley bottom tray.
    C. Pick     — same base pose; grip the new server on the top shelf
                  and lift 3 cm clear.
    D. Insert   — base returns to origin facing +X; align at slot;
                  base advances 0.10 m forward while arms extend to push
                  the server into the rack.

Bimanual sync grip: per-arm grasp welds onto the same server activate at
the same step boundary, and each arm's IK target tracks the matching
front-bezel handle. The 12 kg server is too heavy and too wide for a
single wrist, so both arms share the load. Two grasp welds onto one
freejoint body do over-constrain it, but for a position-only kinematic
puppet (gravity off, no contacts) the pair stays consistent as long as
both arms reach matched handle world coords each step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import mujoco
import numpy as np
from dm_control import mjcf

from arm_handles import ArmHandles, ArmSide, arm_joint_suffixes
from cameras import CameraRole
from ik import PositionOnly, solve_ik
from paths import D405_MESH_STL, D435I_XML
from robots.mobile_aloha import (
    BASE_X_JOINT_NAME,
    BASE_Y_JOINT_NAME,
    BASE_YAW_JOINT_NAME,
    LEFT_ARM_MOUNT_SITE,
    RIGHT_ARM_MOUNT_SITE,
    TOP_CAM_MOUNT_SITE,
    load_mobile_aloha,
)
from robots.ur10e import load_ur10e_with_gripper
from scene_base import (
    BimanualHandleSeparation,
    CubeID,
    GrippablePoseExpectation,
    GripperState,
    GripperStateHold,
    HeldObjectLevelness,
    JointSetStatic,
    PhaseContract,
    PhaseState,
    Position3,
    QaccSentinel,
    Step,
    TaskPhase,
    WeldHoldInvariant,
    make_cube_id,
)
from scene_check import (
    AttachmentConstraint,
    CameraInvariant,
    FixedCameraInvariant,
    TargetingCameraInvariant,
    WeldAttachment,
)
from scenes.mobile_aloha_ur10e_server_swap_layout import (
    BASE_HOME_POSE,
    HOME_ARM_Q_BY_SIDE,
    IK_SEED_Q,
    LAYOUT,
    PHASE_HOMES,
)
from welds import activate_attachment_weld, deactivate_weld

NAME = "mobile_aloha_ur10e_server_swap"
# `ROBOT_KIND` selects the `arm_handles.get_arm_handles` branch.
ROBOT_KIND = "ur10e"
# Planar base joints the IK solver must lock (not optimise). Shared across
# scene-agnostic IK tools.
IK_LOCKED_JOINT_NAMES: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)
# Grippable objects addressable via Step.weld_activate / weld_deactivate.
# Index order is load-bearing: the runner uses it as an int index.
GRIPPABLES: tuple[str, ...] = ("server", "new_server")
N_CUBES = len(GRIPPABLES)


class DataCenterAux(StrEnum):
    """Scene-owned (non-arm) actuators addressable via Step.aux_ctrl.

    Drive the 3-DOF planar base joint added by `load_mobile_aloha`. mink IK
    locks these so the planner uses only the arm chain. No lift entry —
    Mobile ALOHA is lift-less, so z-motion is via the arms.
    """

    BASE_X = BASE_X_JOINT_NAME
    BASE_Y = BASE_Y_JOINT_NAME
    BASE_YAW = BASE_YAW_JOINT_NAME


AUX_ACTUATOR_NAMES: tuple[str, ...] = tuple(m.value for m in DataCenterAux)

# Base waypoints (per NEW_LAYOUT.md). `base_link` starts at world origin so
# qpos values equal world coords.
#   * Action A: back up 0.20 m to clear the extracted server with margin.
#   * Action B/C: park at (0.30, 0.25) facing +Y so the cart at (0.30, 0.90)
#     is directly in front (cart Y - base Y = 0.65 m → near edge ~0.45 m
#     ahead, within UR10e reach).
#   * Action D: push 0.10 m forward so arms can drive the server the last
#     0.65 m without exceeding reach envelope.
BASE_PRE_EXTRACT = -0.20
BASE_AT_CART_X = 0.30
BASE_AT_CART_Y = 0.25
YAW_TO_CART = math.pi / 2.0
BASE_AT_INSERT = 0.10


def base_aux_targets(*, x: float, y: float, yaw: float) -> tuple[tuple[str, float], ...]:
    """Canonical phase-boundary representation for planar base targets."""
    return (
        (DataCenterAux.BASE_X, x),
        (DataCenterAux.BASE_Y, y),
        (DataCenterAux.BASE_YAW, yaw),
    )


CAMERAS: tuple[tuple[str, CameraRole], ...] = (
    # dm_control namespaces attached subtrees as `<model>/`; the d435i.xml's
    # `d435i_cam` compiles to `top/d435i_cam`. The wrist cams are added
    # inside the UR10e subtree before attach, so they pick up the side prefix.
    ("top/d435i_cam", CameraRole.TOP),
    ("left/wrist_d405_cam", CameraRole.WRIST),
    ("right/wrist_d405_cam", CameraRole.WRIST),
)

# Camera invariants — pinned at startup by `scene_check.check_scene`, so a
# future edit that re-parents or flips a camera mode fails fast.
CAMERA_INVARIANTS: tuple[CameraInvariant, ...] = (
    # Top cam tracks `rack_frame` so the optical axis stays on the rack as
    # the base rotates.
    TargetingCameraInvariant(
        name="top/d435i_cam",
        parent_body="top/d435i",
        targetbody="rack_frame",
        mode="targetbody",
    ),
    # Wrist cams: 180°-x quat aligns optical axis (-z) with link +z (gripper /
    # TCP axis), so each wrist view looks at whatever the arm is reaching for.
    FixedCameraInvariant(name="left/wrist_d405_cam", parent_body="left/wrist_3_link"),
    FixedCameraInvariant(name="right/wrist_d405_cam", parent_body="right/wrist_3_link"),
)


# Dimensions live in `DataCenterLayout` (scenes/mobile_aloha_ur10e_server_swap_layout.py).
# World frame: origin at the robot's base centre on the floor, +x toward rack,
# +y to the robot's right, +z up.


# Cross-body AABB overlaps that are by-design (panels meeting at seams,
# brackets flush against walls). `scene_check` skips same-body geom pairs
# automatically.
ALLOWED_STATIC_OVERLAPS: tuple[tuple[str, str], ...] = (
    # Rack panels share cabinet-seam edges where they meet.
    ("rack_rear", "rack_side_L"),
    ("rack_rear", "rack_side_R"),
    ("rack_rear", "rack_top"),
    ("rack_rear", "rack_bottom"),
    ("rack_side_L", "rack_top"),
    ("rack_side_L", "rack_bottom"),
    ("rack_side_R", "rack_top"),
    ("rack_side_R", "rack_bottom"),
    ("rack_top", "rack_bottom"),
    # Shelf is a thin plate spanning the full rack interior; touches the
    # surrounding panels at every edge.
    ("rack_rear", "rack_shelf"),
    ("rack_side_L", "rack_shelf"),
    ("rack_side_R", "rack_shelf"),
    # Bezel handle cylinders protrude forward from the chassis -X face;
    # their AABBs straddle the chassis box AABB by design (the cylinder
    # endpoint sits inside the chassis half-extent so the geom looks
    # mounted, not floating). Listed at the geom level since chassis +
    # handles share the same body.
    ("server", "server_handle_L"),
    ("server", "server_handle_R"),
    ("new_server", "new_server_handle_L"),
    ("new_server", "new_server_handle_R"),
    # Top camera sits directly on top of the camera pole — the pole-top
    # and mesh-bottom AABBs touch by design. The camera-mesh body
    # contains ~9 sub-geoms so a body-pair allow covers the lot.
    ("world", "top/d435i"),
    # Mobile ALOHA's base mesh uses a conservative sphere bound in
    # scene_check, so the rack walls and the cart frame can graze the
    # base mesh sphere even when the actual mesh geometry is fine.
    ("world", "rack_side_L"),
    ("world", "rack_side_R"),
    ("world", "rack_bottom"),
    ("world", "cart_frame"),
)


# Body-pair equalities — single registry shared by build_spec, id resolution,
# and reset. WELD only (no cables → no CONNECTs). `welds.activate_attachment_weld`
# captures the relpose at activation so the body stays where it is rather than
# snapping to body1's origin.


class AttachmentWeldName(StrEnum):
    """Named body-pair equalities used by Step.attach_activate/deactivate."""

    SERVER_IN_RACK = "weld_server_in_rack"
    SERVER_ON_CART_BOTTOM = "weld_server_on_cart_bottom"
    NEW_ON_CART_TOP = "weld_new_on_cart_top"
    NEW_IN_RACK = "weld_new_in_rack"


ATTACHMENTS: tuple[AttachmentConstraint, ...] = (
    WeldAttachment(
        name=AttachmentWeldName.SERVER_IN_RACK,
        body_a="server",
        body_b="rack_frame",
        initially_active=True,
    ),
    WeldAttachment(
        name=AttachmentWeldName.SERVER_ON_CART_BOTTOM,
        body_a="server",
        body_b="cart_frame",
        initially_active=False,
    ),
    WeldAttachment(
        name=AttachmentWeldName.NEW_ON_CART_TOP,
        body_a="new_server",
        body_b="cart_frame",
        initially_active=True,
    ),
    WeldAttachment(
        name=AttachmentWeldName.NEW_IN_RACK,
        body_a="new_server",
        body_b="rack_frame",
        initially_active=False,
    ),
)


# --- Pose expectation helpers ----------------------------------------------
# Re-derived from LAYOUT so a layout edit (rack moves, cart shifts)
# updates the contracts in lockstep. Pose tolerance = 2 cm by default;
# tighter for the rack server slot since the WELD pose-locks it.
def _pose(name: str, position: Position3, *, tolerance_m: float = 0.02) -> GrippablePoseExpectation:
    return GrippablePoseExpectation(
        name=name,
        position=(float(position[0]), float(position[1]), float(position[2])),
        tolerance_m=tolerance_m,
    )


# Tolerances:
#   * 10 mm for unwelded initial poses (untouched scene-reset bodies —
#     no constraint solver drift to worry about).
#   * 60 mm for post-weld poses. mjEQ_WELD's default solref/solimp pulls
#     the body toward the target but doesn't pin it exactly — under
#     gravity the welded body settles ~40-50 mm off target. 60 mm is
#     comfortably above observed drift while still flagging the
#     "server fell out of the rack entirely" case (~hundreds of mm).
_SERVER_IN_RACK_POSE = _pose("server", LAYOUT.server_world_pos_in_rack, tolerance_m=0.01)
_NEW_SERVER_ON_CART_TOP_POSE = _pose(
    "new_server", LAYOUT.new_server_initial_world_pos, tolerance_m=0.01
)
_OLD_SERVER_ON_CART_BOTTOM_POSE = _pose(
    "server", LAYOUT.old_server_stow_world_pos, tolerance_m=0.06
)
_NEW_SERVER_IN_RACK_POSE = _pose("new_server", LAYOUT.server_world_pos_in_rack, tolerance_m=0.06)


# QACC sentinel — applies to every phase. The integrator must not raise
# any new BADQACC warnings during a phase (max_increase=0).
_QACC_SENTINEL = QaccSentinel(max_increase=0)

# Joints that must stay frozen for an arm-only or base-only phase. Each phase
# attaches the appropriate `JointSetStatic` invariant(s). Arm names come from
# the same `arm_joint_suffixes` source the runner + teleop use.
_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{side.value}{suffix}" for side in ARM_PREFIXES for suffix in arm_joint_suffixes(ROBOT_KIND)
)
_ARMS_STATIC = JointSetStatic(joint_names=_ARM_JOINT_NAMES, label="arms")
_BASE_STATIC = JointSetStatic(joint_names=IK_LOCKED_JOINT_NAMES, label="base")

# Gripper hold: during a carry phase, the actuator ctrl must stay at the 2F-85
# closed value (255). Catches the realistic "gripper toggled mid-carry → load
# drops" failure mode that QACC alone wouldn't flag.
_GRIPPER_ACTUATOR_NAMES: tuple[str, ...] = tuple(
    f"{side.value}gripper/fingers_actuator" for side in ARM_PREFIXES
)
_GRIPPERS_CLOSED = GripperStateHold(
    actuator_names=_GRIPPER_ACTUATOR_NAMES,
    expected_ctrl_value=255.0,
    label="grippers closed",
)

# Bimanual coordination: while both grasp welds onto one server are active,
# the L/R TCPs must stay at the bezel handle separation (24 cm). Drift means
# the welds are over-constraining and physics will explode shortly. Each
# carried object gets its own gated invariant.
_OLD_GRASP_WELDS = ("left_grasp_cube0", "right_grasp_cube0")
_NEW_GRASP_WELDS = ("left_grasp_cube1", "right_grasp_cube1")
_BIMANUAL_HANDLE_DISTANCE_M = 2.0 * LAYOUT.handles.y_offset_abs
_OLD_BIMANUAL_HOLD = BimanualHandleSeparation(
    left_tcp_site="left/tcp",
    right_tcp_site="right/tcp",
    target_distance_m=_BIMANUAL_HANDLE_DISTANCE_M,
    requires_active_welds=_OLD_GRASP_WELDS,
)
_NEW_BIMANUAL_HOLD = BimanualHandleSeparation(
    left_tcp_site="left/tcp",
    right_tcp_site="right/tcp",
    target_distance_m=_BIMANUAL_HANDLE_DISTANCE_M,
    requires_active_welds=_NEW_GRASP_WELDS,
)

# Levelness: NEW_LAYOUT.md specifies "server pitch/roll under 1°" for the
# carry. Five degrees gives margin for the soft-descent / push moments
# without masking real tipping. Gated on the same per-server grasp welds
# as the bimanual hold check.
_LEVELNESS_LIMIT_RAD = math.radians(5.0)
_OLD_SERVER_LEVEL = HeldObjectLevelness(
    body_name="server",
    max_pitch_rad=_LEVELNESS_LIMIT_RAD,
    max_roll_rad=_LEVELNESS_LIMIT_RAD,
    requires_active_welds=_OLD_GRASP_WELDS,
)
_NEW_SERVER_LEVEL = HeldObjectLevelness(
    body_name="new_server",
    max_pitch_rad=_LEVELNESS_LIMIT_RAD,
    max_roll_rad=_LEVELNESS_LIMIT_RAD,
    requires_active_welds=_NEW_GRASP_WELDS,
)


_BASE_ORIGIN = base_aux_targets(x=0.0, y=0.0, yaw=0.0)
_BASE_PRE_EXTRACT = base_aux_targets(x=BASE_PRE_EXTRACT, y=0.0, yaw=0.0)
_BASE_AT_CART = base_aux_targets(x=BASE_AT_CART_X, y=BASE_AT_CART_Y, yaw=YAW_TO_CART)
_BASE_AT_INSERT = base_aux_targets(x=BASE_AT_INSERT, y=0.0, yaw=0.0)


PHASE_CONTRACTS: tuple[PhaseContract, ...] = (
    PhaseContract(
        phase=TaskPhase.SETUP,
        starts=PhaseState(
            description="Scene reset: old server in rack, new server on cart.",
            active_attachments=(
                AttachmentWeldName.SERVER_IN_RACK,
                AttachmentWeldName.NEW_ON_CART_TOP,
            ),
            inactive_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_IN_RACK,
            ),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_SERVER_IN_RACK_POSE, _NEW_SERVER_ON_CART_TOP_POSE),
        ),
        ends=PhaseState(
            description="Both arms at home, ready to extract the old server.",
            active_attachments=(
                AttachmentWeldName.SERVER_IN_RACK,
                AttachmentWeldName.NEW_ON_CART_TOP,
            ),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_SERVER_IN_RACK_POSE, _NEW_SERVER_ON_CART_TOP_POSE),
        ),
        invariants=(
            _QACC_SENTINEL,
            _ARMS_STATIC,
            _BASE_STATIC,
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_IN_RACK, must_be_active=True),
            WeldHoldInvariant(name=AttachmentWeldName.NEW_ON_CART_TOP, must_be_active=True),
        ),
    ),
    # Action A — arm-only at base = origin: grip handles, free server, pull
    # 10 cm clear of rail detents. Base reversal is the next phase.
    PhaseContract(
        legal_predecessors=(TaskPhase.SETUP,),
        phase=TaskPhase.REMOVE_OLD_SERVER,
        starts=PhaseState(
            description="Old server pinned in rack; arms at home, base at origin.",
            active_attachments=(
                AttachmentWeldName.SERVER_IN_RACK,
                AttachmentWeldName.NEW_ON_CART_TOP,
            ),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_SERVER_IN_RACK_POSE, _NEW_SERVER_ON_CART_TOP_POSE),
        ),
        ends=PhaseState(
            description="Server gripped bimanually, pulled 10 cm clear of detents; base unchanged.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _BASE_STATIC,
            _OLD_BIMANUAL_HOLD,
            _OLD_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.NEW_ON_CART_TOP, must_be_active=True),
        ),
    ),
    # Base-only: chassis reverses 0.20 m; arms hold their joint config so the
    # welded server rides backward with the chassis.
    PhaseContract(
        legal_predecessors=(TaskPhase.REMOVE_OLD_SERVER,),
        phase=TaskPhase.BACKUP_FROM_RACK,
        starts=PhaseState(
            description="Server gripped bimanually; base at origin.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        ends=PhaseState(
            description="Base reversed to pre-extract pose; arms unchanged.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_PRE_EXTRACT,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _ARMS_STATIC,
            _GRIPPERS_CLOSED,
            _OLD_BIMANUAL_HOLD,
            _OLD_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.NEW_ON_CART_TOP, must_be_active=True),
        ),
    ),
    # Base-only: chassis rotates + translates to the cart's right side.
    PhaseContract(
        legal_predecessors=(TaskPhase.BACKUP_FROM_RACK,),
        phase=TaskPhase.TRAVERSE_TO_CART,
        starts=PhaseState(
            description="Base at pre-extract; server held bimanually.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_PRE_EXTRACT,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        ends=PhaseState(
            description="Base parked at cart facing +Y; arms unchanged.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _ARMS_STATIC,
            _GRIPPERS_CLOSED,
            _OLD_BIMANUAL_HOLD,
            _OLD_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.NEW_ON_CART_TOP, must_be_active=True),
        ),
    ),
    # Action B — arm-only at base = cart: descend, pin to bottom tray, release.
    PhaseContract(
        legal_predecessors=(TaskPhase.TRAVERSE_TO_CART,),
        phase=TaskPhase.STOW_OLD_SERVER,
        starts=PhaseState(
            description="Base at cart; old server held bimanually.",
            active_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(_NEW_SERVER_ON_CART_TOP_POSE,),
        ),
        ends=PhaseState(
            description="Old server pinned on cart bottom; arms released; base unchanged.",
            active_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_ON_CART_TOP,
            ),
            inactive_attachments=(AttachmentWeldName.SERVER_IN_RACK,),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(
                _OLD_SERVER_ON_CART_BOTTOM_POSE,
                _NEW_SERVER_ON_CART_TOP_POSE,
            ),
        ),
        invariants=(
            _QACC_SENTINEL,
            _BASE_STATIC,
            _OLD_BIMANUAL_HOLD,
            _OLD_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.NEW_ON_CART_TOP, must_be_active=True),
        ),
    ),
    # Action C — arm-only at base = cart: grip new server, lift 3 cm clear.
    PhaseContract(
        legal_predecessors=(TaskPhase.STOW_OLD_SERVER,),
        phase=TaskPhase.RETRIEVE_NEW_SERVER,
        starts=PhaseState(
            description="New server pinned on cart top shelf; old server stowed below.",
            active_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_ON_CART_TOP,
            ),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(
                _OLD_SERVER_ON_CART_BOTTOM_POSE,
                _NEW_SERVER_ON_CART_TOP_POSE,
            ),
        ),
        ends=PhaseState(
            description="New server gripped bimanually and lifted clear of the cart.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _BASE_STATIC,
            _NEW_BIMANUAL_HOLD,
            _NEW_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_ON_CART_BOTTOM, must_be_active=True),
        ),
    ),
    # Base-only: cart pose → origin facing +X. Arms hold; new server rides.
    PhaseContract(
        legal_predecessors=(TaskPhase.RETRIEVE_NEW_SERVER,),
        phase=TaskPhase.TRAVERSE_TO_RACK,
        starts=PhaseState(
            description="Base at cart with new server held bimanually.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_AT_CART,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        ends=PhaseState(
            description="Base back at origin facing +X; arms unchanged.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _ARMS_STATIC,
            _GRIPPERS_CLOSED,
            _NEW_BIMANUAL_HOLD,
            _NEW_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_ON_CART_BOTTOM, must_be_active=True),
        ),
    ),
    # Base-only: chassis advances 0.10 m; arms hold so the welded server
    # tracks the chassis straight at the rack opening.
    PhaseContract(
        legal_predecessors=(TaskPhase.TRAVERSE_TO_RACK,),
        phase=TaskPhase.ADVANCE_INTO_RACK,
        starts=PhaseState(
            description="Base at origin facing +X with new server held.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        ends=PhaseState(
            description="Base advanced to insert offset; arms unchanged.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_AT_INSERT,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        invariants=(
            _QACC_SENTINEL,
            _ARMS_STATIC,
            _GRIPPERS_CLOSED,
            _NEW_BIMANUAL_HOLD,
            _NEW_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_ON_CART_BOTTOM, must_be_active=True),
        ),
    ),
    # Action D — arm-only at base = insert: extend arms to seat server,
    # pin in rack, release, withdraw.
    PhaseContract(
        legal_predecessors=(TaskPhase.ADVANCE_INTO_RACK,),
        phase=TaskPhase.INSTALL_NEW_SERVER,
        starts=PhaseState(
            description="Base at insert; new server held bimanually.",
            active_attachments=(AttachmentWeldName.SERVER_ON_CART_BOTTOM,),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_AT_INSERT,
            expected_grippable_poses=(_OLD_SERVER_ON_CART_BOTTOM_POSE,),
        ),
        ends=PhaseState(
            description="New server pinned in rack slot; arms released; base unchanged.",
            active_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_IN_RACK,
            ),
            inactive_attachments=(AttachmentWeldName.NEW_ON_CART_TOP,),
            base_aux=_BASE_AT_INSERT,
            expected_grippable_poses=(
                _OLD_SERVER_ON_CART_BOTTOM_POSE,
                _NEW_SERVER_IN_RACK_POSE,
            ),
        ),
        invariants=(
            _QACC_SENTINEL,
            _BASE_STATIC,
            _NEW_BIMANUAL_HOLD,
            _NEW_SERVER_LEVEL,
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_ON_CART_BOTTOM, must_be_active=True),
        ),
    ),
    # Reset is a sequenced wind-down: arm-home first, then base-home. Each
    # step moves only one set, but the phase as a whole isn't joint-static —
    # so no JointSetStatic invariant; QACC and weld-holds still apply.
    PhaseContract(
        legal_predecessors=(TaskPhase.INSTALL_NEW_SERVER,),
        phase=TaskPhase.RESET,
        starts=PhaseState(
            description="Task completed; new server installed.",
            active_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_IN_RACK,
            ),
            base_aux=_BASE_AT_INSERT,
            expected_grippable_poses=(
                _OLD_SERVER_ON_CART_BOTTOM_POSE,
                _NEW_SERVER_IN_RACK_POSE,
            ),
        ),
        ends=PhaseState(
            description="Arms at home; base back at origin.",
            active_attachments=(
                AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                AttachmentWeldName.NEW_IN_RACK,
            ),
            base_aux=_BASE_ORIGIN,
            expected_grippable_poses=(
                _OLD_SERVER_ON_CART_BOTTOM_POSE,
                _NEW_SERVER_IN_RACK_POSE,
            ),
        ),
        invariants=(
            _QACC_SENTINEL,
            WeldHoldInvariant(name=AttachmentWeldName.NEW_IN_RACK, must_be_active=True),
            WeldHoldInvariant(name=AttachmentWeldName.SERVER_ON_CART_BOTTOM, must_be_active=True),
        ),
    ),
)


# -----------------------------------------------------------------------------
# Grippable addressing
# -----------------------------------------------------------------------------


def grippable_id(name: str) -> CubeID:
    """Resolve a grippable-object name to its bounds-checked CubeID."""
    try:
        index = GRIPPABLES.index(name)
    except ValueError as exc:
        raise KeyError(f"unknown grippable {name!r}; known: {GRIPPABLES}") from exc
    return make_cube_id(index, N_CUBES)


def grasp_weld(side: ArmSide, cube_id: CubeID) -> str:
    """Name of the per-arm grasp weld for a given cube. `_` separator (not `/`)
    because dm_control.mjcf reserves `/` for namespace scoping; matches the
    convention `arm_handles` uses to look up `weld_ids` after compilation."""
    return f"{side.replace('/', '_')}grasp_cube{cube_id}"


# -----------------------------------------------------------------------------
# Spec construction
# -----------------------------------------------------------------------------


def _add_cart(root: mjcf.RootElement, visual_class: str) -> mjcf.Element:
    """Service cart: chrome corner posts, two shelves with retention lips +
    ESD mats, four swivel-caster wheels, push handle.

    All sub-geoms use the `visual` default class (no contacts, no mass) —
    the cart is a visual stand; servers are held by welds, not by sitting
    on the shelves.
    """
    cart_cfg = LAYOUT.cart
    cart_rgba = [0.55, 0.58, 0.60, 1.0]
    chrome_rgba = [0.78, 0.80, 0.82, 1.0]
    mat_rgba = [0.08, 0.08, 0.10, 1.0]
    wheel_rgba = [0.10, 0.10, 0.12, 1.0]
    lip_half_z = 0.010
    mat_thickness = 0.003

    cart = root.worldbody.add(
        "body",
        name="cart_frame",
        pos=[cart_cfg.center_x, cart_cfg.center_y, 0.0],
    )

    # Corner posts — chrome tubing, cylinders default to local +z (vertical).
    post_top_z = cart_cfg.top_shelf_z + 0.01
    post_half_z = post_top_z * 0.5
    post_radius = cart_cfg.post_half
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cart.add(
                "geom",
                dclass=visual_class,
                type="cylinder",
                pos=[
                    sx * (cart_cfg.half_x - cart_cfg.post_half),
                    sy * (cart_cfg.half_y - cart_cfg.post_half),
                    post_half_z,
                ],
                size=[post_radius, post_half_z],
                rgba=chrome_rgba,
            )

    # Top + bottom shelf plates.
    cart.add(
        "geom",
        dclass=visual_class,
        name="cart_top_shelf",
        type="box",
        pos=[0.0, 0.0, cart_cfg.top_shelf_z],
        size=[cart_cfg.half_x, cart_cfg.half_y, cart_cfg.shelf_thickness],
        rgba=cart_rgba,
    )
    cart.add(
        "geom",
        dclass=visual_class,
        name="cart_bottom_shelf",
        type="box",
        pos=[0.0, 0.0, cart_cfg.bottom_shelf_z],
        size=[cart_cfg.half_x, cart_cfg.half_y, cart_cfg.shelf_thickness],
        rgba=cart_rgba,
    )

    # Retention lips on all four shelf edges — thin rails 2 cm above surface.
    for shelf_z, label in (
        (cart_cfg.top_shelf_z, "top"),
        (cart_cfg.bottom_shelf_z, "bot"),
    ):
        lip_z = shelf_z + cart_cfg.shelf_thickness + lip_half_z
        # X-edges (running along Y at +/- half_x)
        for sx in (-1.0, 1.0):
            cart.add(
                "geom",
                dclass=visual_class,
                name=f"cart_{label}_lip_x{int(sx):+d}",
                type="box",
                pos=[sx * cart_cfg.half_x, 0.0, lip_z],
                size=[cart_cfg.shelf_thickness, cart_cfg.half_y, lip_half_z],
                rgba=cart_rgba,
            )
        # Y-edges (running along X at +/- half_y)
        for sy in (-1.0, 1.0):
            cart.add(
                "geom",
                dclass=visual_class,
                name=f"cart_{label}_lip_y{int(sy):+d}",
                type="box",
                pos=[0.0, sy * cart_cfg.half_y, lip_z],
                size=[cart_cfg.half_x, cart_cfg.shelf_thickness, lip_half_z],
                rgba=cart_rgba,
            )

    # ESD mats sit on each shelf, 1 cm inset so the lip rim still shows.
    mat_inset = 0.01
    for shelf_z, label in (
        (cart_cfg.top_shelf_z, "top"),
        (cart_cfg.bottom_shelf_z, "bot"),
    ):
        cart.add(
            "geom",
            dclass=visual_class,
            name=f"cart_{label}_mat",
            type="box",
            pos=[
                0.0,
                0.0,
                shelf_z + cart_cfg.shelf_thickness + mat_thickness,
            ],
            size=[
                cart_cfg.half_x - mat_inset,
                cart_cfg.half_y - mat_inset,
                mat_thickness,
            ],
            rgba=mat_rgba,
        )

    # Casters: vertical fork stub + horizontal cylinder for the wheel.
    # Real swivel casters rotate freely, so a fixed axle orientation at
    # rest reads correctly.
    wheel_radius = cart_cfg.caster_radius
    fork_half_z = wheel_radius * 0.5
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = sx * (cart_cfg.half_x - cart_cfg.post_half)
            y = sy * (cart_cfg.half_y - cart_cfg.post_half)
            # Swivel fork stub.
            cart.add(
                "geom",
                dclass=visual_class,
                type="box",
                pos=[x, y, wheel_radius + fork_half_z],
                size=[
                    cart_cfg.post_half * 0.6,
                    cart_cfg.post_half * 0.6,
                    fork_half_z,
                ],
                rgba=chrome_rgba,
            )
            # `fromto` defines the wheel by its two end caps — natural for
            # "cylinder of length 2*half_w centred at this point, axle along X".
            half_w = wheel_radius * 0.4
            cart.add(
                "geom",
                dclass=visual_class,
                type="cylinder",
                fromto=[x - half_w, y, wheel_radius, x + half_w, y, wheel_radius],
                size=[wheel_radius],
                rgba=wheel_rgba,
            )

    # Push handle on the cart's +Y edge (the side away from the robot — what
    # a human grips when pushing the cart toward the rack).
    handle_y = cart_cfg.half_y + cart_cfg.post_half
    handle_z_top = cart_cfg.handle_height
    handle_z_mid = (cart_cfg.top_shelf_z + handle_z_top) * 0.5
    handle_post_half_z = (handle_z_top - cart_cfg.top_shelf_z) * 0.5
    handle_radius = cart_cfg.post_half
    # Vertical risers: round chrome tubes matching the corner posts.
    for sx in (-1.0, 1.0):
        cart.add(
            "geom",
            dclass=visual_class,
            type="cylinder",
            pos=[
                sx * (cart_cfg.half_x - cart_cfg.post_half),
                handle_y,
                handle_z_mid,
            ],
            size=[handle_radius, handle_post_half_z],
            rgba=chrome_rgba,
        )
    # Horizontal grip bar: round chrome tube spanning the rear edge,
    # axle along world-X.
    cart.add(
        "geom",
        dclass=visual_class,
        type="cylinder",
        fromto=[
            -(cart_cfg.half_x - cart_cfg.post_half),
            handle_y,
            handle_z_top,
            +(cart_cfg.half_x - cart_cfg.post_half),
            handle_y,
            handle_z_top,
        ],
        size=[handle_radius],
        rgba=chrome_rgba,
    )

    return cart


def _add_server_body(
    parent: mjcf.Element,
    *,
    name: str,
    pos: np.ndarray | Sequence[float],
    rgba_chassis: Sequence[float],
    server_cfg: Any,
    handles_cfg: Any,
) -> None:
    """Build a server body (chassis box + two front-bezel handle cylinders).

    Handles are short cylinders along the X axis protruding from the bezel
    (-X) face, positioned to match the IK targets the task plan reaches for.
    """
    server = parent.add("body", name=name, pos=list(pos))
    server.add("freejoint")
    server.add(
        "geom",
        type="box",
        size=list(server_cfg.half),
        rgba=rgba_chassis,
        mass=server_cfg.mass,
        contype=0,
        conaffinity=0,
    )
    # Cylinder `fromto` puts the outer tip on the bezel plane (x = -half[0]
    # in server-local frame) and the inner end recessed by `handle_length`.
    for side_label, y_sign in (("L", -1.0), ("R", +1.0)):
        server.add(
            "geom",
            name=f"{name}_handle_{side_label}",
            type="cylinder",
            fromto=[
                -server_cfg.half[0] - handles_cfg.handle_length,
                y_sign * handles_cfg.y_offset_abs,
                0.0,
                -server_cfg.half[0],
                y_sign * handles_cfg.y_offset_abs,
                0.0,
            ],
            size=[handles_cfg.handle_radius],
            rgba=[0.85, 0.85, 0.85, 1.0],
            contype=0,
            conaffinity=0,
        )


def build_spec() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Assemble the scene with dm_control.mjcf, compile via mujoco, return
    plain `(model, data)` so downstream runtime code uses the standard API."""
    root = load_mobile_aloha()

    root.option.integrator = "implicitfast"
    root.option.cone = "elliptic"
    root.option.impratio = 10.0
    root.option.timestep = 0.002
    # Disable contacts: every body is held by welds or direct qpos writes,
    # so contact detection just adds work and previously triggered a QACC
    # warning at t=41.53 s on left/joint5.
    root.option.flag.contact = "disable"
    # Puppet mode: arms are qpos-driven and the freejoint bodies (servers)
    # are always pinned by a weld, so gravity is just solver noise.
    root.option.gravity = [0.0, 0.0, 0.0]

    # `global` is a Python keyword — access via getattr.
    visual_global = getattr(root.visual, "global")
    visual_global.offwidth = 1920
    visual_global.offheight = 1080

    visual = root.default.add("default", dclass="visual")
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.mass = 0.0

    d405_mesh = root.asset.add("mesh", name="d405", file=str(D405_MESH_STL))

    wb = root.worldbody

    # Lighting matches Menagerie's `agilex_piper/scene.xml` reference setup:
    # bright headlight ambient + one overhead directional.
    root.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    root.visual.headlight.ambient = [0.4, 0.4, 0.4]
    root.visual.headlight.specular = [0.0, 0.0, 0.0]
    wb.add(
        "light",
        name="key_directional",
        pos=[0.4, 0.0, 3.0],
        dir=[0.0, 0.0, -1.0],
        directional="true",
        diffuse=[0.5, 0.5, 0.5],
        specular=[0.1, 0.1, 0.1],
    )
    wb.add(
        "light",
        name="side_fill",
        pos=[0.4, 1.5, 1.8],
        dir=[0.0, -1.0, -0.3],
        diffuse=[0.25, 0.25, 0.28],
        specular=[0.05, 0.05, 0.05],
        castshadow="false",
    )

    # Floor.
    wb.add(
        "geom",
        type="plane",
        size=[5.0, 5.0, 0.1],
        rgba=[0.78, 0.78, 0.80, 1.0],
    )

    # ---------------------- Service cart -----------------------------------
    _add_cart(root, "visual")

    # ---------------------- UR10e arms on the Mobile ALOHA front posts -----
    TCP_OFFSET_FROM_WRIST = [0.0, 0.20, 0.0]  # 2F-85's lens-of-grip along wrist +y
    WRIST_MESH_OFFSET = [0.04, 0.0, 0.0]
    arm_mount_sites = {
        ArmSide.LEFT: LEFT_ARM_MOUNT_SITE,
        ArmSide.RIGHT: RIGHT_ARM_MOUNT_SITE,
    }
    for side in ARM_PREFIXES:
        ur10e_root = load_ur10e_with_gripper(side)
        wrist3 = ur10e_root.find("body", "wrist_3_link")
        if wrist3 is None:
            raise RuntimeError(f"ur10e {side!r} missing 'wrist_3_link' after load")
        wrist3.add(
            "site",
            name="tcp",
            pos=TCP_OFFSET_FROM_WRIST,
            size=[0.006, 0.006, 0.006],
        )
        wrist3.add(
            "geom",
            dclass="visual",
            type="mesh",
            mesh=d405_mesh,
            pos=WRIST_MESH_OFFSET,
            rgba=[0.12, 0.12, 0.14, 1.0],
        )
        wrist3.add(
            "camera",
            name="wrist_d405_cam",
            pos=[0.0, 0.10, 0.0],
            quat=[0.7071067811865476, -0.7071067811865476, 0.0, 0.0],
            mode="fixed",
            fovy=87.0,
        )
        mount_site_name = arm_mount_sites[side]
        mount_site = root.find("site", mount_site_name)
        if mount_site is None:
            raise RuntimeError(
                f"mobile-aloha mount site {mount_site_name!r} not found — "
                "did robots/mobile_aloha.py change site names?"
            )
        mount_site.attach(ur10e_root)

    # ---------------------- Top camera (D435i) on the camera pole ---------
    top_d435i = mjcf.from_path(str(D435I_XML))
    top_d435i.model = "top"
    top_d435i_body = top_d435i.find("body", "d435i")
    top_cam_mount = root.find("site", TOP_CAM_MOUNT_SITE)
    if top_cam_mount is None:
        raise RuntimeError(f"mobile-aloha mount site {TOP_CAM_MOUNT_SITE!r} not found")
    top_cam_mount.attach(top_d435i)

    # ---------------------- Rack (static, open front) ---------------------
    rack_cfg = LAYOUT.rack
    rack = wb.add("body", name="rack_frame", pos=[rack_cfg.center_x, 0.0, rack_cfg.center_z])
    rack_wall_t = rack_cfg.wall_thickness
    panel_rgba = (0.10, 0.10, 0.13, 1.0)
    rhx, rhy, rhz = rack_cfg.half

    def _rack_panel(name: str, pos: Sequence[float], half: Sequence[float]) -> None:
        b = rack.add("body", name=name, pos=list(pos))
        b.add(
            "geom",
            dclass="visual",
            type="box",
            size=list(half),
            rgba=list(panel_rgba),
        )

    _rack_panel("rack_rear", (rhx - rack_wall_t, 0.0, 0.0), (rack_wall_t, rhy, rhz))
    _rack_panel("rack_side_L", (0.0, -rhy + rack_wall_t, 0.0), (rhx, rack_wall_t, rhz))
    _rack_panel("rack_side_R", (0.0, rhy - rack_wall_t, 0.0), (rhx, rack_wall_t, rhz))
    _rack_panel("rack_top", (0.0, 0.0, rhz - rack_wall_t), (rhx, rhy, rack_wall_t))
    _rack_panel("rack_bottom", (0.0, 0.0, -rhz + rack_wall_t), (rhx, rhy, rack_wall_t))
    server_cfg = LAYOUT.server
    # Shelf — a horizontal plate spanning the rack interior just below the
    # server slot, so the server visibly rests on something instead of
    # appearing to float.
    shelf_top_z_world = LAYOUT.server.slot_z - LAYOUT.server.half[2]
    shelf_half_z = 0.006
    shelf_local_z = (shelf_top_z_world - shelf_half_z) - rack_cfg.center_z
    _rack_panel(
        "rack_shelf",
        (0.0, 0.0, shelf_local_z),
        (rhx - rack_wall_t, rhy - rack_wall_t, shelf_half_z),
    )

    # Top-camera TARGETBODY needs `rack_frame` to exist as the target.
    if top_d435i_body is not None:
        top_d435i_body.add(
            "camera",
            name="d435i_cam",
            pos=[0.0, 0.0, 0.0],
            mode="targetbody",
            target=rack,
            fovy=69.0,  # D435i colour-sensor horizontal FOV ≈ 69°
        )

    # ---------------------- Servers (rack + cart-top) ---------------------
    _add_server_body(
        wb,
        name="server",
        pos=LAYOUT.server_world_pos_in_rack,
        rgba_chassis=[0.16, 0.17, 0.19, 1.0],
        server_cfg=server_cfg,
        handles_cfg=LAYOUT.handles,
    )
    _add_server_body(
        wb,
        name="new_server",
        pos=LAYOUT.new_server_initial_world_pos,
        rgba_chassis=[0.24, 0.26, 0.30, 1.0],
        server_cfg=server_cfg,
        handles_cfg=LAYOUT.handles,
    )

    # ---------------------- Actuators: planar base only -------------------
    root.actuator.add(
        "position",
        name=DataCenterAux.BASE_X,
        joint=DataCenterAux.BASE_X,
        kp=20000.0,
        kv=400.0,
        ctrllimited="true",
        ctrlrange=[-1.0, 1.0],
    )
    root.actuator.add(
        "position",
        name=DataCenterAux.BASE_Y,
        joint=DataCenterAux.BASE_Y,
        kp=20000.0,
        kv=400.0,
        ctrllimited="true",
        ctrlrange=[-1.0, 1.0],
    )
    root.actuator.add(
        "position",
        name=DataCenterAux.BASE_YAW,
        joint=DataCenterAux.BASE_YAW,
        kp=8000.0,
        kv=200.0,
        ctrllimited="true",
        ctrlrange=[-3.5, 3.5],  # ±π plus margin
    )

    # ---------------------- Welds (grasp + attachment) --------------------
    # Grasp-weld names must match `grasp_weld(side, i)` — `get_arm_handles`
    # resolves them via the same `<side>_grasp_cube<i>` formula. Both arms
    # register a weld onto each grippable so the bimanual sync grip can
    # activate the L+R pair on the same step.
    for side in ARM_PREFIXES:
        hand = f"{side}wrist_3_link"
        side_us = side.replace("/", "_")
        for i, obj_name in enumerate(GRIPPABLES):
            root.equality.add(
                "weld",
                name=f"{side_us}grasp_cube{i}",
                body1=hand,
                body2=obj_name,
                active="false",
                relpose=[0, 0, 0, 1, 0, 0, 0],
            )

    # Identity relpose at compile time; runtime re-seeds via
    # `welds.activate_attachment_weld`.
    for attachment in ATTACHMENTS:
        active_str = "true" if attachment.initially_active else "false"
        root.equality.add(
            "weld",
            name=attachment.name,
            body1=attachment.body_a,
            body2=attachment.body_b,
            active=active_str,
            relpose=[0, 0, 0, 1, 0, 0, 0],
        )

    # Exclude fragile contacts that would otherwise thrash.
    root.contact.add("exclude", body1="base_link", body2="left/base")
    root.contact.add("exclude", body1="base_link", body2="right/base")
    root.contact.add("exclude", body1="base_link", body2="new_server")
    root.contact.add("exclude", body1="rack_frame", body2="server")

    # XML round-trip so the returned (model, data) are owned by mujoco
    # directly, with no dm_control Physics lifetime concerns.
    xml_str = root.to_xml_string()
    assets = dict(root.get_assets())
    model = mujoco.MjModel.from_xml_string(xml_str, assets)
    data = mujoco.MjData(model)
    return model, data


# -----------------------------------------------------------------------------
# Scene handles (resolved once at runtime by apply_initial_state / task plan)
# -----------------------------------------------------------------------------


@dataclass
class _SceneIds:
    server_body_id: int
    new_server_body_id: int
    rack_body_id: int
    cart_body_id: int
    base_body_id: int
    attachment_eq: dict[str, int] = field(default_factory=dict)


def _resolve_scene_ids(model: mujoco.MjModel) -> _SceneIds:
    def body(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def eq(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    return _SceneIds(
        server_body_id=body("server"),
        new_server_body_id=body("new_server"),
        rack_body_id=body("rack_frame"),
        cart_body_id=body("cart_frame"),
        base_body_id=body("base_link"),
        attachment_eq={a.name: eq(a.name) for a in ATTACHMENTS},
    )


# -----------------------------------------------------------------------------
# Initial state
# -----------------------------------------------------------------------------


def apply_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
    *,
    start_phase: TaskPhase | None = None,
) -> None:
    """Reset to home arm pose, planar base at origin, and weld defaults.

    `start_phase=None` seeds from `HOME_ARM_Q_BY_SIDE` + `BASE_HOME_POSE`.
    Pass a `TaskPhase` to boot mid-demo using a hand-authored `_PhasePose`
    from `PHASE_HOMES`. Unauthored phases fall back to scene home with a
    courtesy print — better to boot home than crash on a typo.
    """
    del cube_body_ids
    mujoco.mj_resetData(model, data)
    ids = _resolve_scene_ids(model)

    phase_pose = PHASE_HOMES.get(start_phase) if start_phase is not None else None
    if start_phase is not None and phase_pose is None:
        print(
            f"[apply_initial_state] start_phase={start_phase.value!r} requested but "
            f"no _PhasePose captured for it — falling back to scene home."
        )
    arm_q_by_side = phase_pose.arm_q if phase_pose is not None else HOME_ARM_Q_BY_SIDE
    base_pose = phase_pose.base_pose if phase_pose is not None else BASE_HOME_POSE

    for arm in arms.values():
        home_q = arm_q_by_side[arm.side]
        for i, idx in enumerate(arm.arm_qpos_idx):
            data.qpos[idx] = home_q[i]
        data.ctrl[arm.act_arm_ids] = home_q
        # 2F-85's `fingers_actuator` is tendon-coupled — no per-finger qpos.
        data.ctrl[arm.act_gripper_id] = arm.gripper_closed
        for eq_id in arm.weld_ids:
            data.eq_active[eq_id] = 0
    for jname, value in zip(
        (DataCenterAux.BASE_X, DataCenterAux.BASE_Y, DataCenterAux.BASE_YAW),
        base_pose,
        strict=True,
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = value
        data.qvel[model.jnt_dofadr[jid]] = 0.0
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid >= 0:
            data.ctrl[aid] = value

    # Propagate qpos to xpos/xmat so the attachment welds below see the
    # actual initial body poses.
    mujoco.mj_forward(model, data)

    # Phase boot reads weld state from the contract, not from `_PhasePose`,
    # so a contract edit can't silently bit-rot a captured phase home.
    if phase_pose is not None and start_phase is not None:
        contract = next((c for c in PHASE_CONTRACTS if c.phase == start_phase), None)
        if contract is None:
            print(
                f"[apply_initial_state] start_phase={start_phase.value!r} has a "
                f"_PhasePose but no PhaseContract — using default attachments."
            )
            _apply_default_attachments(model, data, ids)
        else:
            _apply_contract_attachments(model, data, ids, contract.starts)
    else:
        _apply_default_attachments(model, data, ids)
    mujoco.mj_forward(model, data)


def _apply_default_attachments(model: mujoco.MjModel, data: mujoco.MjData, ids: _SceneIds) -> None:
    """Activate each ATTACHMENT per its `initially_active` declaration.
    Active welds re-seed their relpose from current body poses; the
    compile-time identity relpose would otherwise yank body_b to body_a."""
    for attachment in ATTACHMENTS:
        eq_id = ids.attachment_eq[attachment.name]
        if not attachment.initially_active:
            data.eq_active[eq_id] = 0
            continue
        activate_attachment_weld(
            model,
            data,
            eq_id,
            int(model.eq_obj1id[eq_id]),
            int(model.eq_obj2id[eq_id]),
        )


def _apply_contract_attachments(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: _SceneIds,
    starts: PhaseState,
) -> None:
    """Activate welds to match `starts.active_attachments` /
    `inactive_attachments`. Welds in neither set keep their default state,
    so adding a new attachment doesn't force every captured phase home to
    be re-authored."""
    active = {str(name) for name in starts.active_attachments}
    inactive = {str(name) for name in starts.inactive_attachments}
    for attachment in ATTACHMENTS:
        eq_id = ids.attachment_eq[attachment.name]
        if attachment.name in active:
            activate_attachment_weld(
                model,
                data,
                eq_id,
                int(model.eq_obj1id[eq_id]),
                int(model.eq_obj2id[eq_id]),
            )
        elif attachment.name in inactive:
            deactivate_weld(data, eq_id)
        elif attachment.initially_active:
            activate_attachment_weld(
                model,
                data,
                eq_id,
                int(model.eq_obj1id[eq_id]),
                int(model.eq_obj2id[eq_id]),
            )
        else:
            data.eq_active[eq_id] = 0


# -----------------------------------------------------------------------------
# Task plan
# -----------------------------------------------------------------------------


# Pre-flight IK tolerance: any `snap` call whose residual exceeds this
# aborts `make_task_plan` before viser starts. 2 cm is tight enough to
# catch near-misses (and force every target within fingertip distance) but
# loose enough not to fight currently-passing waypoints.
_IK_POSITION_TOL_M = 0.02


def _snap_factory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
) -> Callable[[Position3], tuple[np.ndarray, float]]:
    """Position-only IK closure, locking the planar base joints.

    The returned `snap(target)` raises `RuntimeError` when the residual
    exceeds `_IK_POSITION_TOL_M` — so an unreachable waypoint aborts plan
    construction instead of quietly shipping a 50 cm approach error into
    the runtime.
    """
    q_seed = {"current": IK_SEED_Q.copy()}
    call_count = {"n": 0}

    def snap(target_pos: Position3) -> tuple[np.ndarray, float]:
        call_count["n"] += 1
        q, err = solve_ik(
            model,
            data,
            arm,
            np.asarray(target_pos, dtype=float),
            orientation=PositionOnly(),
            seed_q=q_seed["current"],
            locked_joint_names=(
                DataCenterAux.BASE_X,
                DataCenterAux.BASE_Y,
                DataCenterAux.BASE_YAW,
            ),
            solver="daqp",
        )
        if err > _IK_POSITION_TOL_M:
            target = np.asarray(target_pos, dtype=float).tolist()
            raise RuntimeError(
                f"IK unreachable on {arm.side} (snap #{call_count['n']}): "
                f"err={err:.3f} m target={target} "
                f"(tol={_IK_POSITION_TOL_M} m). Adjust LAYOUT or move the "
                f"waypoint — do NOT raise the tolerance unless you have "
                f"a physics reason."
            )
        q_seed["current"] = q.copy()
        return q, err

    return snap


def make_task_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
) -> dict[ArmSide, list[Step]]:
    """Scripted four-action server swap (NEW_LAYOUT.md).

    Bimanual sync grip: server-handling steps have both arms moving in
    lockstep with their respective handle world positions, and the grasp
    welds onto the same server activate / deactivate at the same step
    boundary on both arms. Keeps the 12 kg server level and avoids the
    over-constraint glitches that plague a single-arm pin (where the
    unwelded arm's IK drift would bend the held server).

    Phase mapping: SETUP, REMOVE/STOW/RETRIEVE/INSTALL_*_SERVER (Actions
    A-D), RESET.
    """
    scripts: dict[ArmSide, list[Step]] = {side: [] for side in ARM_PREFIXES}

    def push_both(
        label: str,
        duration: float,
        phase: TaskPhase,
        aux: Mapping[Any, float] | None = None,
        gripper: GripperState = "closed",
    ) -> None:
        """Append a "both arms hold home" step to both timelines.

        Home pose is per-side (mirrored arms) — each side gets its own arm_q.
        """
        for side in ARM_PREFIXES:
            scripts[side].append(
                Step(
                    label=label,
                    arm_q=HOME_ARM_Q_BY_SIDE[side].copy(),
                    gripper=gripper,
                    duration=duration,
                    phase=phase,
                    aux_ctrl=aux,
                )
            )

    base_x_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DataCenterAux.BASE_X)
    base_y_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DataCenterAux.BASE_Y)
    base_yaw_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DataCenterAux.BASE_YAW)

    def seed_at_base(
        base_x: float = 0.0,
        base_y: float = 0.0,
        base_yaw: float = 0.0,
    ) -> tuple[
        Callable[[Position3], tuple[np.ndarray, float]],
        Callable[[Position3], tuple[np.ndarray, float]],
    ]:
        """Reset state, set base at the given planar pose, return fresh
        per-arm `snap` closures.

        IK targets are world-fixed; moving/rotating the base changes the
        reach geometry, so each phase seeds with its own base pose before
        running snap.
        """
        apply_initial_state(model, data, arms, cube_body_ids)
        data.qpos[model.jnt_qposadr[base_x_jnt]] = base_x
        data.qpos[model.jnt_qposadr[base_y_jnt]] = base_y
        data.qpos[model.jnt_qposadr[base_yaw_jnt]] = base_yaw
        mujoco.mj_forward(model, data)
        return (
            _snap_factory(model, data, arms[ArmSide.LEFT]),
            _snap_factory(model, data, arms[ArmSide.RIGHT]),
        )

    def base_aux(*, x: float, y: float, yaw: float) -> dict[str, float]:
        return dict(base_aux_targets(x=x, y=y, yaw=yaw))

    aux_at_rack = base_aux(x=0.0, y=0.0, yaw=0.0)
    aux_pre_extract = base_aux(x=BASE_PRE_EXTRACT, y=0.0, yaw=0.0)
    aux_at_cart = base_aux(x=BASE_AT_CART_X, y=BASE_AT_CART_Y, yaw=YAW_TO_CART)
    aux_at_insert = base_aux(x=BASE_AT_INSERT, y=0.0, yaw=0.0)

    # Both sides weld onto the SAME server body in a given action — that's
    # the bimanual sync grip mechanic.
    server_id = grippable_id("server")
    new_server_id = grippable_id("new_server")
    server_grasp = {side: grasp_weld(side, server_id) for side in ARM_PREFIXES}
    new_grasp = {side: grasp_weld(side, new_server_id) for side in ARM_PREFIXES}

    # === SETUP ============================================================
    push_both("home", 1.0, TaskPhase.SETUP, aux=aux_at_rack)
    push_both("settle at home", 1.0, TaskPhase.SETUP, aux=aux_at_rack)

    # === REMOVE_OLD_SERVER (arm-only at base = origin) ====================
    # Approach + grip + 10 cm pull-clear of rail detents. Base reversal is
    # the next phase (BACKUP_FROM_RACK).
    snap_l, snap_r = seed_at_base()
    handle_in_rack: dict[ArmSide, Position3] = {
        side: LAYOUT.handle_world_pos_in_rack(side) for side in ARM_PREFIXES
    }
    # Approach: 7 cm in front of each handle, same Y/Z. Gives the gripper
    # clear airspace before closing on the bezel cylinder.
    approach_in_rack = {
        side: handle_in_rack[side] + np.array([-0.07, 0.0, 0.0]) for side in ARM_PREFIXES
    }
    pull_clear = {side: handle_in_rack[side] + np.array([-0.10, 0.0, 0.0]) for side in ARM_PREFIXES}
    snap_by_side = {ArmSide.LEFT: snap_l, ArmSide.RIGHT: snap_r}
    q_approach_a = {side: snap_by_side[side](approach_in_rack[side])[0] for side in ARM_PREFIXES}
    q_at_handle_a = {side: snap_by_side[side](handle_in_rack[side])[0] for side in ARM_PREFIXES}
    q_pull_clear = {side: snap_by_side[side](pull_clear[side])[0] for side in ARM_PREFIXES}

    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"approach handle {side.rstrip('_/')}",
                q_approach_a[side],
                "open",
                1.6,
                phase=TaskPhase.REMOVE_OLD_SERVER,
                aux_ctrl=aux_at_rack,
            )
        )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"at handle {side.rstrip('_/')}",
                q_at_handle_a[side],
                "open",
                0.8,
                phase=TaskPhase.REMOVE_OLD_SERVER,
                aux_ctrl=aux_at_rack,
            )
        )
    # Activate BOTH grasp welds + deactivate SERVER_IN_RACK on a single step.
    # Splitting across L/R would leave a tick where only one arm grips while
    # the rack weld is gone — the "server falls" failure mode.
    scripts[ArmSide.LEFT].append(
        Step(
            "grip server L+R + free from rack",
            q_at_handle_a[ArmSide.LEFT],
            "closed",
            0.6,
            phase=TaskPhase.REMOVE_OLD_SERVER,
            aux_ctrl=aux_at_rack,
            attach_activate=(server_grasp[ArmSide.LEFT], server_grasp[ArmSide.RIGHT]),
            attach_deactivate=(AttachmentWeldName.SERVER_IN_RACK,),
        )
    )
    scripts[ArmSide.RIGHT].append(
        Step(
            "grip server R (sync)",
            q_at_handle_a[ArmSide.RIGHT],
            "closed",
            0.6,
            phase=TaskPhase.REMOVE_OLD_SERVER,
            aux_ctrl=aux_at_rack,
        )
    )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"pull clear of rails {side.rstrip('_/')}",
                q_pull_clear[side],
                "closed",
                1.2,
                phase=TaskPhase.REMOVE_OLD_SERVER,
                aux_ctrl=aux_at_rack,
            )
        )

    # === BACKUP_FROM_RACK (base-only) =====================================
    # Arms hold q_pull_clear so the welded server rides backward with the
    # chassis. No re-IK at the new base pose — that's the whole point of the
    # split: a base-only phase keeps the arm joints fixed.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"backup from rack {side.rstrip('_/')}",
                q_pull_clear[side],
                "closed",
                1.6,
                phase=TaskPhase.BACKUP_FROM_RACK,
                aux_ctrl=aux_pre_extract,
            )
        )

    # === TRAVERSE_TO_CART (base-only) =====================================
    # Same arm config; chassis rotates + translates to face the cart.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"traverse to cart {side.rstrip('_/')}",
                q_pull_clear[side],
                "closed",
                2.4,
                phase=TaskPhase.TRAVERSE_TO_CART,
                aux_ctrl=aux_at_cart,
            )
        )

    # === STOW_OLD_SERVER (arm-only at base = cart) ========================
    snap_l_cart, snap_r_cart = seed_at_base(
        base_x=BASE_AT_CART_X, base_y=BASE_AT_CART_Y, base_yaw=YAW_TO_CART
    )
    snap_cart = {ArmSide.LEFT: snap_l_cart, ArmSide.RIGHT: snap_r_cart}
    # 5 cm above rest pose so the final descent is a clean vertical settle.
    approach_bottom = {
        side: LAYOUT.handle_world_pos_on_cart_bottom(side) + np.array([0.0, 0.0, 0.05])
        for side in ARM_PREFIXES
    }
    place_bottom = {side: LAYOUT.handle_world_pos_on_cart_bottom(side) for side in ARM_PREFIXES}
    # Retract: 10 cm up + 10 cm back. "Back from cart" is -Y in world coords.
    retract_bottom = {
        side: place_bottom[side] + np.array([0.0, -0.10, 0.10]) for side in ARM_PREFIXES
    }
    q_above_bottom = {side: snap_cart[side](approach_bottom[side])[0] for side in ARM_PREFIXES}
    q_on_bottom = {side: snap_cart[side](place_bottom[side])[0] for side in ARM_PREFIXES}
    q_retract_bottom = {side: snap_cart[side](retract_bottom[side])[0] for side in ARM_PREFIXES}

    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"above tray {side.rstrip('_/')}",
                q_above_bottom[side],
                "closed",
                1.4,
                phase=TaskPhase.STOW_OLD_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )
    # Soft descent: 1.4 s for the last 5 cm. NEW_LAYOUT.md says "halve
    # descent velocity at 2 cm above tray" — approximated by slower
    # interpolation for the whole step.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"settle on tray {side.rstrip('_/')}",
                q_on_bottom[side],
                "closed",
                1.4,
                phase=TaskPhase.STOW_OLD_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )
    # Pin at the canonical cart pose so the new attachment doesn't snap to
    # an arm-tracking-drift offset. L declares the pose; R's matching step
    # just opens the gripper and drops its grasp.
    scripts[ArmSide.LEFT].append(
        Step(
            "release server L (pin on cart bottom)",
            q_on_bottom[ArmSide.LEFT],
            "open",
            0.6,
            phase=TaskPhase.STOW_OLD_SERVER,
            aux_ctrl=aux_at_cart,
            attach_activate_at=(
                (
                    AttachmentWeldName.SERVER_ON_CART_BOTTOM,
                    (
                        float(LAYOUT.old_server_stow_world_pos[0]),
                        float(LAYOUT.old_server_stow_world_pos[1]),
                        float(LAYOUT.old_server_stow_world_pos[2]),
                    ),
                    (1.0, 0.0, 0.0, 0.0),
                ),
            ),
            attach_deactivate=(server_grasp[ArmSide.LEFT], server_grasp[ArmSide.RIGHT]),
        )
    )
    scripts[ArmSide.RIGHT].append(
        Step(
            "release server R (sync)",
            q_on_bottom[ArmSide.RIGHT],
            "open",
            0.6,
            phase=TaskPhase.STOW_OLD_SERVER,
            aux_ctrl=aux_at_cart,
        )
    )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"retract from tray {side.rstrip('_/')}",
                q_retract_bottom[side],
                "open",
                1.0,
                phase=TaskPhase.STOW_OLD_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )

    # === RETRIEVE_NEW_SERVER (arm-only at base = cart) ====================
    approach_top = {
        side: LAYOUT.handle_world_pos_on_cart_top(side) + np.array([0.0, -0.05, 0.0])
        for side in ARM_PREFIXES
    }
    grasp_top = {side: LAYOUT.handle_world_pos_on_cart_top(side) for side in ARM_PREFIXES}
    lift_top = {side: grasp_top[side] + np.array([0.0, 0.0, 0.03]) for side in ARM_PREFIXES}

    q_above_top = {side: snap_cart[side](approach_top[side])[0] for side in ARM_PREFIXES}
    q_at_top = {side: snap_cart[side](grasp_top[side])[0] for side in ARM_PREFIXES}
    q_lift_top = {side: snap_cart[side](lift_top[side])[0] for side in ARM_PREFIXES}

    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"approach top shelf {side.rstrip('_/')}",
                q_above_top[side],
                "open",
                1.4,
                phase=TaskPhase.RETRIEVE_NEW_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"at top handle {side.rstrip('_/')}",
                q_at_top[side],
                "open",
                0.8,
                phase=TaskPhase.RETRIEVE_NEW_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )
    scripts[ArmSide.LEFT].append(
        Step(
            "grip new server L+R + free from cart",
            q_at_top[ArmSide.LEFT],
            "closed",
            0.6,
            phase=TaskPhase.RETRIEVE_NEW_SERVER,
            aux_ctrl=aux_at_cart,
            attach_activate=(new_grasp[ArmSide.LEFT], new_grasp[ArmSide.RIGHT]),
            attach_deactivate=(AttachmentWeldName.NEW_ON_CART_TOP,),
        )
    )
    scripts[ArmSide.RIGHT].append(
        Step(
            "grip new server R (sync)",
            q_at_top[ArmSide.RIGHT],
            "closed",
            0.6,
            phase=TaskPhase.RETRIEVE_NEW_SERVER,
            aux_ctrl=aux_at_cart,
        )
    )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"lift clear {side.rstrip('_/')}",
                q_lift_top[side],
                "closed",
                1.0,
                phase=TaskPhase.RETRIEVE_NEW_SERVER,
                aux_ctrl=aux_at_cart,
            )
        )

    # === TRAVERSE_TO_RACK (base-only) =====================================
    # Arms hold q_lift_top; chassis returns to origin facing +X.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"traverse to rack {side.rstrip('_/')}",
                q_lift_top[side],
                "closed",
                2.4,
                phase=TaskPhase.TRAVERSE_TO_RACK,
                aux_ctrl=aux_at_rack,
            )
        )

    # === ADVANCE_INTO_RACK (base-only) ====================================
    # Arms still hold q_lift_top; chassis steps forward 0.10 m so the held
    # server tracks straight at the rack opening before INSTALL extends.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"advance into rack {side.rstrip('_/')}",
                q_lift_top[side],
                "closed",
                1.4,
                phase=TaskPhase.ADVANCE_INTO_RACK,
                aux_ctrl=aux_at_insert,
            )
        )

    # === INSTALL_NEW_SERVER (arm-only at base = insert) ===================
    # Arms transition q_lift_top → q_inserted, sliding the server from the
    # carry pose into the rack slot (~7 cm forward, ~7.5 cm up). Pin + release.
    snap_l_inserted, snap_r_inserted = seed_at_base(base_x=BASE_AT_INSERT)
    snap_inserted = {ArmSide.LEFT: snap_l_inserted, ArmSide.RIGHT: snap_r_inserted}
    insert_handle = {side: LAYOUT.handle_world_pos_in_rack(side) for side in ARM_PREFIXES}
    q_inserted = {side: snap_inserted[side](insert_handle[side])[0] for side in ARM_PREFIXES}
    # Withdraw 7 cm back; base still at insert offset; server stays welded.
    withdraw_handle = {
        side: insert_handle[side] + np.array([-0.07, 0.0, 0.0]) for side in ARM_PREFIXES
    }
    q_withdraw = {side: snap_inserted[side](withdraw_handle[side])[0] for side in ARM_PREFIXES}

    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"push into rack {side.rstrip('_/')}",
                q_inserted[side],
                "closed",
                2.0,
                phase=TaskPhase.INSTALL_NEW_SERVER,
                aux_ctrl=aux_at_insert,
            )
        )
    scripts[ArmSide.LEFT].append(
        Step(
            "seat in rack L (pin + release L+R)",
            q_inserted[ArmSide.LEFT],
            "open",
            0.6,
            phase=TaskPhase.INSTALL_NEW_SERVER,
            aux_ctrl=aux_at_insert,
            attach_activate_at=(
                (
                    AttachmentWeldName.NEW_IN_RACK,
                    (
                        float(LAYOUT.server_world_pos_in_rack[0]),
                        float(LAYOUT.server_world_pos_in_rack[1]),
                        float(LAYOUT.server_world_pos_in_rack[2]),
                    ),
                    (1.0, 0.0, 0.0, 0.0),
                ),
            ),
            attach_deactivate=(new_grasp[ArmSide.LEFT], new_grasp[ArmSide.RIGHT]),
        )
    )
    scripts[ArmSide.RIGHT].append(
        Step(
            "seat in rack R (sync)",
            q_inserted[ArmSide.RIGHT],
            "open",
            0.6,
            phase=TaskPhase.INSTALL_NEW_SERVER,
            aux_ctrl=aux_at_insert,
        )
    )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"withdraw from rack {side.rstrip('_/')}",
                q_withdraw[side],
                "open",
                1.0,
                phase=TaskPhase.INSTALL_NEW_SERVER,
                aux_ctrl=aux_at_insert,
            )
        )

    # === RESET ============================================================
    # Arm-only "arms to home" then base-only "base to origin", so each step
    # honours the exclusivity invariant on its own.
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"arms to home {side.rstrip('_/')}",
                HOME_ARM_Q_BY_SIDE[side].copy(),
                "open",
                1.6,
                phase=TaskPhase.RESET,
                aux_ctrl=aux_at_insert,
            )
        )
    for side in ARM_PREFIXES:
        scripts[side].append(
            Step(
                f"base to origin {side.rstrip('_/')}",
                HOME_ARM_Q_BY_SIDE[side].copy(),
                "open",
                1.6,
                phase=TaskPhase.RESET,
                aux_ctrl=aux_at_rack,
            )
        )

    for side in ARM_PREFIXES:
        print(f"  [{side}] {len(scripts[side])} steps planned")

    apply_initial_state(model, data, arms, cube_body_ids)
    return scripts
