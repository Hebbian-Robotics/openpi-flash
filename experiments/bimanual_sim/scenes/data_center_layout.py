"""Declarative geometry for the mobile-ALOHA + UR10e + 2F-85 data-center
scene, sized per `experiments/bimanual_sim/NEW_LAYOUT.md`.

Single source of truth for rack / server / cart / bezel-handle world
coordinates. Robot chassis geometry lives in `robots/mobile_aloha.py`.

The scene narrative is the four-action server swap:

    A. Extract old server from rack
    B. Place old server on trolley bottom tray
    C. Pick new server from trolley top shelf
    D. Insert new server back into rack
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from arm_handles import ArmSide
from scene_base import JointConfig, Position3, TaskPhase

Half3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


# -----------------------------------------------------------------------------
# Sub-component layouts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cart:
    """Mobile service trolley parked to the robot's right.

    Sized to a generic two-shelf datacenter utility cart (~36" x 24" x
    36" — Uline / Lakeside / Rubbermaid form factor for general service
    work, as opposed to a purpose-built ServerLIFT with hydraulics).

    Footprint 0.90 m x 0.60 m centered at world (0.30, 0.90). The 0.60 m
    Y span gives 6 cm clearance per side around a 0.48 m wide server.
    Bottom tray Z = 0.25 m, top shelf Z = 0.85 m — 60 cm clearance is
    plenty for a 4U server with hand room.

    The bottom tray is modelled as a static low shelf — real cart trays
    slide forward but the demo doesn't require it.
    """

    center_x: float = 0.30
    center_y: float = 0.90
    half_x: float = 0.45
    half_y: float = 0.30
    top_shelf_z: float = 0.85
    bottom_shelf_z: float = 0.25
    shelf_thickness: float = 0.01
    post_half: float = 0.020
    caster_radius: float = 0.035
    handle_height: float = 0.97


@dataclass(frozen=True)
class _Rack:
    """Static 19" 42U rack standing in front of the robot.

    Outer dimensions per NEW_LAYOUT §1: 0.60 m wide x 1.00 m deep x
    2.00 m tall, with the front face at world X = 0.60 m. That puts
    the rack center at X = 1.10 m (front + half-depth = 0.60 + 0.50)
    and Z = 1.00 m, so the server slot at world Z = 1.00 m sits
    exactly mid-rack — natural for both arm reach (UR10e arm bases at
    Z = 1.00 m) and the wrist camera's view angle.

    The robot at world origin facing +X, with a chassis ~0.40 m deep,
    leaves ~20 cm front-edge-to-rack-face clearance — matching the
    spec's "front edge 20 cm from rack face" callout.
    """

    center_x: float = 1.10
    half: Half3 = (0.50, 0.30, 1.00)
    center_z: float = 1.00
    wall_thickness: float = 0.012

    @property
    def front_face_x(self) -> float:
        return self.center_x - self.half[0]

    @property
    def back_face_x(self) -> float:
        return self.center_x + self.half[0]

    @property
    def bottom_z(self) -> float:
        return self.center_z - self.half[2]

    @property
    def top_z(self) -> float:
        return self.center_z + self.half[2]

    @property
    def side_y_pos(self) -> float:
        return self.half[1]

    @property
    def side_y_neg(self) -> float:
        return -self.half[1]

    def __post_init__(self) -> None:
        if abs(self.bottom_z) > 1e-6:
            raise ValueError(
                f"rack bottom at z={self.bottom_z:.4f}, must rest on floor (z=0): "
                f"set center_z = half[2] ({self.half[2]})"
            )


@dataclass(frozen=True)
class _Server:
    """Server chassis geometry per NEW_LAYOUT §1.

    Half-extents in (X depth, Y width, Z height): the spec's full
    dimensions are 0.70 x 0.48 x 0.09 m. 12 kg mass — heavy enough
    that the demo motion has to be bimanual (one UR10e wrist torque
    rating ≈ 12.5 kg payload at standard reach, so a 12 kg load at
    full extension would saturate a single arm).
    """

    half: Half3 = (0.35, 0.24, 0.045)
    slot_z: float = 1.00
    mass: float = 12.0


@dataclass(frozen=True)
class _BezelHandles:
    """Front-bezel handle pair on each server.

    In the rack-inserted pose, handles sit on the server's -X face
    (the bezel) at Y offsets ±0.12 m from the server centerline. The
    grippers approach along -X (TCP normal pointing into rack). When
    the server is on the cart, handles are still at ±0.12 m from
    server centerline along Y, but the grippers approach from -Y
    (chassis facing +Y to reach the cart).
    """

    y_offset_abs: float = 0.12
    """Half the handle separation (24 cm bezel-to-bezel)."""

    handle_radius: float = 0.012
    """Visual cylinder radius — small enough to fit between the
    2F-85's 85 mm finger gap."""

    handle_length: float = 0.04
    """Length of the handle cylinder normal to the bezel face."""


@dataclass(frozen=True)
class _Base:
    """Mobile ALOHA chassis starting pose (x, y, yaw) in world coordinates.

    Capture a new pose via teleop's "Print base pose for layout" — that
    emits a copy-pasteable line for this default factory.
    """

    home_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Arm:
    """UR10e rest pose + IK seed.

    UR10e's 6 named joints (shoulder_pan, shoulder_lift, elbow,
    wrist_1, wrist_2, wrist_3) — angles in radians.

    `home_q` is keyed by `ArmSide` because the two arms mount at
    mirrored chassis positions (left at body Y = +0.295, right at
    body Y = -0.295) — a single shared joint vector produces non-
    mirrored visual poses, so each side gets its own.

    `ik_seed_q` stays a single shared vector — it's the DAQP starting
    hint, not a visible pose, and the same forward-reaching basin
    works for both arms.
    """

    home_q: Mapping[ArmSide, JointConfig] = field(
        default_factory=lambda: {
            ArmSide.LEFT: np.array([-3.14, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            ArmSide.RIGHT: np.array([-3.14, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
        }
    )
    ik_seed_q: JointConfig = field(
        default_factory=lambda: np.array([0.0, -0.7854, 0.7854, 0.0, -1.5708, 0.0])
    )


@dataclass(frozen=True)
class _PhasePose:
    """Hand-authored start pose for a specific TaskPhase.

    Captured via teleop's "Save current as start of [phase]" button
    and dumped via "Print phase homes for layout". Lets the user boot
    the scene mid-demo by passing `--start-phase REMOVE_OLD_SERVER`
    instead of always starting at scene home — useful for iterating on
    just one phase without replaying the whole pickup sequence.
    """

    arm_q: Mapping[ArmSide, JointConfig]
    base_pose: tuple[float, float, float]


@dataclass(frozen=True)
class _PhaseHomes:
    """Per-phase saved start poses.

    Default factory ships an empty mapping; the user populates it from
    teleop captures. Empty is a valid state — `apply_initial_state`
    falls back to the scene home pose if `start_phase` isn't in the
    map, so unauthored phases just behave like the default boot.
    """

    by_phase: Mapping[TaskPhase, _PhasePose] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Top-level composition
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCenterLayout:
    """Single source of truth for `scenes/data_center.py` geometry.

    Robot-base geometry lives in `robots/mobile_aloha.py`. World-frame
    derived properties (server slot pose, bezel handle positions,
    cart shelf poses) are computed from the dataclass fields below.
    """

    cart: _Cart = field(default_factory=_Cart)
    rack: _Rack = field(default_factory=_Rack)
    server: _Server = field(default_factory=_Server)
    handles: _BezelHandles = field(default_factory=_BezelHandles)
    base: _Base = field(default_factory=_Base)
    arm: _Arm = field(default_factory=_Arm)
    phase_homes: _PhaseHomes = field(default_factory=_PhaseHomes)

    # ---- Rack / server cross-derivations ----

    @property
    def server_front_x(self) -> float:
        return self.rack.front_face_x

    @property
    def server_center_x_in_rack(self) -> float:
        return self.rack.front_face_x + self.server.half[0]

    @property
    def server_world_pos_in_rack(self) -> Position3:
        return np.array([self.server_center_x_in_rack, 0.0, self.server.slot_z])

    # In rack: bezel face at X = front_face_x = 0.60 m, handles at Y = ±0.12 m,
    # Z = slot_z. World-frame IK targets for Action A grasp / Action D release.

    def handle_world_pos_in_rack(self, side: ArmSide) -> Position3:
        y = self.handles.y_offset_abs if side is ArmSide.RIGHT else -self.handles.y_offset_abs
        return np.array([self.rack.front_face_x, y, self.server.slot_z])

    def handle_local_pos_on_server(self, side: ArmSide) -> tuple[float, float, float]:
        """Handle position in the server's local frame — used to attach
        the handle geom to the server body."""
        y = self.handles.y_offset_abs if side is ArmSide.RIGHT else -self.handles.y_offset_abs
        return (-self.server.half[0], y, 0.0)

    # Cart center is at (0.30, 0.90), but the server rests at (0.30, 0.80) —
    # offset -0.10 m in Y so it occupies the robot-side half of the cart
    # footprint, leaving the trolley handle / back-rail on the +Y side
    # (matches NEW_LAYOUT "Old server rest position: Center (30, 80, 29.5)").

    SERVER_ON_CART_Y_OFFSET: float = -0.10

    @property
    def server_on_cart_x(self) -> float:
        return self.cart.center_x

    @property
    def server_on_cart_y(self) -> float:
        return self.cart.center_y + self.SERVER_ON_CART_Y_OFFSET

    @property
    def new_server_initial_world_pos(self) -> Position3:
        """New server resting on the cart's TOP shelf (Action C source)."""
        return np.array(
            [
                self.server_on_cart_x,
                self.server_on_cart_y,
                self.cart.top_shelf_z + self.server.half[2],
            ]
        )

    @property
    def old_server_stow_world_pos(self) -> Position3:
        """Old server's rest position on the cart's BOTTOM tray (Action B target)."""
        return np.array(
            [
                self.server_on_cart_x,
                self.server_on_cart_y,
                self.cart.bottom_shelf_z + self.server.half[2],
            ]
        )

    def handle_world_pos_on_cart_top(self, side: ArmSide) -> Position3:
        """Handle position when the new server sits on the cart top
        shelf — Action C grip target."""
        y_offset = (
            self.handles.y_offset_abs if side is ArmSide.RIGHT else -self.handles.y_offset_abs
        )
        return np.array(
            [
                self.server_on_cart_x,
                self.server_on_cart_y + y_offset,
                self.cart.top_shelf_z + self.server.half[2],
            ]
        )

    def handle_world_pos_on_cart_bottom(self, side: ArmSide) -> Position3:
        """Handle position when the old server is placed on the bottom
        tray — Action B release target."""
        y_offset = (
            self.handles.y_offset_abs if side is ArmSide.RIGHT else -self.handles.y_offset_abs
        )
        return np.array(
            [
                self.server_on_cart_x,
                self.server_on_cart_y + y_offset,
                self.cart.bottom_shelf_z + self.server.half[2],
            ]
        )

    # ---- Invariants ----

    def __post_init__(self) -> None:
        if self.server.half[0] > self.rack.half[0] - 0.02:
            raise ValueError(
                f"server too deep for rack slot: "
                f"server.half[0]={self.server.half[0]} vs rack.half[0]-margin="
                f"{self.rack.half[0] - 0.02}"
            )
        if self.server.half[1] > self.rack.half[1] - 0.02:
            raise ValueError(
                f"server too wide for rack slot: "
                f"server.half[1]={self.server.half[1]} vs rack.half[1]-margin="
                f"{self.rack.half[1] - 0.02}"
            )
        if not self.rack.bottom_z <= self.server.slot_z <= self.rack.top_z:
            raise ValueError(
                f"server slot z={self.server.slot_z} outside rack vertical span "
                f"[{self.rack.bottom_z:.3f}, {self.rack.top_z:.3f}]"
            )
        if self.handles.y_offset_abs > self.server.half[1]:
            raise ValueError(
                f"bezel handle y_offset={self.handles.y_offset_abs} outside "
                f"server half-width ±{self.server.half[1]}"
            )


LAYOUT = DataCenterLayout()


# -----------------------------------------------------------------------------
# Convenience re-exports
# -----------------------------------------------------------------------------

HOME_ARM_Q_BY_SIDE: Mapping[ArmSide, JointConfig] = LAYOUT.arm.home_q
IK_SEED_Q = LAYOUT.arm.ik_seed_q
BASE_HOME_POSE: tuple[float, float, float] = LAYOUT.base.home_pose
PHASE_HOMES: Mapping[TaskPhase, _PhasePose] = LAYOUT.phase_homes.by_phase
