"""Declarative geometry for the mobile-ALOHA + Piper indicator-check scene.

Two rows of 5 racks face each other across an aisle. Each rack is filled
with 21 x 2U servers; each server has a small green indicator light on the
front bezel except one alert server (4th rack of left row, middle slot)
whose light starts red. The robot drives in, yaws to face the alert,
gestures with both arms, and the light flips green via runtime RGBA update.

World frame: aisle along +X; +Y is the robot's left when facing +X. Robot
starts at world `(0, 0, 0)`.

Aisle width (2.10 m) is sized so that the ALOHA chassis can yaw in place
inside the aisle without the front-left corner clipping the left row's
rack fronts — the chassis silhouette is asymmetric (forward extent
0.927 m, side extent 0.435 m), giving a swept-corner radius of ~1.024 m
about chassis origin. With rack fronts at ±1.05 m, that leaves a thin but
collision-free 26 mm margin at the worst rotation angle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from arm_handles import ArmSide
from scene_base import JointConfig, Position3

Half3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]
Row = Literal["left", "right"]


# -----------------------------------------------------------------------------
# Sub-component layouts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rack:
    """Static 19" 42U rack. Width axis is along world X (the row direction);
    depth axis is along world Y (perpendicular to the row, into the rack).

    `half = (X, Y, Z)`: 0.30 wide, 0.50 deep, 1.00 tall — same outer
    dimensions as the live server-swap scene's rack, just rotated 90° about
    Z so multiple racks line up along the row.
    """

    half: Half3 = (0.30, 0.50, 1.00)
    wall_thickness: float = 0.012


@dataclass(frozen=True)
class _Server2U:
    """Single-block 2U server chassis. No bezel handles, no ports — this
    scene doesn't manipulate the server, only points at its indicator.

    Half-extents: 0.4826 m wide (19" rail-to-rail) x 0.700 m deep x 0.0889 m
    tall (one 2U rack-unit). The width axis aligns with the rack's width
    axis (world X for racks in either row).
    """

    half: Half3 = (0.2413, 0.35, 0.04445)
    rgba_chassis: Rgba = (0.16, 0.17, 0.19, 1.0)
    mass: float = 6.0  # irrelevant under the puppet's gravity=0; kept for realism


@dataclass(frozen=True)
class _IndicatorLight:
    """Small cylinder mounted on the server's front bezel, bottom-right
    corner. Cylinder axis is normal to the bezel face (so it points into
    the aisle for the viewer to see).

    `protrusion_total = 2 * half_length` — the cylinder centre sits flush
    with the bezel; half is buried in the chassis, half pokes into the
    aisle as a visible nub.
    """

    # Sized for visibility from the aisle, not realism — 36 mm diameter is
    # 3-4x a real datacenter LED, but at 1 m+ viewing distance through 21-slot
    # racks the smaller real-world size disappears against the chassis.
    radius: float = 0.018
    half_length: float = 0.008
    # Inset from the bezel's bottom-right corner so the light reads as
    # "near the corner" not "at the corner". The corner-edge insets dodge
    # the rack-side panels and the bottom edge gives the impression of a
    # status LED on a real server faceplate.
    inset_from_side: float = 0.040
    inset_from_bottom: float = 0.020
    rgba_green: Rgba = (0.20, 0.85, 0.30, 1.0)
    rgba_red: Rgba = (0.85, 0.20, 0.20, 1.0)


@dataclass(frozen=True)
class _Aisle:
    """Two parallel rows of 5 racks each, oriented so the robot drives along
    +X between them and turns to face one row.

    `row_centre_y_abs`: the |Y| of each row's rack centre. With this set to
    1.55 m and rack half-depth 0.50 m, rack fronts sit at Y = ±1.05 m and
    the aisle spans 2.10 m. That width is calibrated against ALOHA's
    swept-corner radius during in-place yaw (~1.024 m from chassis origin).
    Narrower would clip; wider just wastes floor.
    """

    row_centre_y_abs: float = 1.55
    rack_x_centres: tuple[float, ...] = (3.00, 3.70, 4.40, 5.10, 5.80)
    rack_centre_z: float = 1.00


@dataclass(frozen=True)
class _Servers:
    """How servers stack inside each rack. 21 x 2U at 0.0889 m pitch ≈ 1.867 m
    of stack height; the rack has 1.976 m of usable interior (2.00 m outer
    minus 2 x 0.012 m wall thickness), leaving ~5 cm of headroom.

    Slot index 10 is the middle slot — by convention the alert lives there.
    """

    n_per_rack: int = 21
    slot_pitch: float = 0.0889  # 2U = 88.9 mm


@dataclass(frozen=True)
class _Alert:
    """Identifies the one server whose indicator starts red. The user's
    spec calls for the 4th rack of the left row, middle slot."""

    row: Row = "left"
    rack_index: int = 3  # 0-indexed; 4th of 5
    slot_index: int = 10  # middle of 21


@dataclass(frozen=True)
class _Base:
    """Mobile ALOHA chassis starting pose (x, y, yaw) in world coordinates."""

    home_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class _Arm:
    """Piper bimanual rest pose + IK seed.

    Placeholder values: both sides start at the legacy
    `tiago_piper_server_cable_swap_layout.py` home_q (wrist joint5 = 0
    keeps the gripper plates parallel to the world horizon). The user
    will hand-author the real compressed home pose via teleop after the
    scene boots — `home_q` is per-side so left/right can diverge once the
    user captures separate poses.
    """

    home_q: Mapping[ArmSide, JointConfig] = field(
        default_factory=lambda: {
            ArmSide.LEFT: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ArmSide.RIGHT: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
    )
    ik_seed_q: JointConfig = field(
        default_factory=lambda: np.array([0.0, 1.57, -1.3485, 0.0, 0.2, 0.0])
    )


# -----------------------------------------------------------------------------
# Top-level composition
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorCheckLayout:
    """Single source of truth for `mobile_aloha_piper_indicator_check.py`."""

    rack: _Rack = field(default_factory=_Rack)
    server: _Server2U = field(default_factory=_Server2U)
    light: _IndicatorLight = field(default_factory=_IndicatorLight)
    aisle: _Aisle = field(default_factory=_Aisle)
    servers: _Servers = field(default_factory=_Servers)
    alert: _Alert = field(default_factory=_Alert)
    base: _Base = field(default_factory=_Base)
    arm: _Arm = field(default_factory=_Arm)

    # ---- Derived row geometry -------------------------------------------

    def row_centre_y(self, row: Row) -> float:
        """Y of a row's rack centre line. Left = +Y, right = -Y."""
        return self.aisle.row_centre_y_abs if row == "left" else -self.aisle.row_centre_y_abs

    def rack_front_face_y(self, row: Row) -> float:
        """Y of the rack-front face for `row`. Front faces the aisle, so
        for the left row the front is at the smaller Y (closer to 0)."""
        sign = 1.0 if row == "left" else -1.0
        return sign * (self.aisle.row_centre_y_abs - self.rack.half[1])

    def server_centre_y(self, row: Row) -> float:
        """Y of a server's centre with bezel flush against the rack front."""
        front = self.rack_front_face_y(row)
        sign = 1.0 if row == "left" else -1.0
        return front + sign * self.server.half[1]

    def server_centre_z(self, slot_index: int) -> float:
        """Z of a server centre for slot index in [0, n_per_rack)."""
        offset = (slot_index - (self.servers.n_per_rack - 1) / 2.0) * self.servers.slot_pitch
        return self.aisle.rack_centre_z + offset

    def rack_centre_x(self, rack_index: int) -> float:
        return self.aisle.rack_x_centres[rack_index]

    # ---- Indicator-light naming + position ------------------------------

    @staticmethod
    def light_geom_name(row: Row, rack_index: int, slot_index: int) -> str:
        """Stable MJCF geom name for an indicator light. Used by the scene
        to address the alert light from `Step.set_geom_rgba`. The 0-pad on
        rack/slot keeps lexicographic sort matching numeric order."""
        return f"light_{row}_r{rack_index}_s{slot_index:02d}"

    def light_world_pos(self, row: Row, rack_index: int, slot_index: int) -> Position3:
        """World position of an indicator light's CYLINDER CENTRE.

        Y is on the bezel face, offset outward by `half_length` so the
        cylinder pokes fully into the aisle (entirely visible, not half
        buried). X / Z select the bottom-right corner of the bezel from
        the aisle viewer's perspective: looking at the LEFT row (viewer
        faces +Y), "right" of the rack is world +X; for the RIGHT row
        it's world -X.
        """
        sign_x = 1.0 if row == "left" else -1.0
        sign_y_outward = -1.0 if row == "left" else 1.0
        local_x = sign_x * (self.server.half[0] - self.light.inset_from_side)
        local_z = -self.server.half[2] + self.light.inset_from_bottom
        bezel_y = self.rack_front_face_y(row)
        light_y = bezel_y + sign_y_outward * self.light.half_length
        return np.array(
            [
                self.rack_centre_x(rack_index) + local_x,
                light_y,
                self.server_centre_z(slot_index) + local_z,
            ]
        )

    # ---- Alert helpers --------------------------------------------------

    @property
    def alert_geom_name(self) -> str:
        return self.light_geom_name(self.alert.row, self.alert.rack_index, self.alert.slot_index)

    @property
    def alert_world_pos(self) -> Position3:
        return self.light_world_pos(self.alert.row, self.alert.rack_index, self.alert.slot_index)

    def alert_server_centre(self) -> Position3:
        """World centre of the alert server body (used as the click-target
        anchor for the bimanual reach phase)."""
        return np.array(
            [
                self.rack_centre_x(self.alert.rack_index),
                self.server_centre_y(self.alert.row),
                self.server_centre_z(self.alert.slot_index),
            ]
        )

    # ---- Click pose -----------------------------------------------------

    @property
    def click_chassis_xy(self) -> tuple[float, float]:
        """Chassis world (x, y) for the click. Y places chassis nose flush
        with the alert row's rack front when the chassis is yawed +π/2 (so
        chassis body +x = world +Y for the left row).

        `chassis_nose_offset = 0.927` is the ALOHA mesh's forward extent
        in body frame (CAD +y after the +π/2 body rotation in
        `robots/mobile_aloha.py`)."""
        chassis_nose_offset = 0.927
        sign = 1.0 if self.alert.row == "left" else -1.0
        rack_front = self.rack_front_face_y(self.alert.row)
        return self.rack_centre_x(self.alert.rack_index), rack_front - sign * chassis_nose_offset

    # ---- Invariants -----------------------------------------------------

    def __post_init__(self) -> None:
        # 1. Aisle has to be wide enough for ALOHA's swept-corner radius
        #    (~1.024 m from chassis origin) to clear the opposing rack row
        #    during in-place yaw. Anything tighter clips; the constant
        #    here is the chassis-corner distance baked into
        #    `robots/mobile_aloha.py`.
        max_yaw_swept_radius = 1.024
        margin = self.aisle.row_centre_y_abs - self.rack.half[1] - max_yaw_swept_radius
        if margin < 0.01:
            raise ValueError(
                f"aisle too narrow for ALOHA in-place yaw: row_centre_y_abs="
                f"{self.aisle.row_centre_y_abs} leaves only {margin * 1000:.0f} mm of "
                f"clearance against swept-corner radius {max_yaw_swept_radius}. "
                f"Bump `_Aisle.row_centre_y_abs` to ≥ "
                f"{self.rack.half[1] + max_yaw_swept_radius + 0.02:.3f}."
            )
        # 2. Server stack fits inside the rack interior.
        stack_height = self.servers.n_per_rack * self.servers.slot_pitch
        interior_height = 2.0 * (self.rack.half[2] - self.rack.wall_thickness)
        if stack_height > interior_height:
            raise ValueError(
                f"21 x 2U server stack ({stack_height:.3f} m) exceeds rack "
                f"interior ({interior_height:.3f} m)"
            )
        # 3. Alert indices in range.
        if not 0 <= self.alert.rack_index < len(self.aisle.rack_x_centres):
            raise ValueError(f"alert.rack_index={self.alert.rack_index} out of range")
        if not 0 <= self.alert.slot_index < self.servers.n_per_rack:
            raise ValueError(f"alert.slot_index={self.alert.slot_index} out of range")
        # 4. Server bezel sits inside the rack X extent (servers narrower
        #    than racks).
        if self.server.half[0] > self.rack.half[0] - 0.01:
            raise ValueError(
                f"server too wide for rack: server.half[0]={self.server.half[0]} "
                f"vs rack.half[0]-margin={self.rack.half[0] - 0.01}"
            )


LAYOUT = IndicatorCheckLayout()


# -----------------------------------------------------------------------------
# Convenience re-exports (parallel to the live server-swap layout)
# -----------------------------------------------------------------------------

HOME_ARM_Q_BY_SIDE: Mapping[ArmSide, JointConfig] = LAYOUT.arm.home_q
IK_SEED_Q = LAYOUT.arm.ik_seed_q
BASE_HOME_POSE: tuple[float, float, float] = LAYOUT.base.home_pose
ALERT_LIGHT_GEOM_NAME: str = LAYOUT.alert_geom_name
