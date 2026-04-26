"""Declarative geometry for the new (mobile-aloha + UR10e) data-center scene.

Mirrors `scenes/data_center_tiago_layout.py` for everything that's
robot-agnostic — rack, cart, servers, ports, cables — but swaps the
mobile-base specifics:

* Mobile-base geometry and UR10e mount sites live in
  `robots/mobile_aloha.py`, which is the source of truth for the CAD
  mesh's frame-specific constants.
* No `_LiftTargets` dataclass — the UR10e does all z-motion via its
  own joints, so there's no aux lift to interpolate.
* `_Arm.home_q` / `ik_seed_q` retuned for UR10e's joint convention
  (6 named revolute joints).

Rack dims, server slot z, port positions, cable derivations, and cart
shelf heights all stay verbatim from the legacy layout — the world-
frame IK targets are robot-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from arm_handles import ArmSide
from scene_base import JointConfig, Position3

Half3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


# -----------------------------------------------------------------------------
# Sub-component layouts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cart:
    """Floor-standing service cart parked next to the robot.

    Two shelves sized to hold 2-3 servers each. Same dimensions as the
    legacy scene — robot-agnostic since the cart sits on the floor at
    fixed world coords.
    """

    center_x: float = 0.25
    center_y: float = -0.65
    half_x: float = 0.45
    half_y: float = 0.30
    top_shelf_z: float = 0.85
    bottom_shelf_z: float = 0.60
    shelf_thickness: float = 0.01
    post_half: float = 0.020
    caster_radius: float = 0.035
    handle_height: float = 0.95


@dataclass(frozen=True)
class _Bins:
    """Legacy on-torso bins — unused by the mobile-aloha scene but
    retained for backward-compat (the legacy `_Tiago` scene's
    `new_server_initial_world_pos` referenced these). The new scene
    sources `new_server_initial_world_pos` from `_Cart.top_shelf_z`
    directly so these are dead in this layout — kept only because
    the old layout's typed shape used them; safe to delete in a
    follow-up cleanup."""

    half: Half3 = (0.24, 0.28, 0.08)
    local_x: float = 0.16
    new_local_z: float = 0.13
    old_local_z: float = -0.28
    wall_thickness: float = 0.01


@dataclass(frozen=True)
class _Rack:
    """Static 19" cabinet parked in front of the robot.

    Centre pushed to world x = +1.30 m (front face at +1.00 m) so the
    Stanford ALOHA body's forward boom (which extends to world x =
    +0.926 m at z ≈ 1.0 m) clears the rack front face by ~74 mm. The
    UR10e arms mounted at body centre then reach forward ~1.0 m to
    grip cables / extract servers — well within UR10e's 1.3 m reach
    envelope, and matches the real Stanford ALOHA "drives up to a
    workbench" use case.
    """

    center_x: float = 1.30
    half: Half3 = (0.30, 0.30, 0.65)
    center_z: float = 0.65
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
    """Server chassis geometry (shared by `server` and `new_server`)."""

    half: Half3 = (0.18, 0.22, 0.045)
    slot_z: float = 0.88
    mass: float = 0.5


@dataclass(frozen=True)
class _Ports:
    """Three ports across the server front face (power / network / fiber)."""

    local_x_depth: float = 0.01
    y_offsets: tuple[float, float, float] = (-0.12, 0.0, +0.12)
    colors: tuple[Rgba, Rgba, Rgba] = (
        (0.90, 0.22, 0.22, 1.0),  # red — power
        (0.20, 0.78, 0.30, 1.0),  # green — network
        (0.22, 0.42, 0.90, 1.0),  # blue — fiber
    )


@dataclass(frozen=True)
class _Cables:
    """Cable rod + patch-panel placement. Robot-agnostic."""

    n_seg: int = 14
    seg_len: float = 0.07
    seg_radius: float = 0.009
    conn_len: float = 0.02
    patch_panel_z_below_server: float = 0.07
    patch_panel_half_x: float = 0.012
    patch_panel_half_y: float = 0.20
    patch_panel_half_z: float = 0.025
    connector_stub_len: float = 0.04


@dataclass(frozen=True)
class _Arm:
    """UR10e rest pose + IK seed.

    UR10e's 6 named joints (shoulder_pan, shoulder_lift, elbow,
    wrist_1, wrist_2, wrist_3) — angles in radians.

    `home_q` is keyed by `ArmSide` because the two arms mount at
    mirrored chassis positions (left at body Y = +0.295, right at
    body Y = -0.295) — a single shared joint vector produces non-
    mirrored visual poses, so each side gets its own. The values
    were captured live via teleop's "Print home_q for layout"
    button after hand-posing the arms into a symmetric stow.

    `ik_seed_q` stays a single shared vector — it's the DAQP starting
    hint, not a visible pose, and the same forward-reaching basin
    works for both arms after solve-time mirroring of the seed if
    needed.
    """

    home_q: Mapping[ArmSide, JointConfig] = field(
        default_factory=lambda: {
            ArmSide.LEFT: np.array([-3.14, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            # Same joint vector as LEFT — visualises what "identical
            # joint states" looks like. Because the two arms mount at
            # mirrored chassis Y positions, this is NOT a mirror-image
            # pose; the right arm will match left's joint angles
            # numerically but face the opposite chassis side.
            ArmSide.RIGHT: np.array([-3.14, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
        }
    )
    # IK seed = half-extended forward pose: shoulder_lift = -45°,
    # elbow = +45° puts the TCP roughly 0.85 m forward and 0.42 m up
    # from the base — close to where rack/cart targets live, so DAQP
    # converges from the first snap. The folded `home_q` (shoulder
    # lift = -90°, elbow = +90°) leaves the TCP ~1 m above the base
    # which is too far from the basin for rack reaches now that the
    # bases are pulled back to the body x = 0.226.
    ik_seed_q: JointConfig = field(
        default_factory=lambda: np.array([0.0, -0.7854, 0.7854, 0.0, -1.5708, 0.0])
    )


# -----------------------------------------------------------------------------
# Top-level composition
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCenterLayout:
    """Single source of truth for `scenes/data_center.py` geometry.

    Same shape as the legacy layout, with robot-base geometry kept in
    `robots/mobile_aloha.py` and `_LiftTargets` removed. World-frame
    derived properties (server slot pose, port world positions, cart
    shelf poses, patch panel anchors) are robot-agnostic and identical
    to the legacy layout.
    """

    bins: _Bins = field(default_factory=_Bins)
    cart: _Cart = field(default_factory=_Cart)
    rack: _Rack = field(default_factory=_Rack)
    server: _Server = field(default_factory=_Server)
    ports: _Ports = field(default_factory=_Ports)
    cables: _Cables = field(default_factory=_Cables)
    arm: _Arm = field(default_factory=_Arm)

    # ---- Rack / server cross-derivations (verbatim from legacy) ----

    @property
    def server_front_x(self) -> float:
        return self.rack.front_face_x

    @property
    def server_center_x_in_rack(self) -> float:
        return self.rack.front_face_x + self.server.half[0]

    @property
    def server_world_pos_in_rack(self) -> Position3:
        return np.array([self.server_center_x_in_rack, 0.0, self.server.slot_z])

    @property
    def port_local_x_on_server(self) -> float:
        return -self.server.half[0] + self.ports.local_x_depth

    def port_world_pos(self, i: int) -> Position3:
        return np.array(
            [
                self.server_center_x_in_rack + self.port_local_x_on_server,
                self.ports.y_offsets[i],
                self.server.slot_z,
            ]
        )

    def port_anchor_in_server_frame(self, i: int) -> tuple[float, float, float]:
        return (self.port_local_x_on_server, self.ports.y_offsets[i], 0.0)

    # ---- Patch-panel + cable-anchor derivations ----

    @property
    def patch_panel_world_pos(self) -> Position3:
        return np.array(
            [
                self.rack.front_face_x + self.cables.patch_panel_half_x,
                0.0,
                self.server.slot_z - self.cables.patch_panel_z_below_server,
            ]
        )

    def cable_anchor_world(self, cable_idx: int) -> Position3:
        return np.array(
            [
                self.rack.front_face_x + 2 * self.cables.patch_panel_half_x,
                self.ports.y_offsets[cable_idx],
                self.server.slot_z
                - self.cables.patch_panel_z_below_server
                + self.cables.patch_panel_half_z,
            ]
        )

    @property
    def cable_max_len(self) -> float:
        return self.cables.n_seg * self.cables.seg_len + self.cables.conn_len

    # ---- Cart placements ----

    @property
    def new_server_initial_world_pos(self) -> Position3:
        return np.array(
            [
                self.cart.center_x,
                self.cart.center_y,
                self.cart.top_shelf_z + self.server.half[2],
            ]
        )

    @property
    def old_server_stow_world_pos(self) -> Position3:
        return np.array(
            [
                self.cart.center_x,
                self.cart.center_y,
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
        for i, y in enumerate(self.ports.y_offsets):
            if abs(y) > self.server.half[1]:
                raise ValueError(
                    f"port {i} y_offset={y} outside server half-width ±{self.server.half[1]}"
                )


LAYOUT = DataCenterLayout()


# -----------------------------------------------------------------------------
# Convenience re-exports
# -----------------------------------------------------------------------------

HOME_ARM_Q_BY_SIDE: Mapping[ArmSide, JointConfig] = LAYOUT.arm.home_q
IK_SEED_Q = LAYOUT.arm.ik_seed_q

_PORT_WORLD_POS: Callable[[int], Position3] = LAYOUT.port_world_pos
