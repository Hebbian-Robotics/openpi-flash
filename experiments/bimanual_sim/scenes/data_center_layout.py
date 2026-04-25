"""Declarative geometry for the data-center scene.

Every dimension, offset, and derived anchor used by `scenes/data_center.py`
lives here as a nested frozen dataclass tree. The scene module imports a
single `LAYOUT = DataCenterLayout()` instance and reads positions via
`LAYOUT.rack.front_face_x`, `LAYOUT.server_world_pos_in_rack`, etc. — so
every spatial relationship is an attribute access that `grep` can find,
and changing one number (e.g. `LAYOUT.rack.center_x`) propagates through
every derived value automatically.

`__post_init__` on the sub-dataclasses and on `DataCenterLayout` asserts
the cross-component invariants the scene depends on (rack rests on the
floor, server fits inside the rack slot, lift targets within the TIAGo
joint range). Violations raise at import time — you never get a broken
scene into MuJoCo compilation.

No MuJoCo or numpy-side imports here beyond `np.array` for pose literals;
the module is cheap to import and unit-testable without a model in hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from robots.tiago import torso_world_pos_at_zero as tiago_torso_world_pos_at_zero
from scene_base import JointConfig, Position3

Half3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


# -----------------------------------------------------------------------------
# Sub-component layouts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tiago:
    """PAL TIAGo base + torso lift. `torso_world_pos_at_zero` is read from the
    upstream MJCF so the scene can't silently drift against a hardcoded copy."""

    torso_world_pos_at_zero: tuple[float, float, float] = field(
        default_factory=tiago_torso_world_pos_at_zero
    )
    lift_range: float = 0.35  # TIAGo's upstream torso_lift_joint range

    def torso_world_z(self, lift_qpos: float) -> float:
        """World z of torso_lift_link at a given lift qpos."""
        return self.torso_world_pos_at_zero[2] + lift_qpos


@dataclass(frozen=True)
class _ArmMount:
    """Piper arm attachment points in torso-local frame.

    `x = 0.0` puts the shoulder pivot at the torso centreline (middle
    of the cladding depth-wise), so the piper base link is fully inside
    the robot body silhouette and the first arm link is what extends
    outboard (user ask: "arms should stick out of the sides"). Any
    further back would start to clip into the TIAGo torso shell mesh
    and push the reach budget past Piper's ~0.55 m envelope; see
    `_Rack.center_x` docstring for the coupling.
    """

    x: float = 0.0
    y_abs: float = 0.30  # mounts at ±y_abs from torso centreline
    z: float = -0.15


@dataclass(frozen=True)
class _Cart:
    """Floor-standing service cart parked next to the robot.

    Generic data-centre service cart: two horizontal shelves (top +
    bottom) sized so 2-3 rack-unit servers fit on each, four vertical
    corner posts, push handle on the rear edge, four caster wheels
    underneath. New server starts on the top shelf; old server stows
    on the bottom shelf after extraction. Sized for "could carry any
    server, not just the one we're moving" — half_x and half_y bumped
    up so the deck reads as a real serviceman cart rather than a
    custom-fit pedestal for one chassis.
    """

    # Cart centre in world coordinates (front-left of the robot).
    center_x: float = 0.25  # bumped forward so longer cart still fits
    center_y: float = -0.65  # bumped further out so wider cart clears robot
    half_x: float = 0.45  # 90 cm long along the cart's long axis
    half_y: float = 0.30  # 60 cm deep (perpendicular to long axis)
    # Shelf z's are calibrated so the lift's puppet-mode range can drop
    # the carried server onto the bottom deck without the arms having
    # to reach further than a level pose. Server slot z=0.88, lift
    # drop = 0.23 m at full retract, so server-bottom lands at the
    # `bottom_shelf_z + shelf_thickness` ≈ 0.605 mark when lift=stow.
    top_shelf_z: float = 0.85  # top deck — new server starts here
    bottom_shelf_z: float = 0.60  # bottom deck — old server stows here
    shelf_thickness: float = 0.01
    post_half: float = 0.020  # 4 cm × 4 cm corner posts
    caster_radius: float = 0.035
    handle_height: float = 0.95  # top of the push handle (above top shelf)


@dataclass(frozen=True)
class _Bins:
    """Legacy on-torso bins — kept for backward-compat default values
    used by `new_server_initial_world_pos`. The data-center scene no
    longer renders bins; the new server starts on the cart's top shelf
    instead. Remove once `new_server_initial_world_pos` is retired.
    """

    half: Half3 = (0.24, 0.28, 0.08)
    local_x: float = 0.16
    new_local_z: float = 0.13
    old_local_z: float = -0.28
    wall_thickness: float = 0.01


@dataclass(frozen=True)
class _Rack:
    """Static 19" cabinet parked in front of the robot.

    Real-rack proportions: 60 cm wide × 60 cm deep × 130 cm tall (a
    half-height telco enclosure). `center_x = 0.70` puts the rack
    front face at x=0.40 — ~20 cm in front of the robot's base radius
    and within Piper's reach budget for cable-port targets at the
    front face (~0.50 m horizontal from the arm base). The server
    seats near the front of the rack interior so the arms don't have
    to reach 30 cm deep into the cabinet.
    """

    center_x: float = 0.70
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
        # Rack must rest on the floor (z = 0). A floating rack was the visible
        # symptom that prompted the user's "make it tall enough" report; this
        # guard fails at import rather than after viser renders.
        if abs(self.bottom_z) > 1e-6:
            raise ValueError(
                f"rack bottom at z={self.bottom_z:.4f}, must rest on floor (z=0): "
                f"set center_z = half[2] ({self.half[2]})"
            )


@dataclass(frozen=True)
class _Server:
    """Server chassis geometry (shared by `server` and `new_server` bodies)."""

    half: Half3 = (0.18, 0.22, 0.045)
    slot_z: float = 0.88  # world z of server centre when installed in rack
    mass: float = 0.5


@dataclass(frozen=True)
class _Ports:
    """Three ports across the server front face (power / network / fiber)."""

    local_x_depth: float = 0.01  # how far the port geom protrudes from server front
    y_offsets: tuple[float, float, float] = (-0.12, 0.0, +0.12)
    colors: tuple[Rgba, Rgba, Rgba] = (
        (0.90, 0.22, 0.22, 1.0),  # red — power
        (0.20, 0.78, 0.30, 1.0),  # green — network
        (0.22, 0.42, 0.90, 1.0),  # blue — fiber
    )


@dataclass(frozen=True)
class _Cables:
    """Cable rod + patch-panel placement.

    Cables emerge from a 1U-style patch panel mounted in the rack
    immediately below the server slot — the standard data-centre
    layout where switch ports sit one rack-unit below or above the
    server they connect to. Short rigid rods run vertically from
    each patch-panel port up to the matching server-front port, all
    inside the rack interior. Visible through the open front, and
    the arms unplug by gripping a connector and pulling forward.
    """

    n_seg: int = 14
    seg_len: float = 0.07
    # 18 mm-diameter cable — visible at HD render distance while still
    # in believable proportion to a 9 cm-tall server (~1:5). Earlier
    # stages: 24 mm read as fire hose; 14 mm was hard to see; 8 mm
    # vanished. 18 mm balances proportion with visibility.
    seg_radius: float = 0.009
    conn_len: float = 0.02
    # Patch panel sits this far below the server slot (typical 1U gap
    # between rack-mounted hardware) — short enough that cables don't
    # over-stretch when an arm pulls the connector forward.
    patch_panel_z_below_server: float = 0.07
    # Patch panel is a thin box on the rack interior front face. Sized
    # so its ports (at port_y_offsets, mirroring server ports below)
    # span the full rack interior width.
    patch_panel_half_x: float = 0.012  # 24 mm thick (front-to-back)
    patch_panel_half_y: float = 0.20  # 40 cm wide (mirrors rack interior)
    patch_panel_half_z: float = 0.025  # 5 cm tall (1U ≈ 4.4 cm)
    # Visible "cable plug tail" coming out of the connector toward the
    # server — reads as a stubby ferrule on the connector head.
    connector_stub_len: float = 0.04


@dataclass(frozen=True)
class _Arm:
    """Piper arm rest pose + IK seed.

    `home_q` is the visual rest pose shown before the task begins; `ik_seed_q`
    is what the differential-IK solver starts from. They're decoupled because
    DAQP fails to find a feasible solution when seeded from the compact home
    pose for forward-extending cable/server targets (user report: "IK refused
    to solve cable 1") but converges cleanly from the Menagerie forward-
    horizontal keyframe. The runner linearly interpolates from `home_q` to the
    first IK-solved `q` so the transition stays smooth.
    """

    # Wrist (joint5) at 0 keeps the gripper plates flat / parallel to
    # the world horizon — matches Menagerie's `agilex_piper/scene.xml`
    # home keyframe and avoids the askew "twisted gripper" look an
    # earlier joint5=1 rad value produced.
    home_q: JointConfig = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.5708, 0.0, 0.0, 0.0])
    )
    ik_seed_q: JointConfig = field(
        default_factory=lambda: np.array([0.0, 1.57, -1.3485, 0.0, 0.2, 0.0])
    )


@dataclass(frozen=True)
class _LiftTargets:
    """Task-plan lift qpos targets. All must lie in [0, tiago.lift_range]."""

    home: float = 0.05
    cables: float = 0.30  # rack cable-port height
    server: float = 0.28  # server-slot height (just below cables)
    stow: float = 0.05  # cart-bottom stow (low lift puts arm near cart shelves)
    pick_new: float = 0.05  # cart-top pickup (same low lift as stow)


# -----------------------------------------------------------------------------
# Top-level composition
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCenterLayout:
    """Single source of truth for `scenes/data_center.py`'s geometry.

    Cross-component derivations (anything that depends on more than one
    sub-component, e.g. a server's world position pulling from both
    `rack.front_face_x` and `server.half[0]`) live as `@property` on this
    top-level class — the sub-components stay ignorant of each other.
    """

    tiago: _Tiago = field(default_factory=_Tiago)
    arm_mount: _ArmMount = field(default_factory=_ArmMount)
    bins: _Bins = field(default_factory=_Bins)
    cart: _Cart = field(default_factory=_Cart)
    rack: _Rack = field(default_factory=_Rack)
    server: _Server = field(default_factory=_Server)
    ports: _Ports = field(default_factory=_Ports)
    cables: _Cables = field(default_factory=_Cables)
    arm: _Arm = field(default_factory=_Arm)
    lift: _LiftTargets = field(default_factory=_LiftTargets)

    # ---- Rack / server cross-derivations ----

    @property
    def server_front_x(self) -> float:
        """World x of the server's front face when installed in the rack.
        Equivalent to `rack.front_face_x`; kept as a named property because
        scene code reads "server-front" intent more clearly than "rack-front"
        at the call site."""
        return self.rack.front_face_x

    @property
    def server_center_x_in_rack(self) -> float:
        """World x of the server body centre when installed in the rack."""
        return self.rack.front_face_x + self.server.half[0]

    @property
    def server_world_pos_in_rack(self) -> Position3:
        return np.array([self.server_center_x_in_rack, 0.0, self.server.slot_z])

    @property
    def port_local_x_on_server(self) -> float:
        """Port geom's x offset in the server body's local frame (recessed)."""
        return -self.server.half[0] + self.ports.local_x_depth

    def port_world_pos(self, i: int) -> Position3:
        """World position of port `i`'s geom centre on the rack-mounted server."""
        return np.array(
            [
                self.server_center_x_in_rack + self.port_local_x_on_server,
                self.ports.y_offsets[i],
                self.server.slot_z,
            ]
        )

    def port_anchor_in_server_frame(self, i: int) -> tuple[float, float, float]:
        """Port anchor point in the server body's local frame, for the
        mjEQ_CONNECT `data` field of a port↔cable-connector equality."""
        return (self.port_local_x_on_server, self.ports.y_offsets[i], 0.0)

    # ---- Patch-panel + cable-anchor derivations ----
    # The patch panel is a 1U fixture mounted on the rack front rails,
    # one rack-unit below the server slot. Each cable anchors at the
    # patch-panel face directly under its corresponding server port, so
    # the rod runs straight up between them — short, visible, plausibly
    # data-centre-shaped.

    @property
    def patch_panel_world_pos(self) -> Position3:
        """World position of the patch-panel body's centre."""
        return np.array(
            [
                self.rack.front_face_x + self.cables.patch_panel_half_x,
                0.0,
                self.server.slot_z - self.cables.patch_panel_z_below_server,
            ]
        )

    def cable_anchor_world(self, cable_idx: int) -> Position3:
        """World anchor for cable `cable_idx` (0..2). Sits on the front
        face of the patch panel, directly below port `cable_idx` on the
        server above."""
        return np.array(
            [
                self.rack.front_face_x + 2 * self.cables.patch_panel_half_x,
                self.ports.y_offsets[cable_idx],
                self.server.slot_z - self.cables.patch_panel_z_below_server
                + self.cables.patch_panel_half_z,
            ]
        )

    @property
    def cable_max_len(self) -> float:
        """Cap on cable length (beyond the direct anchor→port run)."""
        return self.cables.n_seg * self.cables.seg_len + self.cables.conn_len

    # ---- Cart placements ----

    @property
    def new_server_initial_world_pos(self) -> Position3:
        """World pose the replacement server is spawned at: resting on the
        cart's top shelf, server centre `server.half[2]` above the
        shelf top surface."""
        return np.array(
            [
                self.cart.center_x,
                self.cart.center_y,
                self.cart.top_shelf_z + self.server.half[2],
            ]
        )

    @property
    def old_server_stow_world_pos(self) -> Position3:
        """World pose the OLD server gets stowed at after extraction:
        resting on the cart's bottom shelf."""
        return np.array(
            [
                self.cart.center_x,
                self.cart.center_y,
                self.cart.bottom_shelf_z + self.server.half[2],
            ]
        )

    # ---- Invariants ----

    def __post_init__(self) -> None:
        # Server must fit inside the rack slot.
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
        # Every lift target within TIAGo's actual joint range.
        for name in ("home", "cables", "server", "stow", "pick_new"):
            v = getattr(self.lift, name)
            if not 0.0 <= v <= self.tiago.lift_range:
                raise ValueError(
                    f"lift.{name}={v} outside TIAGo joint range [0, {self.tiago.lift_range}]"
                )
        # Port y offsets must fit within the server body's half-width.
        for i, y in enumerate(self.ports.y_offsets):
            if abs(y) > self.server.half[1]:
                raise ValueError(
                    f"port {i} y_offset={y} outside server half-width ±{self.server.half[1]}"
                )


LAYOUT = DataCenterLayout()


# -----------------------------------------------------------------------------
# Convenience re-exports for scene code that prefers flat names
# -----------------------------------------------------------------------------
# These keep `scenes/data_center.py` readable: instead of
# `LAYOUT.arm.home_q` spelled out at every call site, the scene module can
# import these at the top.

HOME_ARM_Q = LAYOUT.arm.home_q
IK_SEED_Q = LAYOUT.arm.ik_seed_q

_PORT_WORLD_POS: Callable[[int], Position3] = LAYOUT.port_world_pos
