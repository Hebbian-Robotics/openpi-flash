"""Data-center server-swap scene.

A bimanual Piper pair rides on a lift carriage attached to a (static) mobile
base parked in front of a rack. The task plan runs through a full server swap:

    home -> lift to cables
    -> unplug cable 1 (left arm)
    -> unplug cable 2 (right arm)
    -> unplug cable 3 (left arm)
    -> both arms grab server handle rails, pull server out of rack
    -> stow server in lower compartment (old_bin)
    -> both arms grab replacement from upper compartment (new_bin)
    -> slide new server into rack
    -> replug cables 1, 2, 3
    -> retract lift, home both arms

Novel in this scene compared to sink_bimanual:
  * Pipers attached inside a nested body (lift_carriage) so a slide joint
    raises/lowers arms + onboard compartments together.
  * Scene-owned actuator ("lift") controlled via Step.aux_ctrl; IK freezes it
    via `locked_joint_names=("lift",)` so the solver uses only the arm chain.
  * Cable objects: 7-segment ball-jointed chains anchored to the rack,
    terminating in a grippable connector block. Grasp the head to pull out.
  * Three port shapes + colors (power/net/fiber) for visual distinction.
  * Three cameras (top + 2 wrist) declared in MJCF for future EGL capture.

Type-safety conventions:
  * Attachment welds live in one registry (`ATTACHMENTS`) iterated by
    `build_spec`, `_resolve_scene_ids`, and `apply_initial_state` — no more
    three-way duplication of the same name list with the same active flags.
  * Weld names are `AttachmentWeldName` (StrEnum) so typos are caught by ty.
  * Aux actuator keys are `DataCenterAux` (StrEnum) for the same reason.
  * Grippable objects are addressed by `CubeID` constructed through
    `grippable_id("cable1_connector")` — the bounds check and the name-to-index
    translation happen in one place, instead of scattered `_index_of(f"...")`
    string-building at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np
from dm_control import mjcf

from arm_handles import ArmHandles, ArmSide
from cameras import CameraRole
from ik import PositionOnly, solve_ik
from paths import D405_MESH_STL, D435I_XML
from robots.piper import load_piper
from robots.tiago import load_tiago
from scene_base import CubeID, Position3, Step, make_cube_id
from scene_check import (
    AttachmentConstraint,
    CameraInvariant,
    ConnectAttachment,
    FixedCameraInvariant,
    TargetingCameraInvariant,
    WeldAttachment,
)
from scenes.data_center_layout import HOME_ARM_Q, IK_SEED_Q, LAYOUT
from welds import activate_attachment_weld

NAME = "data_center"
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)
# Grippable objects addressable via Step.weld_activate / weld_deactivate.
# Index order matters: the runner uses it as an int index into this list.
GRIPPABLES: tuple[str, ...] = (
    # Cable connector bodies live inside the `cable{N}` attachment subtrees
    # (each cable is its own mjcf.RootElement attached at a site on the
    # rack), so dm_control namespaces them as `cable1/connector` etc.
    "cable1/connector",
    "cable2/connector",
    "cable3/connector",
    "server",
    "new_server",
)
N_CUBES = len(GRIPPABLES)  # runner uses this to size ArmHandles.weld_ids


class DataCenterAux(StrEnum):
    """Scene-owned (non-arm) actuators addressable via Step.aux_ctrl.

    `LIFT` maps directly to PAL TIAGo's upstream `torso_lift_joint` name so the
    scene's aux-ctrl dispatch reaches the joint Menagerie already declares in
    tiago.xml (we add only a position actuator on top).
    """

    LIFT = "torso_lift_joint"


AUX_ACTUATOR_NAMES: tuple[str, ...] = tuple(m.value for m in DataCenterAux)

CAMERAS: tuple[tuple[str, CameraRole], ...] = (
    # Realsense D435i attached at the top of the camera pole. dm_control
    # namespaces names of attached subtrees with `<model>/`, so the camera
    # the d435i.xml declares as `d435i_cam` compiles as `top/d435i_cam`.
    ("top/d435i_cam", CameraRole.TOP),
    # Wrist D405 cameras are added directly inside the piper subtree
    # (`{side}link6`) before attach so they pick up the piper's namespace
    # prefix. Result: `left/wrist_d405_cam`, `right/wrist_d405_cam`.
    ("left/wrist_d405_cam", CameraRole.WRIST),
    ("right/wrist_d405_cam", CameraRole.WRIST),
)

# Camera invariants — pinned at startup by `scene_check.check_scene`. Each
# entry locks a camera's structural identity (parent body, MuJoCo cam mode,
# optional targetbody) so a future edit that flips the top cam to FIXED,
# or accidentally re-parents a wrist cam to base_link, fails fast at
# startup with a clear message instead of as "the wrist view looks
# wrong" days later in viser.
CAMERA_INVARIANTS: tuple[CameraInvariant, ...] = (
    # Top camera rides the camera pole's mesh body (`top/d435i`), attached
    # to the static `base_link` (not the moving torso_lift_link) so the
    # view stays stable regardless of lift qpos. Targets `rack_frame`, so
    # the optical axis tracks the rack as the robot's base rotates.
    TargetingCameraInvariant(
        name="top/d435i_cam",
        parent_body="top/d435i",
        targetbody="rack_frame",
        mode="targetbody",
    ),
    # Wrist cams mount on each arm's link6 (`left/link6`, `right/link6`)
    # in FIXED mode. The 180°-x quat on the camera makes its optical
    # axis (-z) align with link6's +z (the gripper axis), so each wrist
    # view always looks at whatever the arm is reaching for.
    FixedCameraInvariant(name="left/wrist_d405_cam", parent_body="left/link6"),
    FixedCameraInvariant(name="right/wrist_d405_cam", parent_body="right/link6"),
)


# -----------------------------------------------------------------------------
# Scene dimensions
# -----------------------------------------------------------------------------
# Every dimension, offset, and derived anchor lives in `DataCenterLayout`
# (scenes/data_center_layout.py). `LAYOUT` is the module-level default
# instance; read positions via `LAYOUT.rack.front_face_x`,
# `LAYOUT.server_world_pos_in_rack`, `LAYOUT.port_world_pos(i)`,
# `LAYOUT.cable_anchor_world(i)`, etc. `HOME_ARM_Q` / `IK_SEED_Q` are
# re-exported from that module for call-site convenience.
#
# World frame: origin at the robot's base centre on the floor, +x toward
# rack, +y to the robot's right, +z up.


# -----------------------------------------------------------------------------
# Static-overlap allow-list for `scene_check.check_scene`
# -----------------------------------------------------------------------------
# Pairs of geom or body names whose AABBs intentionally overlap in the
# compiled scene. `check_scene` skips same-body geom pairs automatically;
# this allow-list captures the cross-body overlaps that are by-design
# (panels meeting at cabinet seams, a bracket flush against a side wall).
# Seeded after the first `--inspect` run surfaced the legitimate overlaps.
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
    # surrounding panels at every edge. Patch panel sits on top of it.
    ("rack_rear", "rack_shelf"),
    ("rack_side_L", "rack_shelf"),
    ("rack_side_R", "rack_shelf"),
    ("rack_shelf", "patch_panel"),
    # Shelf top is at server bottom z; cable anchors mount on top of the
    # patch panel which sits on top of the shelf — anchor bodies graze
    # the shelf top by design.
    ("rack_shelf", "cable1/anchor"),
    ("rack_shelf", "cable2/anchor"),
    ("rack_shelf", "cable3/anchor"),
    # Cables emerge from a 1U patch panel mounted on the rack front rails,
    # one rack-unit below the server slot. Each cable's anchor body
    # touches the patch-panel face where it emerges, and the rod body
    # starts co-located with the anchor (rest pose) so the two overlap
    # by design. Under dm_control namespacing the cable subtrees compile
    # as `cable{N}/anchor`, `cable{N}/rod`, `cable{N}/connector`.
    ("patch_panel", "cable1/anchor"),
    ("patch_panel", "cable2/anchor"),
    ("patch_panel", "cable3/anchor"),
    ("patch_panel", "cable1/rod"),
    ("patch_panel", "cable2/rod"),
    ("patch_panel", "cable3/rod"),
    ("cable1/anchor", "cable1/rod"),
    ("cable2/anchor", "cable2/rod"),
    ("cable3/anchor", "cable3/rod"),
    # Top camera sits directly on top of the camera pole — the pole-top
    # and mesh-bottom AABBs touch by design. The camera-mesh body
    # contains ~9 sub-geoms so a body-pair allow covers the lot.
    ("world", "top/d435i"),
    # TIAGo's base mesh uses a conservative sphere bound in scene_check
    # (rbound ≈ 0.30 m at mesh origin z≈0.16), so after pulling the rack
    # closer to center_x=0.58 the rbound sphere grazes the rack walls
    # even though the actual mesh geometry is 25+ cm away. Body-pair
    # allow since the base mesh fuses into the `world` body.
    ("world", "rack_side_L"),
    ("world", "rack_side_R"),
    ("world", "rack_bottom"),
    # Same conservative-sphere story for the floor-standing cart at
    # cart.center_y=-0.55: its casters and bottom shelf sit close enough
    # to the floor that TIAGo's base-mesh sphere bound overlaps them by
    # design.
    ("world", "cart_frame"),
)


# -----------------------------------------------------------------------------
# Body-pair equalities — single registry drives build_spec / id resolution /
# reset. Two constraint kinds are used:
#
#   "connect" — port↔cable-connector. `mjEQ_CONNECT` pins a point on body1
#   (in body1's local frame) to body2's origin. No orientation constraint, so
#   the cable's last body can rotate freely in its socket — which is what a
#   real plug does. This replaces the old weld-with-captured-relpose approach
#   that forced the cable tip to hold the arbitrary orientation the composite
#   happened to have at init (cable running from a side bracket, so tip +x
#   pointed along -y world) — the connector geom then sat crooked next to
#   the port, not seated in it. User: "the cables aren't plugged into the
#   server properly".
#
#   "weld" — server↔rack, server↔bin, new_server↔rack. `mjEQ_WELD` with
#   relpose captured at activation time (`welds.activate_attachment_weld`)
#   so the body stays where it is when the weld flips on rather than
#   snapping to body1's origin.
# -----------------------------------------------------------------------------


class AttachmentWeldName(StrEnum):
    """Named body-pair equalities used by Step.attach_activate/deactivate.

    The "weld_" prefix is historical — PORT1..3 entries are now `connect`
    equalities rather than welds, but renaming would break downstream Step
    references with no functional benefit.
    """

    PORT1_OLD = "weld_port1_old"
    PORT2_OLD = "weld_port2_old"
    PORT3_OLD = "weld_port3_old"
    PORT1_NEW = "weld_port1_new"
    PORT2_NEW = "weld_port2_new"
    PORT3_NEW = "weld_port3_new"
    SERVER_IN_RACK = "weld_server_in_rack"
    SERVER_ON_CART_BOTTOM = "weld_server_on_cart_bottom"
    NEW_ON_CART_TOP = "weld_new_on_cart_top"
    NEW_IN_RACK = "weld_new_in_rack"


# Single source of truth for body-pair equalities. Entries are
# `WeldAttachment` (full-pose pin) or `ConnectAttachment` (point-pin with an
# anchor in body_a's local frame); the variant itself encodes the constraint
# kind, so build_spec / apply_initial_state pattern-match on `isinstance(...)`
# rather than reading a separate `kind` field. `scene_check.check_scene`
# consumes the same tuple to verify each compiled equality matches its
# declared variant.
ATTACHMENTS: tuple[AttachmentConstraint, ...] = (
    # Port connects — body_a is the SERVER (anchor is in server's local
    # frame); body_b is the cable connector whose origin gets pinned.
    # Cable connectors live inside the `cable{N}` attachment subtrees so
    # dm_control namespaces them as `cable{N}/connector`. Anchor positions
    # come from `LAYOUT.port_anchor_in_server_frame` so the port geom
    # position and the connect anchor stay in lockstep.
    ConnectAttachment(
        name=AttachmentWeldName.PORT1_OLD,
        body_a="server",
        body_b="cable1/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(0),
        initially_active=True,
    ),
    ConnectAttachment(
        name=AttachmentWeldName.PORT2_OLD,
        body_a="server",
        body_b="cable2/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(1),
        initially_active=True,
    ),
    ConnectAttachment(
        name=AttachmentWeldName.PORT3_OLD,
        body_a="server",
        body_b="cable3/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(2),
        initially_active=True,
    ),
    ConnectAttachment(
        name=AttachmentWeldName.PORT1_NEW,
        body_a="new_server",
        body_b="cable1/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(0),
        initially_active=False,
    ),
    ConnectAttachment(
        name=AttachmentWeldName.PORT2_NEW,
        body_a="new_server",
        body_b="cable2/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(1),
        initially_active=False,
    ),
    ConnectAttachment(
        name=AttachmentWeldName.PORT3_NEW,
        body_a="new_server",
        body_b="cable3/connector",
        anchor_in_a=LAYOUT.port_anchor_in_server_frame(2),
        initially_active=False,
    ),
    # Server placement welds (full pose):
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


# -----------------------------------------------------------------------------
# Grippable addressing
# -----------------------------------------------------------------------------


def grippable_id(name: str) -> CubeID:
    """Resolve a grippable-object name to its bounds-checked CubeID.

    Centralising the name→index translation means the scene never builds
    `f"cable{i+1}_connector"` strings next to a bare `_index_of(...)` call;
    `grippable_id` is the only place that knows GRIPPABLES is a list.
    """
    try:
        index = GRIPPABLES.index(name)
    except ValueError as exc:
        raise KeyError(f"unknown grippable {name!r}; known: {GRIPPABLES}") from exc
    return make_cube_id(index, N_CUBES)


def grasp_weld(side: ArmSide, cube_id: CubeID) -> str:
    """Name of the per-arm grasp weld for a given cube. Uses `_` as the
    side separator (`left_grasp_cube0`) because dm_control.mjcf reserves
    `/` for namespace scoping. Matches the convention `arm_handles`
    uses to look up `weld_ids` after compilation."""
    return f"{side.replace('/', '_')}grasp_cube{cube_id}"


# -----------------------------------------------------------------------------
# Spec construction
# -----------------------------------------------------------------------------


def _add_cart(root: mjcf.RootElement, visual_class: str) -> mjcf.Element:
    """Service cart: 4 corner posts, 2 shelves, 4 casters, push handle.

    Returns the cart_frame body so weld equalities can reference it
    later. All sub-geoms use the `visual` default class (no contacts,
    no mass) — the cart is a visual stand; servers are held in place by
    welds, not by sitting on the shelves.
    """
    cart_cfg = LAYOUT.cart
    cart_rgba = [0.55, 0.58, 0.60, 1.0]
    metal_rgba = [0.40, 0.42, 0.44, 1.0]
    wheel_rgba = [0.10, 0.10, 0.12, 1.0]

    cart = root.worldbody.add(
        "body",
        name="cart_frame",
        pos=[cart_cfg.center_x, cart_cfg.center_y, 0.0],
    )

    # Corner posts.
    post_top_z = cart_cfg.top_shelf_z + 0.01
    post_half_z = post_top_z * 0.5
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cart.add(
                "geom",
                dclass=visual_class,
                type="box",
                pos=[
                    sx * (cart_cfg.half_x - cart_cfg.post_half),
                    sy * (cart_cfg.half_y - cart_cfg.post_half),
                    post_half_z,
                ],
                size=[cart_cfg.post_half, cart_cfg.post_half, post_half_z],
                rgba=metal_rgba,
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

    # Casters at each corner.
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cart.add(
                "geom",
                dclass=visual_class,
                type="sphere",
                pos=[
                    sx * (cart_cfg.half_x - cart_cfg.post_half),
                    sy * (cart_cfg.half_y - cart_cfg.post_half),
                    cart_cfg.caster_radius,
                ],
                size=[cart_cfg.caster_radius],
                rgba=wheel_rgba,
            )

    # Push handle on the rear edge (the side facing away from the rack —
    # cart_y is negative, so the rack is at +y from cart's POV; the handle
    # sits at -y so an operator pushing the cart faces the rack).
    handle_y = -cart_cfg.half_y - cart_cfg.post_half
    handle_z_top = cart_cfg.handle_height
    handle_z_mid = (cart_cfg.top_shelf_z + handle_z_top) * 0.5
    handle_post_half_z = (handle_z_top - cart_cfg.top_shelf_z) * 0.5
    for sx in (-1.0, 1.0):
        cart.add(
            "geom",
            dclass=visual_class,
            type="box",
            pos=[
                sx * (cart_cfg.half_x - cart_cfg.post_half),
                handle_y,
                handle_z_mid,
            ],
            size=[cart_cfg.post_half, cart_cfg.post_half, handle_post_half_z],
            rgba=metal_rgba,
        )
    cart.add(
        "geom",
        dclass=visual_class,
        type="box",
        pos=[0.0, handle_y, handle_z_top],
        size=[cart_cfg.half_x, cart_cfg.post_half, cart_cfg.post_half],
        rgba=metal_rgba,
    )

    return cart


def _quat_align_x_to(direction: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternion that rotates +x̂ onto the given direction vector.

    Used to orient the cable composite (which lays bodies along local +x by
    default) toward an arbitrary target in the parent frame.
    """
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    cross = np.array([0.0, -v[2], v[1]])  # x̂ × v
    sin_a = float(np.linalg.norm(cross))
    if sin_a < 1e-9:
        # v ≈ ±x̂: identity for +x, 180° about ŷ for -x.
        return np.array([1.0, 0.0, 0.0, 0.0]) if v[0] > 0 else np.array([0.0, 0.0, 1.0, 0.0])
    axis = cross / sin_a
    half = float(np.arccos(np.clip(v[0], -1.0, 1.0)) / 2.0)
    sin_h = float(np.sin(half))
    return np.array([float(np.cos(half)), axis[0] * sin_h, axis[1] * sin_h, axis[2] * sin_h])


def _build_cable_root(
    cable_idx: int, port_y: float, rgba
) -> tuple[mjcf.RootElement, np.ndarray, np.ndarray]:
    """Build a `cable{N}` `mjcf.RootElement` — anchor + single rigid rod
    on a ball joint + connector tip. Returns (root, local_anchor_in_rack,
    attach_quat) so the caller can attach via a site on the rack with
    the right pose.

    Topology:
        anchor (rigid box, attached at rack site)
          └─ rod (ball joint, capsule along +x, length = anchor↔port)
                └─ connector (rigid sphere, no joint, at rod's +x tip)

    Length matches the direct anchor↔port distance so the rest pose lays
    the rod straight from the patch-panel anchor to the server port —
    visually clean, no curling, no chain segments poking past the server.
    A multi-segment chain we tried earlier let the connector follow long
    arm pulls but produced a tangled rest shape because of the slack.
    The replug task plan now uses a small (~5 cm) pull distance compatible
    with the rigid rod's reach.
    """
    cables_cfg = LAYOUT.cables
    rack = LAYOUT.rack
    world_anchor = LAYOUT.cable_anchor_world(cable_idx)
    rack_origin = np.array([rack.center_x, 0.0, rack.center_z])
    local_anchor = world_anchor - rack_origin

    world_port = LAYOUT.port_world_pos(list(LAYOUT.ports.y_offsets).index(port_y))
    direction = world_port - world_anchor
    direction_len = float(np.linalg.norm(direction))
    cable_len = min(direction_len, LAYOUT.cable_max_len)
    attach_quat = _quat_align_x_to(direction)
    seg_radius = cables_cfg.seg_radius

    cable = mjcf.RootElement(model=f"cable{cable_idx + 1}")
    anchor = cable.worldbody.add("body", name="anchor")
    anchor.add(
        "geom",
        type="box",
        size=[0.015, 0.010, 0.010],
        rgba=[0.3, 0.3, 0.3, 1.0],
        contype=0,
        conaffinity=0,
    )
    rod = anchor.add("body", name="rod", pos=[0.0, 0.0, 0.0])
    rod.add("joint", name="rod_ball", type="ball", damping=20.0)
    rod.add(
        "geom",
        type="capsule",
        fromto=[0.0, 0.0, 0.0, cable_len, 0.0, 0.0],
        size=[seg_radius],
        rgba=list(rgba),
        mass=0.05,
        contype=0,
        conaffinity=0,
    )
    connector = rod.add("body", name="connector", pos=[cable_len, 0.0, 0.0])
    # Connector head ~24 mm sphere — reads as the plug ferrule on a
    # 14 mm cable, big enough to spot from the 3/4 hero camera.
    connector.add(
        "geom",
        type="sphere",
        pos=[0.0, 0.0, 0.0],
        size=[0.012],
        rgba=list(rgba),
        mass=0.02,
    )
    return cable, local_anchor, attach_quat


def _add_server_body(
    parent: mjcf.Element,
    *,
    name: str,
    pos,
    rgba_chassis,
    server_cfg,
    port_local_x: float,
    port_y_offsets,
    port_colors,
) -> None:
    """Build a server `<body>` (chassis + 3 ports) at `parent`. Both
    `server` and `new_server` share this layout."""
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
    # No "handle rail" geoms on the server front face — earlier we placed
    # two small white rectangles at ±15 cm to mark grasp targets, but they
    # read as decoration nobody could explain. The arms grip the chassis
    # sides directly via weld activation at the same world coords; the
    # boxes were visual noise.
    shapes = ("box", "box", "cylinder")
    for i, (y, rgba, shape) in enumerate(zip(port_y_offsets, port_colors, shapes, strict=True)):
        if shape == "box":
            size = [0.010, 0.018, 0.018] if i == 0 else [0.010, 0.012, 0.007]
            server.add(
                "geom",
                type="box",
                pos=[port_local_x, y, 0.0],
                size=size,
                rgba=list(rgba),
            )
        else:  # cylinder — fiber (fromto only; MuJoCo forbids pos+fromto)
            server.add(
                "geom",
                type="cylinder",
                fromto=[port_local_x - 0.010, y, 0.0, port_local_x + 0.005, y, 0.0],
                size=[0.010],
                rgba=list(rgba),
            )


def build_spec() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the data-center scene as a `dm_control.mjcf.RootElement`,
    compile it via `mujoco.MjModel.from_xml_string`, and return the
    `(model, data)` pair.

    Returns plain MuJoCo handles (not a `mjcf.Physics`) so downstream
    runtime code (runner.advance_arm, tools/_runtime.advance_timeline)
    operates against the standard mujoco.MjModel/MjData API. dm_control
    is used only as the assembly layer.
    """
    # Start from PAL TIAGo (wheels + torso_lift_link). `robots.tiago.load_tiago`
    # strips the upstream single arm + head, removes the `reference` freejoint
    # (required for mink IK — see that module's docstring), prunes orphan
    # contact excludes, and clears the model name so TIAGo's body names
    # compile unprefixed.
    root = load_tiago()

    # Physics options. mjcf maps these to the `<option>` element's attributes.
    root.option.integrator = "implicitfast"
    root.option.cone = "elliptic"
    root.option.impratio = 10.0
    root.option.timestep = 0.002
    # Disable global contacts via the `<flag>` child of `<option>`. Every body
    # in this scene is held by welds or direct qpos writes; contact detection
    # just adds work and previously triggered a QACC warning at t=41.53 s on
    # left/joint5.
    root.option.flag.contact = "disable"
    # Puppet mode: gravity = 0. The arms are qpos-driven (no PD tracking,
    # no need for gravity to load the actuators), and the freejoint
    # bodies (servers, cable connectors) are always pinned by some weld
    # — gravity would just be solver noise.
    root.option.gravity = [0.0, 0.0, 0.0]

    # Off-screen framebuffer big enough for HD hero renders (proposal
    # decks, marketing). Default MuJoCo offwidth/offheight is 640×480
    # which produces a noticeably grainy still. mjcf names this child
    # `global` (a Python keyword) — access via getattr.
    visual_global = getattr(root.visual, "global")
    visual_global.offwidth = 1920
    visual_global.offheight = 1080

    # Default class for visual-only geoms: no contact, zero mass. Geoms that
    # set `dclass="visual"` inherit these flags, so we don't repeat the
    # contype=0 / conaffinity=0 / mass=0.0 trio at every call site (camera
    # pole, rack panels, cart shelves, D405 wrist meshes).
    visual = root.default.add("default", dclass="visual")
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.mass = 0.0

    # D405 mesh asset, shared by both wrist cameras. Added at scene root so
    # both attached pipers can reference it. mjcf resolves cross-namescope
    # references by Element identity, not string lookup — passing the
    # `d405_mesh` Element to a geom inside a sub-namescope (e.g. `left/`)
    # produces the correct compiled `mesh="d405"` reference. Passing the
    # bare string would prefix it as `left/d405` and fail to find the
    # asset at compile time.
    d405_mesh = root.asset.add("mesh", name="d405", file=str(D405_MESH_STL))

    wb = root.worldbody

    # Lighting — TIAGo's upstream MJCF ships no lights, and the default
    # MuJoCo scene comes with a single dim headlight that leaves stills
    # looking grainy/black. Three-point setup: a key spot above the
    # robot, a fill light from the rack side, and a back/rim light from
    # behind to separate the robot from the dark background.
    wb.add(
        "light",
        name="key",
        pos=[0.4, -0.5, 2.5],
        dir=[0.0, 0.5, -1.0],
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
        castshadow="false",
    )
    wb.add(
        "light",
        name="fill",
        pos=[0.4, 1.5, 1.8],
        dir=[0.0, -1.0, -0.4],
        diffuse=[0.5, 0.5, 0.55],
        specular=[0.1, 0.1, 0.1],
        castshadow="false",
    )
    wb.add(
        "light",
        name="rim",
        pos=[-1.0, 0.0, 2.0],
        dir=[1.0, 0.0, -0.5],
        diffuse=[0.4, 0.42, 0.5],
        specular=[0.1, 0.1, 0.1],
        castshadow="false",
    )

    # Floor (TIAGo's upstream MJCF has no floor — the per-variant scene.xml adds it)
    wb.add(
        "geom",
        type="plane",
        size=[5.0, 5.0, 0.1],
        rgba=[0.78, 0.78, 0.80, 1.0],
    )

    # Reference to the moving lift body — Piper arms attach via sites on it.
    torso = root.find("body", "torso_lift_link")
    base = root.find("body", "base_link")

    # ---------------------- Service cart (new + old server stowage) -----------
    _add_cart(root, "visual")

    # ---------------------- Shoulder bar bridging torso to arm bases ----------
    # The Piper bases mount at ±arm_mount.y_abs (= 30 cm) outboard of the
    # torso centreline, but TIAGo's torso shell is much narrower than that
    # — without a connecting geom the arms read as "floating in mid-air
    # next to the robot". Add a shoulder bar on the torso at arm_mount.z
    # (the arms' attach height) spanning the full ±arm_mount.y_abs so
    # there's a visible structure reaching out from the body to each arm
    # mount point.
    arm_mount_cfg = LAYOUT.arm_mount
    SHOULDER_HALF_Z = 0.07  # 14 cm tall bar
    SHOULDER_OVERHANG = 0.08  # bar extends past each arm base by this much
    torso.add(
        "geom",
        dclass="visual",
        name="shoulder_bar",
        type="box",
        # Positioned so the bar's TOP face sits at arm_mount.z (where the
        # arm bases attach) — arms read as "mounted on top of the bar"
        # rather than embedded in it. Centre is one half-height below.
        pos=[arm_mount_cfg.x, 0.0, arm_mount_cfg.z - SHOULDER_HALF_Z],
        # Half-extents in y bumped past arm_mount.y_abs by SHOULDER_OVERHANG
        # so the bar reads as a substantial cross-piece extending outboard
        # of the arm mount points, not just spanning between them.
        size=[
            0.07,
            arm_mount_cfg.y_abs + SHOULDER_OVERHANG,
            SHOULDER_HALF_Z,
        ],  # 14 cm × 76 cm × 14 cm
        # Anodised-aluminium grey, distinct from TIAGo's white shell so
        # the shoulder bar reads as its own structural element bridging
        # torso ↔ arm bases.
        rgba=[0.35, 0.36, 0.38, 1.0],
    )

    # ---------------------- Piper arms on the torso --------------------------
    # Pre-build each piper, add its TCP site + wrist camera + wrist mesh
    # INSIDE the piper subtree so they pick up the namespace prefix
    # (`left/tcp`, `left/wrist_d405_cam`), THEN attach via a site on the
    # torso. `arm_mount.x = 0` keeps the shoulder pivot at the torso
    # centreline; arm_mount.z = -0.15 is the natural shoulder height;
    # ±arm_mount.y_abs places each arm on its respective side.
    arm_mount = LAYOUT.arm_mount
    WRIST_MESH_OFFSET = [0.0, 0.0, 0.03]
    for side, y_sign in ((ArmSide.LEFT, -1.0), (ArmSide.RIGHT, 1.0)):
        piper = load_piper(side)
        link6 = piper.find("body", "link6")
        if link6 is None:
            raise RuntimeError(f"piper {side!r} missing 'link6' after load")
        # TCP site — IK / weld code addresses this by `f"{side}tcp"` (e.g.
        # `"left/tcp"`); inside the piper namescope the local name is `tcp`.
        link6.add(
            "site",
            name="tcp",
            pos=[0.0, 0.0, 0.14],
            size=[0.006, 0.006, 0.006],
        )
        # Wrist camera mesh — orientation chosen so the mount face sits
        # against link6 and the lens points sideways (matches how a real
        # D405 mounts on an Aloha-style Piper wrist).
        link6.add(
            "geom",
            dclass="visual",
            type="mesh",
            mesh=d405_mesh,
            pos=WRIST_MESH_OFFSET,
            rgba=[0.12, 0.12, 0.14, 1.0],
        )
        # FIXED-mode camera with a 180°-x quat so the optical axis (cam -z)
        # aligns with link6 +z (the gripper / TCP axis) — the wrist view
        # always looks at whatever the arm is reaching for, symmetrically
        # for both sides.
        link6.add(
            "camera",
            name="wrist_d405_cam",
            pos=[0.0, 0.0, 0.08],
            quat=[0.0, 1.0, 0.0, 0.0],
            mode="fixed",
            fovy=87.0,
        )
        mount_site = torso.add(
            "site",
            name=f"{side.rstrip('/')}_arm_mount",
            pos=[arm_mount.x, y_sign * arm_mount.y_abs, arm_mount.z],
            quat=[1.0, 0.0, 0.0, 0.0],
            size=[0.001, 0.001, 0.001],
        )
        mount_site.attach(piper)

    # ---------------------- Top camera pole (static, above the arms) --------
    # TIAGo's head is stripped in `load_tiago` (we don't actuate it). We
    # replace it with a tall static pole rigidly welded to `base_link` (not
    # the moving `torso_lift_link`) so the top camera view stays stable
    # regardless of the torso lift qpos.
    TOP_POLE_X = -0.20
    base.add(
        "geom",
        dclass="visual",
        name="top_camera_pole",
        type="box",
        pos=[TOP_POLE_X, 0.0, 0.74],
        size=[0.025, 0.025, 0.74],
        rgba=[0.32, 0.32, 0.35, 1.0],
    )
    # D435i's native mesh frame has local +z as the long camera-body axis
    # and local +x as the lens direction. The 90°-x quat below rotates
    # local +z → world -y so the body lies horizontal; lens still points
    # +x (toward the rack). Mesh is ~2.6 cm tall after rotation; attach
    # z = 1.485 puts the bottom flush with the pole top (allowlisted in
    # ALLOWED_STATIC_OVERLAPS).
    top_d435i = mjcf.from_path(str(D435I_XML))
    top_d435i.model = "top"
    top_d435i_body = top_d435i.find("body", "d435i")
    top_cam_mount = base.add(
        "site",
        name="top_cam_mount",
        pos=[TOP_POLE_X, 0.0, 1.485],
        quat=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
        size=[0.001, 0.001, 0.001],
    )
    top_cam_mount.attach(top_d435i)
    # The TARGETBODY camera is added AFTER `rack_frame` is built (below);
    # mjcf resolves cross-namescope refs by Element identity, and the
    # rack_frame Element doesn't exist yet at this point.

    # ---------------------- Rack (static, open front) ------------------------
    rack_cfg = LAYOUT.rack
    rack = wb.add("body", name="rack_frame", pos=[rack_cfg.center_x, 0.0, rack_cfg.center_z])
    rack_wall_t = rack_cfg.wall_thickness
    panel_rgba = (0.10, 0.10, 0.13, 1.0)
    rhx, rhy, rhz = rack_cfg.half

    def _rack_panel(name: str, pos, half) -> None:
        b = rack.add("body", name=name, pos=list(pos))
        # Visual-only (class "visual"): the server is weld-attached so it
        # doesn't need a shelf to rest on, and arms reach through the open
        # front. Letting panels collide just creates solver noise around the
        # server body.
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
    # appearing to float. Thin (12 mm) but wide; the patch panel sits on
    # the front edge of this shelf as a 1U cable-management fixture.
    shelf_top_z_world = LAYOUT.server.slot_z - LAYOUT.server.half[2]  # = 0.835
    shelf_half_z = 0.006
    shelf_local_z = (shelf_top_z_world - shelf_half_z) - rack_cfg.center_z
    _rack_panel(
        "rack_shelf",
        (0.0, 0.0, shelf_local_z),
        (rhx - rack_wall_t, rhy - rack_wall_t, shelf_half_z),
    )
    # Patch panel — a 1U cable-management fixture mounted at the front of
    # the shelf. Each cable anchors on its front face directly under the
    # matching port on the server above. Standard data-centre layout
    # (switch one U below the server it patches into).
    cables_cfg = LAYOUT.cables
    patch_panel_world = LAYOUT.patch_panel_world_pos
    patch_panel_local_x = patch_panel_world[0] - rack_cfg.center_x
    patch_panel_local_z = patch_panel_world[2] - rack_cfg.center_z
    patch_panel_body = rack.add(
        "body",
        name="patch_panel",
        pos=[patch_panel_local_x, 0.0, patch_panel_local_z],
    )
    patch_panel_body.add(
        "geom",
        dclass="visual",
        type="box",
        size=[
            cables_cfg.patch_panel_half_x,
            cables_cfg.patch_panel_half_y,
            cables_cfg.patch_panel_half_z,
        ],
        rgba=[0.18, 0.20, 0.22, 1.0],
    )
    # Indicator LEDs on the patch panel, one per cable port, matching the
    # cable's colour — reads as "this socket goes to that cable".
    for y, rgba in zip(LAYOUT.ports.y_offsets, LAYOUT.ports.colors, strict=True):
        patch_panel_body.add(
            "geom",
            dclass="visual",
            type="box",
            pos=[cables_cfg.patch_panel_half_x, y, 0.012],
            size=[0.001, 0.018, 0.005],
            rgba=list(rgba),
        )

    # Now that `rack_frame` is built, add the top-camera with a TARGETBODY
    # reference to it. Passing the Element (not the string `"rack_frame"`)
    # lets mjcf resolve the reference across namescopes — the camera lives
    # inside `top/`, the target lives at scene root.
    if top_d435i_body is not None:
        top_d435i_body.add(
            "camera",
            name="d435i_cam",
            pos=[0.0, 0.0, 0.0],
            mode="targetbody",
            target=rack,
            fovy=69.0,  # D435i colour-sensor horizontal FOV ≈ 69°
        )

    # ---------------------- Servers (rack + cart-top) ----------------------
    port_local_x = LAYOUT.port_local_x_on_server
    port_y_offsets = LAYOUT.ports.y_offsets
    port_colors = LAYOUT.ports.colors
    _add_server_body(
        wb,
        name="server",
        pos=LAYOUT.server_world_pos_in_rack,
        rgba_chassis=[0.16, 0.17, 0.19, 1.0],
        server_cfg=server_cfg,
        port_local_x=port_local_x,
        port_y_offsets=port_y_offsets,
        port_colors=port_colors,
    )
    _add_server_body(
        wb,
        name="new_server",
        pos=LAYOUT.new_server_initial_world_pos,
        rgba_chassis=[0.24, 0.26, 0.30, 1.0],
        server_cfg=server_cfg,
        port_local_x=port_local_x,
        port_y_offsets=port_y_offsets,
        port_colors=port_colors,
    )

    # ---------------------- Cables (3 ball-jointed sticks on rack) ---------
    for i, (y, rgba) in enumerate(zip(port_y_offsets, port_colors, strict=True)):
        cable_root, local_anchor, attach_quat = _build_cable_root(i, y, rgba)
        cable_mount = rack.add(
            "site",
            name=f"cable{i + 1}_mount",
            pos=list(local_anchor),
            quat=list(attach_quat),
            size=[0.001, 0.001, 0.001],
        )
        cable_mount.attach(cable_root)

    # ---------------------- Actuator: torso lift -------------------------
    # Position actuator on TIAGo's existing torso_lift_joint. TIAGo's
    # upstream MJCF ships only the joint (no actuator); we add the position
    # servo here. kp=80000 / kv=800 keeps the torso pinned to its commanded
    # qpos under puppet mode (qpos is direct-written each tick anyway, so
    # the actuator's job is mostly cosmetic — but the gain is calibrated
    # for a possible PD-mode revert).
    root.actuator.add(
        "position",
        name=DataCenterAux.LIFT,
        joint=DataCenterAux.LIFT,
        kp=80000.0,
        kv=800.0,
        ctrllimited="true",
        ctrlrange=[0.0, LAYOUT.tiago.lift_range],
    )

    # ---------------------- Welds (grasp + attachment) ---------------------
    # Grasp welds: (arm, grippable_object). All inactive at start. Names
    # must match `grasp_weld(side, i)` — `get_arm_handles` looks them up
    # via the same `<side>_grasp_cube<i>` formula. The body lookups
    # use slash-namespaced piper bodies (`left/link6`).
    for side in ARM_PREFIXES:
        hand = f"{side}link6"
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

    # Attachment equalities — single iteration over the ATTACHMENTS registry.
    # Variant dispatch via isinstance: `ConnectAttachment` carries its anchor
    # in body_a's local frame; `WeldAttachment` is identity-pose at compile
    # time and the relpose gets re-seeded at runtime by
    # `welds.activate_attachment_weld`.
    for attachment in ATTACHMENTS:
        active_str = "true" if attachment.initially_active else "false"
        if isinstance(attachment, ConnectAttachment):
            ax, ay, az = attachment.anchor_in_a
            root.equality.add(
                "connect",
                name=attachment.name,
                body1=attachment.body_a,
                body2=attachment.body_b,
                active=active_str,
                anchor=[ax, ay, az],
                # Soft solref (200 ms time-constant) so port↔cable connect
                # equalities ease into place when activated rather than
                # shock-loading the closed loop (cable chain → connector →
                # CONNECT → server → SERVER_IN_RACK weld → rack → patch
                # panel → cable anchor). Stiff defaults pushed the welded
                # server out of position when PORT_NEW activated mid-replug
                # (user report: "new server starts flying around when put
                # down"). 200 ms is still plenty fast to pull the
                # connector to its port within ~one task-plan tick.
                solref=[0.2, 1.0],
            )
        else:  # WeldAttachment
            root.equality.add(
                "weld",
                name=attachment.name,
                body1=attachment.body_a,
                body2=attachment.body_b,
                active=active_str,
                relpose=[0, 0, 0, 1, 0, 0, 0],
            )

    # Exclude fragile contacts that would otherwise thrash. dm_control
    # namespacing renamed the piper roots to `left/base_link`,
    # `right/base_link`.
    root.contact.add("exclude", body1="torso_lift_link", body2="left/base_link")
    root.contact.add("exclude", body1="torso_lift_link", body2="right/base_link")
    root.contact.add("exclude", body1="torso_lift_link", body2="new_server")
    root.contact.add("exclude", body1="rack_frame", body2="server")

    # Compile via XML round-trip so the returned (model, data) are owned
    # by mujoco directly — no dm_control Physics lifetime concerns.
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
    lift_actuator_id: int
    server_body_id: int
    new_server_body_id: int
    rack_body_id: int
    # The moving TIAGo body the bins hang off (renamed from `carriage_body_id`
    # now that the mobile embodiment is TIAGo rather than our hand-built
    # `lift_carriage`).
    torso_body_id: int
    cable_connector_body_ids: list[int]
    # Attachment welds by enum member — derived from ATTACHMENTS, so it always
    # matches the spec in build_spec.
    attachment_eq: dict[AttachmentWeldName, int] = field(default_factory=dict)


def _resolve_scene_ids(model: mujoco.MjModel) -> _SceneIds:
    def body(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def eq(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    return _SceneIds(
        lift_actuator_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, DataCenterAux.LIFT),
        server_body_id=body("server"),
        new_server_body_id=body("new_server"),
        rack_body_id=body("rack_frame"),
        torso_body_id=body("torso_lift_link"),
        cable_connector_body_ids=[body(f"cable{i + 1}_connector") for i in range(3)],
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
) -> None:
    """Reset to home arm pose, lift at LAYOUT.lift.home, welds at spec defaults."""
    mujoco.mj_resetData(model, data)
    ids = _resolve_scene_ids(model)
    for arm in arms.values():
        for i, idx in enumerate(arm.arm_qpos_idx):
            data.qpos[idx] = HOME_ARM_Q[i]
        data.qpos[arm.qpos_idx[6]] = arm.gripper_open
        data.qpos[arm.qpos_idx[7]] = -arm.gripper_open
        data.ctrl[arm.act_arm_ids] = HOME_ARM_Q
        data.ctrl[arm.act_gripper_id] = arm.gripper_open
        # All grasp welds start inactive.
        for eq_id in arm.weld_ids:
            data.eq_active[eq_id] = 0
    # Lift at home
    data.ctrl[ids.lift_actuator_id] = LAYOUT.lift.home
    lift_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DataCenterAux.LIFT)
    data.qpos[model.jnt_qposadr[lift_jnt]] = LAYOUT.lift.home

    # Propagate qpos to xpos/xmat so the attachment welds below see the
    # actual initial body poses (not stale values from the previous tick).
    mujoco.mj_forward(model, data)

    # Attachment equalities: WELDs need their relpose seeded from current
    # body poses so they pin at compile-time positions (without this, each
    # weld's default identity relpose would drag body_b to body_a's origin
    # — server to rack centre, new_server to torso centre, etc.). CONNECTs
    # already carry their anchor in eq_data from build_spec — the anchor
    # is in body_a's local frame and doesn't depend on runtime pose, so
    # they just flip on/off via eq_active.
    for attachment in ATTACHMENTS:
        eq_id = ids.attachment_eq[attachment.name]
        if not attachment.initially_active:
            data.eq_active[eq_id] = 0
            continue
        if isinstance(attachment, ConnectAttachment):
            data.eq_active[eq_id] = 1
        else:  # WeldAttachment
            activate_attachment_weld(
                model,
                data,
                eq_id,
                int(model.eq_obj1id[eq_id]),
                int(model.eq_obj2id[eq_id]),
            )
    mujoco.mj_forward(model, data)


# -----------------------------------------------------------------------------
# Task plan
# -----------------------------------------------------------------------------


# Pre-flight IK tolerance: any `snap` call whose residual exceeds this
# aborts `make_task_plan` before viser starts. 2 cm is tight enough to
# catch near-misses (and force every target within fingertip distance) but
# loose enough not to fight currently-passing waypoints. Override by
# editing this constant if a scene genuinely needs looser limits.
_IK_POSITION_TOL_M = 0.02


def _snap_factory(model, data, arm):
    """Position-only IK closure, locking the lift so the solver can't use it.

    The returned `snap(target)` raises `RuntimeError` when the residual
    exceeds `_IK_POSITION_TOL_M` — so an unreachable waypoint aborts plan
    construction instead of quietly shipping a 50 cm approach error into
    the runtime. Error message includes the arm side, an in-closure call
    counter (so "snap #7 on left_" pinpoints the call site), and the XYZ
    target, which together name the failing IK attempt uniquely.
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
            locked_joint_names=(DataCenterAux.LIFT,),
            # DAQP over ProxQP: ProxQP raises NoSolutionFound on some of the
            # long-reach TIAGo-mounted targets — DAQP finds a feasible solve
            # (sub-mm error) in the same cases.
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


def _port_world_pos(port_idx: int) -> Position3:
    """World position of port geom `port_idx` on the rack-mounted server.
    Thin wrapper around `LAYOUT.port_world_pos` kept for legacy call sites."""
    return LAYOUT.port_world_pos(port_idx)


def _torso_world_z(lift_qpos: float) -> float:
    """World z of torso_lift_link at a given lift qpos. Mirrors TIAGo's
    upstream `<body name="torso_lift_link" pos="-0.062 0 0.8885">` plus the
    slide displacement."""
    return LAYOUT.tiago.torso_world_z(lift_qpos)


def _other_side(side: ArmSide) -> ArmSide:
    return ArmSide.RIGHT if side == ArmSide.LEFT else ArmSide.LEFT


def make_task_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
) -> dict[ArmSide, list[Step]]:
    """Scripted data-center server swap.

    Weld choreography:
      * Cable unplug — "grip connector" step attach_activates the per-arm
        grasp weld AND attach_deactivates the port-old weld simultaneously,
        so the cable connector is never double-pinned.
      * Server extract — "grip server" step activates each arm's grasp weld
        in place (no teleport). Left arm also deactivates the server-in-rack
        weld.
      * Server stow — "seat in bin" step activates server-in-old-bin weld;
        the following "release" step deactivates both grasp welds.
      * New-server grab/install — mirror.
      * Cable replug — "seat in port" step activates port-new weld; the
        following step releases the grasp.
    """
    scripts: dict[ArmSide, list[Step]] = {side: [] for side in ARM_PREFIXES}
    lift_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DataCenterAux.LIFT)

    def push_both(label: str, duration: float, aux: dict[str, float] | None = None) -> None:
        for side in ARM_PREFIXES:
            scripts[side].append(
                Step(
                    label=label,
                    arm_q=HOME_ARM_Q.copy(),
                    gripper="open",
                    duration=duration,
                    aux_ctrl=aux,
                )
            )

    def idle(
        other_side: ArmSide,
        label_prefix: str,
        durations: list[float],
        aux: dict[str, float] | None = None,
    ) -> None:
        """Append matching idle-at-home steps on the other arm's timeline."""
        for i, d in enumerate(durations):
            scripts[other_side].append(
                Step(
                    label=f"{label_prefix}.{i + 1}",
                    arm_q=HOME_ARM_Q.copy(),
                    gripper="open",
                    duration=d,
                    aux_ctrl=aux,
                )
            )

    def seed_at_lift(lift_qpos: float) -> tuple:
        """Set carriage to lift_qpos and return fresh snap closures."""
        apply_initial_state(model, data, arms, cube_body_ids)
        data.qpos[model.jnt_qposadr[lift_jnt]] = lift_qpos
        mujoco.mj_forward(model, data)
        return (
            _snap_factory(model, data, arms[ArmSide.LEFT]),
            _snap_factory(model, data, arms[ArmSide.RIGHT]),
        )

    # 0) Open: both at home, lift down. Then raise lift to cable height and
    # hold the target for a beat so the TIAGo torso PD has time to track the
    # step before the IK-planned approach starts. User report: "when it
    # starts up the telescopic lift pushes the whole platform up which makes
    # the robot arms not aligned with the cable" — the IK was planned with
    # lift qpos = LIFT_CABLES, but the commanded ctrl was still settling
    # when the first approach step began, so the arm landed above the actual
    # cable position. Bumping the ramp duration to 3 s plus a 1 s hold gives
    # the physical lift time to catch up with ctrl before arms start moving.
    push_both("home", 1.0, aux={DataCenterAux.LIFT: LAYOUT.lift.home})
    push_both("lift to cables", 3.0, aux={DataCenterAux.LIFT: LAYOUT.lift.cables})
    push_both("settle at cables", 1.0, aux={DataCenterAux.LIFT: LAYOUT.lift.cables})

    # Port → its "*_old" and "*_new" attachment welds. Indexed by port_idx.
    _PORT_OLD = (
        AttachmentWeldName.PORT1_OLD,
        AttachmentWeldName.PORT2_OLD,
        AttachmentWeldName.PORT3_OLD,
    )
    _PORT_NEW = (
        AttachmentWeldName.PORT1_NEW,
        AttachmentWeldName.PORT2_NEW,
        AttachmentWeldName.PORT3_NEW,
    )

    # 1) Unplug cables sequentially (L1, R2, L3). Other arm idles at home.
    def plan_unplug(port_idx: int, active_side: ArmSide) -> None:
        cable_id = grippable_id(f"cable{port_idx + 1}/connector")
        cable_grasp_weld = grasp_weld(active_side, cable_id)
        port_weld = _PORT_OLD[port_idx]

        snap_l, snap_r = seed_at_lift(LAYOUT.lift.cables)
        snap = snap_l if active_side == ArmSide.LEFT else snap_r

        # Read the connector body's world pose AFTER seed_at_lift so the IK
        # target actually lands on the connector — not SERVER_FRONT_X minus a
        # few cm, which was the MVP approximation and left TCP 2–3 cm short
        # of the connector geom (user report: "arms never actually touch the
        # cable heads"). The connector is pinned to the server by the port
        # weld at this point, so its world pose is deterministic.
        connector_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"cable{port_idx + 1}/connector"
        )
        connector_pos = np.asarray(data.xpos[connector_bid], dtype=float).copy()

        approach = connector_pos + np.array([-0.10, 0.0, 0.0])
        at_conn = connector_pos
        # Pull short — the rigid cable rod has length ≈ direct anchor↔port
        # distance (~5 cm), so the connector can swing on the rod's ball
        # joint but can't be yanked further than rod_len from the anchor
        # without ripping the closed constraint loop. 4 cm forward is
        # well within reach and reads as a clear visible pull.
        pulled = connector_pos + np.array([-0.04, 0.0, 0.02])
        # No "park" position any more — release the cable right at the
        # pulled-free spot (14 cm in front of port). Earlier we parked
        # the cable to the side of the rack, but the cable composite
        # length didn't reach those side positions reliably (cable over-
        # stretches → elastic plugin builds force → QACC blows up at
        # replug time). Releasing close to the port also makes the
        # subsequent replug a short arm motion instead of a long swing.

        q_approach, _ = snap(approach)
        q_at, _ = snap(at_conn)
        q_pulled, _ = snap(pulled)

        aux = {DataCenterAux.LIFT: LAYOUT.lift.cables}
        # Step durations include dedicated settle time before grip: the
        # piper PD with its stock ±100 N·m forcerange takes ~1.5 s to
        # converge from a 1-rad-magnitude step change against the arm's
        # own gravity, and the grasp weld freezes whatever relative pose
        # link6 has at grip time — so an under-settled approach step
        # would weld the connector 3–5 cm off-centre and then drag it
        # around at that offset for the rest of the sequence.
        active_steps = [
            Step(f"approach c{port_idx + 1}", q_approach, "open", 2.0, aux_ctrl=aux),
            Step(f"at connector c{port_idx + 1}", q_at, "open", 1.6, aux_ctrl=aux),
            Step(
                f"grip + unplug c{port_idx + 1}",
                q_at,
                "closed",
                0.5,
                aux_ctrl=aux,
                attach_activate=(cable_grasp_weld,),
                attach_deactivate=(port_weld,),
            ),
            Step(f"pull c{port_idx + 1} free", q_pulled, "closed", 1.0, aux_ctrl=aux),
            Step(
                f"release c{port_idx + 1}",
                q_pulled,
                "open",
                0.4,
                aux_ctrl=aux,
                attach_deactivate=(cable_grasp_weld,),
            ),
        ]
        scripts[active_side].extend(active_steps)
        idle(
            _other_side(active_side),
            f"wait c{port_idx + 1}",
            [s.duration for s in active_steps],
            aux=aux,
        )

    plan_unplug(0, ArmSide.LEFT)
    plan_unplug(1, ArmSide.RIGHT)
    plan_unplug(2, ArmSide.LEFT)

    # 2) Server extraction — both arms together. After pull-out the
    # server is lowered onto the cart's bottom shelf (off-robot, on the
    # robot's left side). The lift stays at server height through the
    # extract; the carry over to the cart is an arm motion, not a lift
    # drop.
    snap_l, snap_r = seed_at_lift(LAYOUT.lift.server)
    aux_server = {DataCenterAux.LIFT: LAYOUT.lift.server}
    aux_stow = {DataCenterAux.LIFT: LAYOUT.lift.stow}

    # Handle bar world x = server_body_center_x - (server.half[0] + 0.015)
    # = LAYOUT.server_front_x - 0.015. Targets aim for the handle bar centre
    # so the gripper fingers span the bar when TCP seats on it.
    handle_x = LAYOUT.server_front_x - 0.015
    handle_L = np.array([handle_x, -0.15, LAYOUT.server.slot_z])
    handle_R = np.array([handle_x, +0.15, LAYOUT.server.slot_z])
    approach_L = handle_L + np.array([-0.07, 0.0, 0.0])
    approach_R = handle_R + np.array([-0.07, 0.0, 0.0])
    pulled_L = handle_L + np.array([-0.32, 0.0, 0.0])
    pulled_R = handle_R + np.array([-0.32, 0.0, 0.0])
    # Cart bottom-shelf stow: arm centre target sits over the cart's
    # bottom shelf. Server then snaps to cart's stow-pos via the
    # SERVER_ON_CART_BOTTOM weld with explicit world pose, so the arm
    # tracking offset doesn't bleed into the final stow position.
    stow_world_pos = LAYOUT.old_server_stow_world_pos
    stow_z_world = float(stow_world_pos[2])
    # Cart-bottom-shelf stow: LEFT arm only — cart sits on the robot's
    # left-y side, well outside the right arm's reach envelope. The
    # right arm releases at the rack-pulled position and returns home
    # while the left arm carries solo to the cart.
    stow_L = np.array(
        [float(stow_world_pos[0]) + 0.0, float(stow_world_pos[1]) + 0.15, stow_z_world]
    )

    qL_app, _ = snap_l(approach_L)
    qL_at, _ = snap_l(handle_L)
    qL_out, _ = snap_l(pulled_L)
    qR_app, _ = snap_r(approach_R)
    qR_at, _ = snap_r(handle_R)
    qR_out, _ = snap_r(pulled_R)
    # Re-seed left-arm IK at the stow lift height (LIFT.stow=0.05)
    # before solving qL_stow — the IK's notion of arm-base z must
    # match the runtime's at the moment the arm reaches the stow
    # waypoint, otherwise the lift transition between extract (lift
    # at server height) and stow (lift at cart height) drops the arm
    # by ~23 cm of its IK-planned z.
    snap_l_stow, _ = seed_at_lift(LAYOUT.lift.stow)
    qL_stow, _ = snap_l_stow(stow_L)

    # SERVER GRASPING POLICY: only the LEFT arm welds to the server. The right
    # arm tracks the opposite handle kinematically for visual bimanual effect,
    # but activating two grasp welds (one per arm) over-constrains the server
    # body — any tiny mismatch in the two arms' IK solutions yanks the
    # server around as both welds pull in different directions (user
    # report: "the whole replacing the server phase is so messy with the
    # server flying around"). One rigid weld → server follows exactly one
    # arm, smooth carry. Durations on the pull / lower / carry / insert
    # phases bumped ~1.5× so the motion is visibly slow and deliberate.
    server_id = grippable_id("server")
    grasp_L = grasp_weld(ArmSide.LEFT, server_id)

    scripts[ArmSide.LEFT].extend(
        [
            Step("to server handle L", qL_app, "open", 1.8, aux_ctrl=aux_server),
            Step("at handle L", qL_at, "open", 0.8, aux_ctrl=aux_server),
            Step(
                "grip + extract L",
                qL_at,
                "closed",
                0.6,
                aux_ctrl=aux_server,
                attach_activate=(grasp_L,),
                attach_deactivate=(AttachmentWeldName.SERVER_IN_RACK,),
            ),
            Step("pull server L", qL_out, "closed", 2.5, aux_ctrl=aux_server),
            Step("carry to cart L", qL_stow, "closed", 2.5, aux_ctrl=aux_stow),
            Step(
                "stow on cart L",
                qL_stow,
                "open",
                0.8,
                aux_ctrl=aux_stow,
                # Pin server at the cart's bottom-shelf canonical pose,
                # not wherever the arm happens to be at the instant of
                # release. Explicit pose; immune to arm-tracking offset.
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
                attach_deactivate=(grasp_L,),
            ),
        ]
    )
    # Right arm during server extraction: bimanual visual at the rack
    # (right arm reaches the opposite handle and pulls alongside the
    # left), then releases and idles at home while the left arm
    # carries solo to the cart on the robot's left side. The cart is
    # out of the right arm's reach envelope — single-arm carry from
    # the pull-out point to the cart side is what the geometry allows.
    home_R_pose = HOME_ARM_Q.copy()
    scripts[ArmSide.RIGHT].extend(
        [
            Step("to server handle R", qR_app, "open", 1.8, aux_ctrl=aux_server),
            Step("at handle R", qR_at, "open", 0.8, aux_ctrl=aux_server),
            Step("hold handle R", qR_at, "open", 0.6, aux_ctrl=aux_server),
            Step("pull server R", qR_out, "open", 2.5, aux_ctrl=aux_server),
            # Right arm doesn't follow to cart; returns to home so it's
            # out of the way of left arm's carry path.
            Step("return R", home_R_pose, "open", 2.5, aux_ctrl=aux_stow),
            Step("idle at home R", home_R_pose, "open", 0.8, aux_ctrl=aux_stow),
        ]
    )

    # 3) New server pickup from cart top shelf (LEFT arm only; cart is on
    # robot's left side, out of right arm's reach envelope) + install in
    # rack (also LEFT-arm-driven — same single-arm carry policy as
    # extraction so we never have two grasp welds fighting on one body).
    snap_l, snap_r = seed_at_lift(LAYOUT.lift.pick_new)
    aux_pick = {DataCenterAux.LIFT: LAYOUT.lift.pick_new}

    cart_top_world = LAYOUT.new_server_initial_world_pos
    new_handle_L = np.array(
        [
            float(cart_top_world[0]),
            float(cart_top_world[1]),
            float(cart_top_world[2]),
        ]
    )
    new_approach_L = new_handle_L + np.array([0.0, 0.0, 0.10])  # 10 cm above
    qL_napp, _ = snap_l(new_approach_L)
    qL_ngrip, _ = snap_l(new_handle_L)

    snap_l, snap_r = seed_at_lift(LAYOUT.lift.server)
    # Single-arm install: only the LEFT arm reaches into the rack slot
    # for the new server (the cart pickup is left-arm-only by reach
    # geometry, so handing off to the right arm mid-carry isn't worth
    # the choreography).
    slot_L = np.array([handle_x, -0.05, LAYOUT.server.slot_z])
    pre_slot_L = slot_L + np.array([-0.12, 0.0, 0.0])
    qL_pre, _ = snap_l(pre_slot_L)
    qL_in, _ = snap_l(slot_L)

    new_server_id = grippable_id("new_server")
    grasp_new_L = grasp_weld(ArmSide.LEFT, new_server_id)

    scripts[ArmSide.LEFT].extend(
        [
            Step("to cart top L", qL_napp, "open", 1.8, aux_ctrl=aux_pick),
            Step("at cart top L", qL_ngrip, "open", 0.8, aux_ctrl=aux_pick),
            Step(
                "grip new + release cart L",
                qL_ngrip,
                "closed",
                0.6,
                aux_ctrl=aux_pick,
                attach_activate=(grasp_new_L,),
                attach_deactivate=(AttachmentWeldName.NEW_ON_CART_TOP,),
            ),
            Step("raise new L", qL_pre, "closed", 2.5, aux_ctrl=aux_server),
            Step("insert L", qL_in, "closed", 1.8, aux_ctrl=aux_server),
            Step(
                "seat in rack + release L",
                qL_in,
                "open",
                0.8,
                aux_ctrl=aux_server,
                # Pin new_server in the rack's upper slot at exactly the
                # canonical world pose, not wherever the arm happens to
                # be at this instant. Same explicit-pose pattern as the
                # cart-bottom stow above.
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
                attach_deactivate=(grasp_new_L,),
            ),
        ]
    )
    # Right arm stays at home through the new-server install (cart is
    # out of reach, install is single-arm). Five matching idle steps so
    # the right-arm timeline length stays in lockstep with the left.
    scripts[ArmSide.RIGHT].extend(
        [
            Step("idle R cart-app", home_R_pose, "open", 1.8, aux_ctrl=aux_pick),
            Step("idle R cart-at", home_R_pose, "open", 0.8, aux_ctrl=aux_pick),
            Step("idle R cart-grip", home_R_pose, "open", 0.6, aux_ctrl=aux_pick),
            Step("idle R raise", home_R_pose, "open", 2.5, aux_ctrl=aux_server),
            Step("idle R insert", home_R_pose, "open", 1.8, aux_ctrl=aux_server),
            Step("idle R seat", home_R_pose, "open", 0.8, aux_ctrl=aux_server),
        ]
    )

    # 4) Replug cables sequentially into NEW server (active side rotates).
    def plan_replug(port_idx: int, active_side: ArmSide) -> None:
        cable_id = grippable_id(f"cable{port_idx + 1}/connector")
        cable_grasp_weld = grasp_weld(active_side, cable_id)
        port_weld_new = _PORT_NEW[port_idx]

        snap_l, snap_r = seed_at_lift(LAYOUT.lift.cables)
        snap = snap_l if active_side == ArmSide.LEFT else snap_r

        # After unplug, cable was released at "pulled" pose = port - 14 cm
        # in x, +3 cm in z. That's where the cable sits now (gravity=0,
        # nothing's moved it). Replug grip aims for that same spot.
        port_pos = _port_world_pos(port_idx)
        # Regrip aligns with the unplug "pulled" position (4 cm forward
        # of port + 2 cm up) — that's where the connector dangles after
        # release. Approach + seated stay close so the rigid rod can
        # follow without tearing the port-CONNECT loop.
        regrip = port_pos + np.array([-0.04, 0.0, 0.02])
        approach = port_pos + np.array([-0.03, 0.0, 0.0])
        seated = port_pos + np.array([-0.005, 0.0, 0.0])

        q_regrip, _ = snap(regrip)
        q_a, _ = snap(approach)
        q_s, _ = snap(seated)

        aux = {DataCenterAux.LIFT: LAYOUT.lift.cables}
        active_steps = [
            Step(f"to c{port_idx + 1}", q_regrip, "open", 1.2, aux_ctrl=aux),
            Step(
                f"grip c{port_idx + 1}",
                q_regrip,
                "closed",
                0.5,
                aux_ctrl=aux,
                attach_activate=(cable_grasp_weld,),
            ),
            Step(f"approach port {port_idx + 1}", q_a, "closed", 1.0, aux_ctrl=aux),
            Step(
                f"plug port {port_idx + 1}",
                q_s,
                "closed",
                0.7,
                aux_ctrl=aux,
                attach_activate=(port_weld_new,),
            ),
            Step(
                f"release c{port_idx + 1}",
                q_s,
                "open",
                0.4,
                aux_ctrl=aux,
                attach_deactivate=(cable_grasp_weld,),
            ),
        ]
        scripts[active_side].extend(active_steps)
        idle(
            _other_side(active_side),
            f"wait replug{port_idx + 1}",
            [s.duration for s in active_steps],
            aux=aux,
        )

    plan_replug(0, ArmSide.LEFT)
    plan_replug(1, ArmSide.RIGHT)
    plan_replug(2, ArmSide.LEFT)

    # 5) Return home
    push_both("return home", 1.6, aux={DataCenterAux.LIFT: LAYOUT.lift.home})

    for side in ARM_PREFIXES:
        print(f"  [{side}] {len(scripts[side])} steps planned")

    apply_initial_state(model, data, arms, cube_body_ids)
    return scripts
