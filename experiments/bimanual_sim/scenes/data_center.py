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

from arm_handles import ArmHandles, ArmSide
from cameras import CameraRole
from ik import PositionOnly, solve_ik
from paths import D405_MESH_STL, D435I_XML
from robots.piper import attach_piper
from robots.tiago import load_tiago
from scene_base import CubeID, Position3, Step, make_cube_id
from scenes.data_center_layout import HOME_ARM_Q, IK_SEED_Q, LAYOUT
from welds import activate_attachment_weld

NAME = "data_center"
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)
# Grippable objects addressable via Step.weld_activate / weld_deactivate.
# Index order matters: the runner uses it as an int index into this list.
GRIPPABLES: tuple[str, ...] = (
    "cable1_connector",
    "cable2_connector",
    "cable3_connector",
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
    # Realsense D435i mounted on top of the torso lift, pointing forward-down.
    ("top_d435i_cam", CameraRole.TOP),
    # Realsense D405 on each arm's link6, along the gripper axis.
    ("left_wrist_d405_cam", CameraRole.WRIST),
    ("right_wrist_d405_cam", CameraRole.WRIST),
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
    # Rack shelves span the whole interior between the surrounding panels.
    ("rack_rear", "rack_shelf"),
    ("rack_side_L", "rack_shelf"),
    ("rack_side_R", "rack_shelf"),
    ("rack_rear", "rack_lower_shelf"),
    ("rack_side_L", "rack_lower_shelf"),
    ("rack_side_R", "rack_lower_shelf"),
    # Cable bracket is a decorative bar bolted flush against the +y side.
    ("cable_bracket", "rack_side_R"),
    # Each cable anchor body sits inside the bracket (that's how the cable
    # emerges from the fixture), and the composite's first-segment body
    # (`cable{i}_B_first`) is at the anchor origin — both overlap the
    # bracket and each other by design.
    ("cable_bracket", "cable1_anchor"),
    ("cable_bracket", "cable2_anchor"),
    ("cable_bracket", "cable3_anchor"),
    ("cable_bracket", "cable1_B_first"),
    ("cable_bracket", "cable2_B_first"),
    ("cable_bracket", "cable3_B_first"),
    ("cable1_anchor", "cable1_B_first"),
    ("cable2_anchor", "cable2_B_first"),
    ("cable3_anchor", "cable3_B_first"),
    # The cable composite's first body also crosses the rack side panel
    # (the bracket sits outside +y; the first segment extends back into
    # the rack toward the port).
    ("rack_side_R", "cable1_B_first"),
    ("rack_side_R", "cable2_B_first"),
    ("rack_side_R", "cable3_B_first"),
    # Top camera sits directly on top of the camera pole — the pole-top
    # and mesh-bottom AABBs touch by design. The camera-mesh body
    # contains ~9 sub-geoms so a body-pair allow covers the lot.
    ("world", "top_d435i"),
    # TIAGo's base mesh uses a conservative sphere bound in scene_check
    # (rbound ≈ 0.30 m at mesh origin z≈0.16), so after pulling the rack
    # closer to center_x=0.58 the rbound sphere grazes the rack walls
    # even though the actual mesh geometry is 25+ cm away. Body-pair
    # allow since the base mesh fuses into the `world` body.
    ("world", "rack_side_L"),
    ("world", "rack_side_R"),
    ("world", "rack_bottom"),
    ("world", "rack_lower_shelf"),
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
    SERVER_ON_LOWER_SHELF = "weld_server_on_lower_shelf"
    NEW_IN_BIN = "weld_new_in_bin"
    NEW_IN_RACK = "weld_new_in_rack"


class ConstraintKind(StrEnum):
    """Which MuJoCo equality type backs an entry in `ATTACHMENTS`."""

    WELD = "weld"  # mjEQ_WELD — pins full pose (position + orientation)
    CONNECT = "connect"  # mjEQ_CONNECT — pins only a point (like a ball joint)


@dataclass(frozen=True)
class _AttachmentWeldSpec:
    """One scene body-pair equality. The registry below is the single source
    of truth for the name, the two bodies, the initial active flag, and the
    constraint kind — consumed by `build_spec` (create the equality),
    `_resolve_scene_ids` (collect the eq id), and `apply_initial_state` (reset
    to the spec default).
    """

    name: AttachmentWeldName
    body_a: str
    body_b: str
    initially_active: bool
    kind: ConstraintKind = ConstraintKind.WELD
    # Only meaningful when kind == CONNECT: the anchor point in body_a's
    # local frame that body_b's origin should be pinned to. For port
    # connects this is the port geom's position on the server body.
    connect_anchor_in_a: tuple[float, float, float] = (0.0, 0.0, 0.0)


# Port anchor in server's local frame. Server and new_server share identical
# port layouts, so both reuse this tuple when a CONNECT equality is built.
# Delegated to `LAYOUT.port_anchor_in_server_frame` so the port geom position
# and the connect anchor stay in lockstep.
def _port_anchor_in_server_frame(port_idx: int) -> tuple[float, float, float]:
    return LAYOUT.port_anchor_in_server_frame(port_idx)


ATTACHMENTS: tuple[_AttachmentWeldSpec, ...] = (
    # Port connects — body_a is the SERVER (anchor is in server's local
    # frame); body_b is the cable connector whose origin gets pinned.
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT1_OLD,
        "server",
        "cable1_connector",
        True,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(0),
    ),
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT2_OLD,
        "server",
        "cable2_connector",
        True,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(1),
    ),
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT3_OLD,
        "server",
        "cable3_connector",
        True,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(2),
    ),
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT1_NEW,
        "new_server",
        "cable1_connector",
        False,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(0),
    ),
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT2_NEW,
        "new_server",
        "cable2_connector",
        False,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(1),
    ),
    _AttachmentWeldSpec(
        AttachmentWeldName.PORT3_NEW,
        "new_server",
        "cable3_connector",
        False,
        kind=ConstraintKind.CONNECT,
        connect_anchor_in_a=_port_anchor_in_server_frame(2),
    ),
    # Full-pose welds — these need orientation pinning so a stowed server
    # keeps its rack-aligned orientation and doesn't roll around.
    _AttachmentWeldSpec(AttachmentWeldName.SERVER_IN_RACK, "server", "rack_frame", True),
    # Old server gets staged on the rack's lower shelf (not a torso bin) —
    # the robot's cart only carries the NEW server at init. Weld to
    # rack_frame (static) so the stowed old server stays put when the
    # robot lifts.
    _AttachmentWeldSpec(AttachmentWeldName.SERVER_ON_LOWER_SHELF, "server", "rack_frame", False),
    _AttachmentWeldSpec(AttachmentWeldName.NEW_IN_BIN, "new_server", "torso_lift_link", True),
    _AttachmentWeldSpec(AttachmentWeldName.NEW_IN_RACK, "new_server", "rack_frame", False),
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
    """Name of the per-arm grasp weld for a given cube. Matches the
    `{prefix}grasp_cube{i}` convention used when the arm handles are built
    in `arm_handles.get_arm_handles`."""
    return f"{side}grasp_cube{cube_id}"


# -----------------------------------------------------------------------------
# Spec construction
# -----------------------------------------------------------------------------


def _static_box(parent, name: str, pos, half, rgba):
    body = parent.add_body(name=name, pos=list(pos))
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), rgba=list(rgba))
    return body


def _add_bin(parent, name_prefix: str, local_pos) -> None:
    """Open shelf — a single floor plate.

    The bin used to be a U of walls (back + 2 sides + floor) but the back
    was visually wider than the torso shell (protruded out the robot's
    sides) and the side walls sat in the path of the piper shoulder/elbow
    on cable-reach poses — compile-time contacts made the arm physically
    unable to track its IK-planned q, so TCP settled ~28 cm above the
    planned height. Server stow is a weld, not contact-based, so none of
    those walls buy physical correctness; only the floor remains for the
    visual of a shelf holding a server.
    """
    x, y, z = local_pos
    bx, by, bz = LAYOUT.bins.half
    wall_t = LAYOUT.bins.wall_thickness
    rgba = (0.72, 0.72, 0.75, 1.0)
    _static_box(parent, f"{name_prefix}_floor", (x, y, z - bz - wall_t), (bx, by, wall_t), rgba)


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


def _add_cable(spec: mujoco.MjSpec, rack_body, cable_idx: int, port_y: float, rgba) -> None:
    """Build a cable via MuJoCo's native composite + elasticity plugin.

    Previously hand-rolled as a 7-segment ball-jointed capsule chain. The
    native `composite type="cable"` with the `mujoco.elasticity.cable`
    plugin implements Discrete Elastic Rods — proper bend/twist stiffness
    in SI units (Pa) and a maintained first-party plugin rather than hand-
    tuned ball damping. `composite type="rope"` was removed in MuJoCo 3.x;
    the cable plugin is the canonical replacement.

    The composite tip body inside the sub-spec is renamed to `connector`
    before attach so, with prefix `cable{i+1}_`, it resolves as
    `cable{i+1}_connector` in the compiled model — matching the name the
    grasp welds and `ATTACHMENTS` registry reference. The per-port distinct
    connector geom (box / flat rectangle / cylinder) is added to that same
    body so it moves with the last segment.

    The attach frame's `quat` rotates the composite's default +x layout to
    point from the shared side-mounted bracket toward the port. Per-cable
    stagger along rack +x comes from `LAYOUT.cable_anchor_world`.
    """
    cables_cfg = LAYOUT.cables
    rack = LAYOUT.rack
    world_anchor = LAYOUT.cable_anchor_world(cable_idx)
    rack_origin = np.array([rack.center_x, 0.0, rack.center_z])
    local_anchor = world_anchor - rack_origin

    # Connector tip needs to land at the port geom's world position so the
    # port connect equality anchors the connector seated inside the socket.
    world_port = LAYOUT.port_world_pos(list(LAYOUT.ports.y_offsets).index(port_y))
    direction = world_port - world_anchor
    direction_len = float(np.linalg.norm(direction))
    # MuJoCo's composite cable with count=N creates N-1 actual bodies spanning
    # `size * (N-2)/(N-1)` along its +x axis (the first slot is consumed by
    # the parent anchor body, and the last body lands one segment short of
    # `size`). Empirically, count=11 → 10 bodies → last body at 0.9 × size.
    # To make the tip (B_last) land AT the port, inflate `cable_len` by 10/9.
    # Without this, the tip sits 4–5 cm short of the port at init and the
    # connect equality has to yank the cable on every reset, which makes
    # "plugged in" look like a cable straining toward the socket.
    count = cables_cfg.n_seg + 1
    composite_span_fraction = (count - 2) / (count - 1)  # 9/10 for count=11
    nominal_cable_len = direction_len / composite_span_fraction
    cable_len = min(nominal_cable_len, LAYOUT.cable_max_len)
    attach_quat = _quat_align_x_to(direction)
    r, g, b, a = rgba
    cable_xml = f"""<mujoco>
      <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
      </extension>
      <worldbody>
        <body name="anchor">
          <geom type="box" size="0.015 0.010 0.010" rgba="0.3 0.3 0.3 1"/>
          <composite type="cable" curve="s" count="{count} 1 1"
                     size="{cable_len:.6f}" offset="0 0 0" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <!-- 1e6 Pa bend / 2e6 twist: softer than the 2e7/5e7 first
                   used (Cat-6-stiff), which resisted the arm-yank at
                   grip-replug hard enough that the ball-joint chain
                   couldn't absorb the ~40 cm offset between the cable's
                   settled rest pose and the arm's dangle target — QACC
                   blew up at t=43.7 s. 1e6 still keeps the cable mostly
                   straight under its own weight but lets the chain
                   stretch without numerical instability. -->
              <config key="twist" value="2e6"/>
              <config key="bend" value="1e6"/>
              <config key="vmax" value="0.05"/>
            </plugin>
            <joint kind="main" damping="2.0"/>
            <geom type="capsule" size="{cables_cfg.seg_radius}"
                  rgba="{r} {g} {b} {a}" mass="0.02"/>
          </composite>
        </body>
      </worldbody>
    </mujoco>"""

    cable_spec = mujoco.MjSpec.from_string(cable_xml)
    # Rename the composite tip so it resolves as cable{i+1}_connector after
    # the attach prefix; ATTACHMENTS + grasp welds reference that exact name.
    tip = cable_spec.body("B_last")
    tip.name = "connector"

    # Spherical connector head centred on the tip body's origin. A sphere is
    # orientation-invariant, so it sits cleanly inside the port socket geom
    # regardless of which direction the cable routes in from — no matter
    # whether the tip body's +x ends up pointing along world -y (bracket on
    # the side), +x (frontal), or anywhere else, the rendered plug stays
    # seated. Previously each cable had a directional box/rectangle/cylinder
    # offset along tip +x by CONN_LEN/2, which made the plug appear crooked
    # next to its port when the cable direction didn't align with the
    # server's +x axis. Port visual distinction is still carried by the
    # socket geoms on the server (box / flat box / cylinder) + matching
    # colour on the sphere.
    tip.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.0],
        size=[0.014],
        rgba=list(rgba),
        mass=0.02,
    )

    attach_frame = rack_body.add_frame(pos=list(local_anchor), quat=list(attach_quat))
    spec.attach(cable_spec, prefix=f"cable{cable_idx + 1}_", frame=attach_frame)


def build_spec() -> mujoco.MjSpec:
    # Start from PAL TIAGo (wheels + torso_lift_link). `robots.tiago.load_tiago`
    # strips the upstream single arm + head, removes the `reference` freejoint
    # (required for mink IK — see that module's docstring), and prunes
    # now-orphan contact excludes. All customizations are declared in its
    # `TiagoConfig`; defaults are what the data-center scene needs.
    spec = load_tiago()
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.timestep = 0.002
    # Puppet mode: gravity = 0. The arms are qpos-driven (no PD tracking,
    # no need for gravity to load the actuators), and the freejoint
    # bodies (servers, cable connectors) are always pinned by some weld
    # — gravity would just be solver noise. Removing it eliminates the
    # entire class of "server drifted while a weld was being captured"
    # bugs and lets the cable composite settle to its rest pose without
    # arm-yank instability.
    spec.option.gravity = [0.0, 0.0, 0.0]

    # Default class for visual-only geoms: no contact, zero mass. Geoms that
    # pass `default=visual` inherit these flags, so we don't repeat the
    # contype=0 / conaffinity=0 / mass=0.0 trio at every call site (shoulder
    # plate, camera pole, D405 wrist meshes, rack panels). Using MJCF's
    # native `<default>` mechanism via MjSpec instead of a Python-level kwargs
    # shim keeps the defaults visible in `spec.to_xml()` output for anyone
    # inspecting the compiled model. `add_geom(default=...)` is the MjSpec
    # Python binding for `<geom class="visual"/>`.
    visual = spec.add_default("visual", spec.default)
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.mass = 0.0

    wb = spec.worldbody

    # Floor (TIAGo's upstream MJCF has no floor — the per-variant scene.xml adds it)
    wb.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.78, 0.78, 0.80, 1.0],
    )

    # Reference to the moving lift body — everything arm-mount-and-bin-related
    # hangs off this so it rides the torso slide joint.
    torso = spec.body("torso_lift_link")

    # ---------------------- Onboard bin (carries new_server) ----------------
    # One compartment on the torso, holding the new_server at init. Old
    # server gets staged on the rack's lower shelf — not another torso bin
    # — so the robot stays visually simpler and the demo's "robot brings
    # the new server, leaves the old on the rack" story is readable at a
    # glance.
    _add_bin(torso, "new_bin", (LAYOUT.bins.local_x, 0.0, LAYOUT.bins.new_local_z))

    # ---------------------- Piper arms on the torso --------------------------
    # Arms sit:
    #   - well forward of the torso column so the compact home pose keeps TCP
    #     clear of the rack front face at startup and the arm base is close
    #     enough to the rack that cable/server targets stay inside Piper's
    #     ~0.55 m reach envelope
    #   - outboard of the bins (|y| > bin half-width) so the piper base link
    #     doesn't intersect the bin side walls
    #   - below torso origin so TCP ends up at rack-slot height and the arms
    #     visibly occupy the space BETWEEN the chest and waist bins
    # All three offsets live in `LAYOUT.arm_mount`.
    arm_mount = LAYOUT.arm_mount
    lf = torso.add_frame(
        pos=[arm_mount.x, -arm_mount.y_abs, arm_mount.z], quat=[1.0, 0.0, 0.0, 0.0]
    )
    attach_piper(spec, prefix=ArmSide.LEFT, frame=lf)
    rf = torso.add_frame(pos=[arm_mount.x, arm_mount.y_abs, arm_mount.z], quat=[1.0, 0.0, 0.0, 0.0])
    attach_piper(spec, prefix=ArmSide.RIGHT, frame=rf)

    # Torso cladding — one chunky visual box enclosing the piper base links
    # and the back of the two bins. Depth is fixed ±0.10 m around the torso
    # column (20 cm front-to-back) so it reads as a proportionate humanoid
    # torso rather than ballooning whenever arm_mount.x moves; width is
    # 6 cm inboard of arm_mount.y_abs so the piper base is still visible
    # on the outside of the cladding (arms stick out of the side).
    # Vertical extent spans the bin bottoms and tops so the bins look
    # integrated with the body rather than floating above/below it.
    _clad_x_min, _clad_x_max = -0.10, 0.10
    # Cladding top tracks the (only) bin's top; bottom is a fixed torso-
    # local offset below the arm mount so the cladding reads as "torso
    # reaches all the way down to the wheel platform" rather than dangling
    # above empty space once we dropped the old_bin.
    _clad_z_min = arm_mount.z - 0.20
    _clad_z_max = LAYOUT.bins.new_local_z + LAYOUT.bins.half[2]
    torso.add_geom(
        default=visual,
        name="torso_cladding",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[
            (_clad_x_min + _clad_x_max) * 0.5,
            0.0,
            (_clad_z_min + _clad_z_max) * 0.5,
        ],
        size=[
            (_clad_x_max - _clad_x_min) * 0.5,
            arm_mount.y_abs - 0.06,
            (_clad_z_max - _clad_z_min) * 0.5,
        ],
        rgba=[0.78, 0.78, 0.82, 1.0],
    )

    # TCP sites on each arm's link6 (required by ik.solve_ik / welds).
    for side in ARM_PREFIXES:
        link6 = spec.body(f"{side}link6")
        link6.add_site(name=f"{side}tcp", pos=[0.0, 0.0, 0.14], size=[0.006, 0.006, 0.006])

    # ---------------------- Top camera pole (static, above the arms) --------
    # TIAGo's head is stripped in `load_tiago_without_arm` (user asked to
    # remove it — we don't actuate it, and the head geoms were competing with
    # the rack for vertical space). We replace it with a tall static pole
    # rigidly welded to `base_link` (not the moving `torso_lift_link`), so
    # the top camera view stays stable regardless of the torso lift qpos.
    # The D435i mounts at the top and points at `rack_frame` via TARGETBODY,
    # so the optical axis always aims at the rack irrespective of how the
    # base is parked.
    #
    # `TOP_POLE_X = -0.20` places the pole behind the onboard bins (which
    # span world x ∈ [-0.012, 0.388] via their torso-local +x mount). Running
    # the pole forward of that would visibly clip through the new/old_bin
    # compartments and through whichever server they hold (user report: "the
    # tall stand... should be moved further back into the robotic body so
    # that it doesn't clip into the server compartments").
    TOP_POLE_X = -0.20
    base = spec.body("base_link")
    # Pole top at z = 1.48 (half = 0.74, pos = 0.74, so span 0 → 1.48).
    # Camera body attaches at z = 1.485 with the 90°-x rotation below
    # making the mesh ~2.6 cm tall vertically — so mesh-bottom (~1.472)
    # sits ~0.8 cm *inside* the pole top, reading as "camera mounted on
    # the pole" rather than floating above a gap. The overlap is
    # allowlisted (see ALLOWED_STATIC_OVERLAPS).
    base.add_geom(
        default=visual,
        name="top_camera_pole",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[TOP_POLE_X, 0.0, 0.74],
        size=[0.025, 0.025, 0.74],
        rgba=[0.32, 0.32, 0.35, 1.0],
    )
    # D435i's native mesh frame has local +z as the long (~9.2 cm) camera-
    # body axis and local +x as the lens direction. At identity orientation
    # the camera stood on end like a pencil — visually wrong. 90° about +x
    # (quat = [cos 45°, sin 45°, 0, 0]) rotates local +z → world -y, so the
    # long axis is horizontal; +x (lens) stays pointing world +x (at the
    # rack, which is at world +x from the pole). After rotation the mesh
    # is ~2.6 cm tall vertically, so attach z = 1.485 puts the mesh bottom
    # flush with the pole top instead of floating 3 cm above it.
    top_cam_frame = base.add_frame(
        pos=[TOP_POLE_X, 0.0, 1.485],
        quat=[0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
    )
    spec.attach(
        mujoco.MjSpec.from_file(str(D435I_XML)),
        prefix="top_",
        frame=top_cam_frame,
    )
    spec.body("top_d435i").add_camera(
        name="top_d435i_cam",
        pos=[0.0, 0.0, 0.0],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
        targetbody="rack_frame",
        fovy=69.0,  # D435i colour-sensor horizontal FOV ≈ 69°
    )

    # ---------------------- Wrist cameras: Realsense D405 --------------------
    # Load the D405 mesh once as a shared asset; reference it on each link6.
    # Camera uses FIXED mode with a 180°-about-x quat so its optical axis
    # (MuJoCo cam -z) aligns with link6's +z (the tool axis — TCP sits along
    # link6 +z at [0, 0, 0.14]). Switched away from TARGETBODY targetting
    # `link8`: link8 is one finger (asymmetric w.r.t. link6 +z), so both
    # wrist cams pointed the same way in world regardless of side — not
    # useful for a bimanual scene. FIXED along +z is symmetric and always
    # aimed at the grasp point.
    spec.add_mesh(name="d405", file=str(D405_MESH_STL))
    # Wrist D405: mesh was 8 cm along link6 +z with a 180°-about-x quat on
    # the geom. That put the mesh forward of the gripper fingertips and
    # flipped its orientation (user report: "wrist cameras floating out
    # of the grippers"). The D405 STL's frame has +z as the back mount
    # face and +y as the lens axis, so placing the mesh at the link6
    # body's outer surface (z ≈ 0.03) with identity orientation keeps
    # the mount side against link6 and the lens pointing sideways, which
    # is how the real wrist cam sits on an Aloha-style Piper.
    #
    # The CAMERA sensor stays at the TCP site's axis with a 180°-x quat
    # — MuJoCo cam -z maps to link6 +z so the sensor looks *out* along
    # the gripper axis toward whatever the arm is reaching for, which is
    # what the scene's task plan expects. Mesh visualisation and camera
    # sensor direction are independent here.
    WRIST_MESH_OFFSET = [0.0, 0.0, 0.03]
    for side, cam_name in (
        (ArmSide.LEFT, "left_wrist_d405"),
        (ArmSide.RIGHT, "right_wrist_d405"),
    ):
        link6 = spec.body(f"{side}link6")
        link6.add_geom(
            default=visual,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="d405",
            pos=WRIST_MESH_OFFSET,
            rgba=[0.12, 0.12, 0.14, 1.0],
        )
        link6.add_camera(
            name=f"{cam_name}_cam",
            pos=[0.0, 0.0, 0.08],  # optical centre halfway to TCP along gripper axis
            quat=[0.0, 1.0, 0.0, 0.0],  # 180°x: cam -z aligned with link6 +z
            mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
            fovy=87.0,  # D405 FOV
        )

    # ---------------------- Rack (static, open front) ------------------------
    # A 19" cabinet built as 5 panels (rear + 2 sides + top + bottom) with the
    # FRONT FULLY OPEN so arms can reach in. Previously a solid box — looked
    # like a closed-door cabinet. We also add a cosmetic shelf at server
    # height and a hinged-open door tucked to the side for flavor.
    rack_cfg = LAYOUT.rack
    rack = wb.add_body(name="rack_frame", pos=[rack_cfg.center_x, 0.0, rack_cfg.center_z])
    rack_wall_t = rack_cfg.wall_thickness
    panel_rgba = (0.10, 0.10, 0.13, 1.0)
    rhx, rhy, rhz = rack_cfg.half

    def _rack_panel(name: str, pos, half) -> None:
        b = rack.add_body(name=name, pos=list(pos))
        # Visual-only (class "visual"): the server is weld-attached so it
        # doesn't need a shelf to rest on, and arms reach through the open
        # front. Letting panels collide just creates solver noise around the
        # server body.
        b.add_geom(
            default=visual,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(half),
            rgba=list(panel_rgba),
        )

    _rack_panel("rack_rear", (rhx - rack_wall_t, 0.0, 0.0), (rack_wall_t, rhy, rhz))
    _rack_panel("rack_side_L", (0.0, -rhy + rack_wall_t, 0.0), (rhx, rack_wall_t, rhz))
    _rack_panel("rack_side_R", (0.0, rhy - rack_wall_t, 0.0), (rhx, rack_wall_t, rhz))
    _rack_panel("rack_top", (0.0, 0.0, rhz - rack_wall_t), (rhx, rhy, rack_wall_t))
    _rack_panel("rack_bottom", (0.0, 0.0, -rhz + rack_wall_t), (rhx, rhy, rack_wall_t))
    # Cosmetic shelf just below the server slot.
    server_cfg = LAYOUT.server
    server_local_z = server_cfg.slot_z - rack_cfg.center_z
    _rack_panel(
        "rack_shelf",
        (0.0, 0.0, server_local_z - server_cfg.half[2] - rack_wall_t),
        (rhx - rack_wall_t, rhy - rack_wall_t, rack_wall_t),
    )
    # Lower staging shelf — fully inside the rack (same footprint as the
    # cosmetic rack_shelf above). Earlier iteration extended 20 cm forward
    # of the rack front to bring the stow target into Piper reach; user
    # read that as "shelf protruding too much". The arm now stows the
    # server at the upper-slot x (rack interior) instead — see the stow
    # target retargeting in make_task_plan.
    lower_shelf_local_z = rack_cfg.lower_shelf_z_world - rack_cfg.center_z
    _rack_panel(
        "rack_lower_shelf",
        (0.0, 0.0, lower_shelf_local_z - server_cfg.half[2] - rack_wall_t),
        (rhx - rack_wall_t, rhy - rack_wall_t, rack_wall_t),
    )
    # Cable management bracket — a horizontal bar bolted to the rack's +y
    # (right) outside wall. All three cables anchor to this bar (spread
    # along rack +x via `LAYOUT.cable_anchor_world`).
    bracket_center = LAYOUT.cable_bracket_center
    cable_bracket_local_x = bracket_center[0] - rack_cfg.center_x
    cable_bracket_local_z = bracket_center[2] - rack_cfg.center_z
    cable_bracket_local_y = rhy + 0.010  # just outside the side panel
    _rack_panel(
        "cable_bracket",
        (cable_bracket_local_x, cable_bracket_local_y, cable_bracket_local_z),
        (0.06, 0.010, 0.020),  # 12 cm long (x) × 2 cm deep (y) × 4 cm tall (z)
    )
    # (No door — the rack is open-front. An earlier cosmetic hinged-open door
    # was clipping into the robot's workspace; not worth the geometry fiddle
    # for a purely visual element. Many real 19" racks run doorless anyway.)

    # Server: freejoint bodies must be direct children of worldbody.
    port_local_x = LAYOUT.port_local_x_on_server
    port_y_offsets = LAYOUT.ports.y_offsets
    port_colors = LAYOUT.ports.colors
    # contype=0 conaffinity=0: the server is entirely weld-driven (to the
    # rack, to the lower staging shelf, or to the arm's grasp weld) — it
    # never needs contact-based physics. Leaving default contacts on
    # produced a QACC blowup at ~t=44 s because the inserted server's AABB
    # grazed the rack's lower shelf + rack_shelf + rack_bottom, and the
    # constraint solver couldn't simultaneously hold the weld AND resolve
    # the contacts.
    server = wb.add_body(name="server", pos=list(LAYOUT.server_world_pos_in_rack))
    server.add_freejoint()
    server.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(server_cfg.half),
        rgba=[0.16, 0.17, 0.19, 1.0],
        mass=server_cfg.mass,
        contype=0,
        conaffinity=0,
    )
    for _side, y_off in (("l", -0.15), ("r", +0.15)):
        server.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-server_cfg.half[0] - 0.015, y_off, 0.0],
            size=[0.015, 0.015, 0.025],
            rgba=[0.85, 0.85, 0.85, 1.0],
        )
    _shapes = ("box", "box", "cylinder")
    for i, (y, rgba, shape) in enumerate(zip(port_y_offsets, port_colors, _shapes, strict=True)):
        if shape == "box":
            # i == 0: power (square); else: net (flatter rectangle).
            size = [0.010, 0.018, 0.018] if i == 0 else [0.010, 0.012, 0.007]
            server.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[port_local_x, y, 0.0],
                size=size,
                rgba=list(rgba),
            )
        else:  # cylinder — fiber (fromto only; MuJoCo forbids pos+fromto)
            server.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[port_local_x - 0.010, y, 0.0, port_local_x + 0.005, y, 0.0],
                size=[0.010],
                rgba=list(rgba),
            )

    # Cables (anchored to rack, terminating in connector bodies).
    for i, (y, rgba) in enumerate(zip(port_y_offsets, port_colors, strict=True)):
        _add_cable(spec, rack, i, y, rgba)

    # ---------------------- Replacement server -------------------------------
    # Place the freejoint body resting on the new_bin floor at initial lift
    # height, so the initial NEW_IN_BIN weld captures a realistic relative
    # pose. `LAYOUT.new_server_initial_world_pos` computes this from the
    # TIAGo torso pose, the upper-bin offset, and the server's own half-extent.
    new_server = wb.add_body(name="new_server", pos=list(LAYOUT.new_server_initial_world_pos))
    new_server.add_freejoint()
    new_server.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(server_cfg.half),
        rgba=[0.24, 0.26, 0.30, 1.0],
        mass=server_cfg.mass,
        contype=0,
        conaffinity=0,
    )
    for _side, y_off in (("l", -0.15), ("r", +0.15)):
        new_server.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-server_cfg.half[0] - 0.015, y_off, 0.0],
            size=[0.015, 0.015, 0.025],
            rgba=[0.85, 0.85, 0.85, 1.0],
        )
    for i, (y, rgba, shape) in enumerate(zip(port_y_offsets, port_colors, _shapes, strict=True)):
        if shape == "box":
            size = [0.010, 0.018, 0.018] if i == 0 else [0.010, 0.012, 0.007]
            new_server.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[port_local_x, y, 0.0],
                size=size,
                rgba=list(rgba),
            )
        else:
            new_server.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[port_local_x - 0.010, y, 0.0, port_local_x + 0.005, y, 0.0],
                size=[0.010],
                rgba=list(rgba),
            )

    # ---------------------- Actuators ----------------------------------------
    # Position actuator on TIAGo's existing torso_lift_joint. TIAGo's upstream
    # MJCF ships only the joint (no actuator); we add the position servo here.
    # kp=80000 / kv=800 holds a ~5 kg arm-and-bin payload with <1 mm static
    # droop. Earlier kp=5000 produced ~4 cm of sag under the same payload
    # (force / kp), which landed IK-planned targets above the cable ports
    # by enough to miss the connectors entirely at grip time.
    spec.add_actuator(
        name=DataCenterAux.LIFT,  # -> "torso_lift_joint"
        target=DataCenterAux.LIFT,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[80000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0.0, -80000.0, -800.0, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=True,
        ctrlrange=[0.0, LAYOUT.tiago.lift_range],
    )

    # ---------------------- Welds (grasp + attachment) ------------------------
    # Grasp welds: (arm, grippable_object). All inactive at start. Names must
    # match the `{prefix}grasp_cube{i}` convention that `get_arm_handles`
    # expects when it builds `ArmHandles.weld_ids`.
    for side in ARM_PREFIXES:
        hand = f"{side}link6"
        for i, obj_name in enumerate(GRIPPABLES):
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name=f"{side}grasp_cube{i}",
                name1=hand,
                name2=obj_name,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=False,
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            )

    # Attachment equalities — single iteration over ATTACHMENTS registry.
    # Welds pin full pose; connects pin only body_b's origin to a point in
    # body_a's frame (used for port↔cable connections so the cable can
    # swivel naturally at the plug).
    for attachment in ATTACHMENTS:
        if attachment.kind is ConstraintKind.CONNECT:
            ax, ay, az = attachment.connect_anchor_in_a
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name=attachment.name,
                name1=attachment.body_a,
                name2=attachment.body_b,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=attachment.initially_active,
                # eq_data for mjEQ_CONNECT: [anchor_xyz (in body1 frame), pad...].
                data=[ax, ay, az, 0, 0, 0, 0, 0, 0, 0, 0],
                # Tight solver reference so the plug stays seated instead of
                # drifting under cable weight. Default (0.02, 1.0) lets the
                # connector sag 2 cm under gravity; 2 ms time-constant makes
                # it effectively rigid for our 2 ms physics step.
                solref=[0.002, 1.0],
                solimp=[0.99, 0.999, 1e-6, 0.5, 2.0],
            )
        else:  # WELD
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name=attachment.name,
                name1=attachment.body_a,
                name2=attachment.body_b,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                active=attachment.initially_active,
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            )

    # Exclude fragile contacts that would otherwise thrash.
    # Piper bases sit on the torso; without exclude the shared contact noise
    # produces oscillation around the arm-mount point.
    spec.add_exclude(bodyname1="torso_lift_link", bodyname2="left_base_link")
    spec.add_exclude(bodyname1="torso_lift_link", bodyname2="right_base_link")
    # new_server starts inside the upper bin on the torso — weld-held but the
    # bin walls would otherwise contact it on every integrator step.
    spec.add_exclude(bodyname1="torso_lift_link", bodyname2="new_server")
    spec.add_exclude(bodyname1="rack_frame", bodyname2="server")

    return spec


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
        if attachment.kind is ConstraintKind.CONNECT:
            data.eq_active[eq_id] = 1
        else:  # WELD
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
        cable_id = grippable_id(f"cable{port_idx + 1}_connector")
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
            model, mujoco.mjtObj.mjOBJ_BODY, f"cable{port_idx + 1}_connector"
        )
        connector_pos = np.asarray(data.xpos[connector_bid], dtype=float).copy()

        approach = connector_pos + np.array([-0.10, 0.0, 0.0])
        at_conn = connector_pos
        pulled = connector_pos + np.array([-0.14, 0.0, 0.03])
        side_y = -0.28 if active_side == ArmSide.LEFT else +0.28
        dangle = np.array([LAYOUT.server_front_x - 0.28, side_y, LAYOUT.server.slot_z - 0.10])

        q_approach, _ = snap(approach)
        q_at, _ = snap(at_conn)
        q_pulled, _ = snap(pulled)
        q_dangle, _ = snap(dangle)

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
            Step(f"pull c{port_idx + 1} free", q_pulled, "closed", 0.9, aux_ctrl=aux),
            Step(f"park c{port_idx + 1}", q_dangle, "closed", 1.0, aux_ctrl=aux),
            Step(
                f"release c{port_idx + 1}",
                q_dangle,
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

    # 2) Server extraction — both arms together. After pull-out the server
    # is lowered onto the rack's lower staging shelf (not a torso bin);
    # the lift drops to put the arms at shelf height so the drop-off is
    # the same short downward motion it used to be for the bin stow.
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
    # Lower-shelf stow: target the rack's upper-slot x so the server
    # ends up INSIDE the rack on the lower shelf (same depth as the
    # live slot above), rather than dangling 20 cm in front. The arm's
    # 0.36 m forward reach at LIFT.stow (arm base world z≈0.79, target
    # z≈0.64, 0.15 m y inboard) fits inside Piper's ~0.55 m envelope.
    stow_z_world = LAYOUT.rack.lower_shelf_z_world + LAYOUT.server.half[2] + 0.015
    # 6 cm forward of the upper-slot handle so the reach stays inside
    # Piper's envelope at the lower stow height (target z 0.64 vs arm
    # base 0.79 — the 15 cm vertical drop eats into horizontal reach).
    stow_handle_x = handle_x - 0.06
    stow_L = np.array([stow_handle_x, -0.15, stow_z_world])
    stow_R = np.array([stow_handle_x, +0.15, stow_z_world])

    qL_app, _ = snap_l(approach_L)
    qL_at, _ = snap_l(handle_L)
    qL_out, _ = snap_l(pulled_L)
    qR_app, _ = snap_r(approach_R)
    qR_at, _ = snap_r(handle_R)
    qR_out, _ = snap_r(pulled_R)
    # Stow targets are reached at LIFT.stow (lift_qpos=0.05), not
    # LIFT.server (0.28) where the rest of the extraction targets are
    # solved. Re-seed snap at LIFT.stow before solving the stow IK so
    # the IK's notion of arm-base z matches the runtime's at the moment
    # the arm reaches the stow waypoint. Without this, qL_stow's TCP
    # ends up 23 cm below the planned target (the lift drops between
    # IK time and runtime, dragging the arm down with it).
    snap_l_stow, snap_r_stow = seed_at_lift(LAYOUT.lift.stow)
    qL_stow, _ = snap_l_stow(stow_L)
    qR_stow, _ = snap_r_stow(stow_R)

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
            Step("lower to shelf L", qL_stow, "closed", 2.5, aux_ctrl=aux_stow),
            Step(
                "seat + release L",
                qL_stow,
                "open",
                0.8,
                aux_ctrl=aux_stow,
                # Pin server at the rack's interior on the lower shelf
                # (centered, just resting on the shelf top). Explicit
                # pose so arm-tracking offsets at this moment don't
                # propagate into the server's final pose.
                attach_activate_at=(
                    (
                        AttachmentWeldName.SERVER_ON_LOWER_SHELF,
                        (
                            float(LAYOUT.server_center_x_in_rack),
                            0.0,
                            float(LAYOUT.rack.lower_shelf_z_world) + float(LAYOUT.server.half[2]),
                        ),
                        (1.0, 0.0, 0.0, 0.0),
                    ),
                ),
                attach_deactivate=(grasp_L,),
            ),
        ]
    )
    scripts[ArmSide.RIGHT].extend(
        [
            Step("to server handle R", qR_app, "open", 1.8, aux_ctrl=aux_server),
            Step("at handle R", qR_at, "open", 0.8, aux_ctrl=aux_server),
            # Right gripper stays OPEN through the whole carry: the left
            # arm owns the grasp weld, so any force the right fingers
            # would apply to the server body just fights that weld. A
            # closed-gripper right hand contacting a welded body produced
            # QACC blowups around t=44s during the install phase.
            Step("hold handle R", qR_at, "open", 0.6, aux_ctrl=aux_server),
            Step("pull server R", qR_out, "open", 2.5, aux_ctrl=aux_server),
            Step("lower to shelf R", qR_stow, "open", 2.5, aux_ctrl=aux_stow),
            Step("release R", qR_stow, "open", 0.8, aux_ctrl=aux_stow),
        ]
    )

    # 3) New server pickup from new_bin + install in rack
    snap_l, snap_r = seed_at_lift(LAYOUT.lift.pick_new)
    aux_pick = {DataCenterAux.LIFT: LAYOUT.lift.pick_new}

    new_handle_z = _torso_world_z(LAYOUT.lift.pick_new) + LAYOUT.bins.new_local_z + 0.06
    new_handle_L = np.array([0.14, -0.15, new_handle_z])
    new_handle_R = np.array([0.14, +0.15, new_handle_z])
    new_approach_L = new_handle_L + np.array([-0.06, 0.0, 0.0])
    new_approach_R = new_handle_R + np.array([-0.06, 0.0, 0.0])

    qL_napp, _ = snap_l(new_approach_L)
    qL_ngrip, _ = snap_l(new_handle_L)
    qR_napp, _ = snap_r(new_approach_R)
    qR_ngrip, _ = snap_r(new_handle_R)

    snap_l, snap_r = seed_at_lift(LAYOUT.lift.server)
    # Rack-slot target aligns with the handle bar of the new-server (same
    # geometry as the old server), so arms end the carry with the server
    # seated in the slot rather than 2 cm in front of it.
    slot_L = np.array([handle_x, -0.15, LAYOUT.server.slot_z])
    slot_R = np.array([handle_x, +0.15, LAYOUT.server.slot_z])
    pre_slot_L = slot_L + np.array([-0.12, 0.0, 0.0])
    pre_slot_R = slot_R + np.array([-0.12, 0.0, 0.0])

    qL_pre, _ = snap_l(pre_slot_L)
    qL_in, _ = snap_l(slot_L)
    qR_pre, _ = snap_r(pre_slot_R)
    qR_in, _ = snap_r(slot_R)

    # Same single-arm policy as server extraction: only the LEFT arm welds
    # to the new_server; the right arm shadow-carries for visual effect.
    new_server_id = grippable_id("new_server")
    grasp_new_L = grasp_weld(ArmSide.LEFT, new_server_id)

    scripts[ArmSide.LEFT].extend(
        [
            Step("to new_bin L", qL_napp, "open", 1.8, aux_ctrl=aux_pick),
            Step("at new_bin L", qL_ngrip, "open", 0.8, aux_ctrl=aux_pick),
            Step(
                "grip new + release bin L",
                qL_ngrip,
                "closed",
                0.6,
                aux_ctrl=aux_pick,
                attach_activate=(grasp_new_L,),
                attach_deactivate=(AttachmentWeldName.NEW_IN_BIN,),
            ),
            Step("raise new L", qL_pre, "closed", 2.0, aux_ctrl=aux_server),
            Step("insert L", qL_in, "closed", 1.8, aux_ctrl=aux_server),
            Step(
                "seat in rack + release L",
                qL_in,
                "open",
                0.8,
                aux_ctrl=aux_server,
                # Pin new_server in the rack's upper slot at exactly the
                # canonical world pose, not wherever the arm happens to
                # be at this instant. The grasp_offset captured at
                # pickup was relative to where the arm was in the bin;
                # without an explicit final pose that offset propagates
                # straight into the install pose and the new server
                # ends up several cm off-slot.
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
    scripts[ArmSide.RIGHT].extend(
        [
            Step("to new_bin R", qR_napp, "open", 1.8, aux_ctrl=aux_pick),
            Step("at new_bin R", qR_ngrip, "open", 0.8, aux_ctrl=aux_pick),
            # Right gripper stays OPEN through the new-server carry for
            # the same reason as server extraction — see the "hold
            # handle R" comment above. Closed-right-gripper vs. welded
            # new_server was the direct cause of the t=44s QACC blowup
            # during `insert R`.
            Step("hold new R", qR_ngrip, "open", 0.6, aux_ctrl=aux_pick),
            Step("raise new R", qR_pre, "open", 2.0, aux_ctrl=aux_server),
            Step("insert R", qR_in, "open", 1.8, aux_ctrl=aux_server),
            Step("release R", qR_in, "open", 0.8, aux_ctrl=aux_server),
        ]
    )

    # 4) Replug cables sequentially into NEW server (active side rotates).
    def plan_replug(port_idx: int, active_side: ArmSide) -> None:
        cable_id = grippable_id(f"cable{port_idx + 1}_connector")
        cable_grasp_weld = grasp_weld(active_side, cable_id)
        port_weld_new = _PORT_NEW[port_idx]

        snap_l, snap_r = seed_at_lift(LAYOUT.lift.cables)
        snap = snap_l if active_side == ArmSide.LEFT else snap_r

        side_y = -0.28 if active_side == ArmSide.LEFT else +0.28
        dangle = np.array([LAYOUT.server_front_x - 0.28, side_y, LAYOUT.server.slot_z - 0.10])
        port_pos = _port_world_pos(port_idx)
        approach = port_pos + np.array([-0.08, 0.0, 0.0])
        seated = port_pos + np.array([-0.005, 0.0, 0.0])

        q_d, _ = snap(dangle)
        q_a, _ = snap(approach)
        q_s, _ = snap(seated)

        aux = {DataCenterAux.LIFT: LAYOUT.lift.cables}
        active_steps = [
            Step(f"to c{port_idx + 1} dangle", q_d, "open", 1.2, aux_ctrl=aux),
            Step(
                f"grip c{port_idx + 1}",
                q_d,
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

    # Replug phase — re-enabled now that puppet mode eliminates the
    # earlier QACC blowup at ~t=44 s. Under direct-qpos arm motion +
    # gravity=0, the cable composite stays where it's left between
    # phases (no gravity drag), and the arm doesn't overshoot when
    # picking the connector back up (no PD lag). The original `attach_activate`
    # of grasp_weld captures whatever offset separates arm and connector,
    # but with gravity=0 the offset is small (whatever the unplug pose
    # left it at).
    plan_replug(0, ArmSide.LEFT)
    plan_replug(1, ArmSide.RIGHT)
    plan_replug(2, ArmSide.LEFT)

    # 5) Return home
    push_both("return home", 1.6, aux={DataCenterAux.LIFT: LAYOUT.lift.home})

    for side in ARM_PREFIXES:
        print(f"  [{side}] {len(scripts[side])} steps planned")

    apply_initial_state(model, data, arms, cube_body_ids)
    return scripts
