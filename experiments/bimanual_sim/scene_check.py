"""Compile-time sanity checks and schematic printer for MJCF scenes.

`check_scene` runs a pass of cheap invariants right after the scene is
compiled and `apply_initial_state` has been called, before any physics or
viser startup. Every geometry bug surfaces as a structured
`SceneCheckViolation` with a human-readable detail string; violations are
aggregated and raised together via `SceneCheckError` so the dev sees the
full list, not just the first failure.

`print_schematic` writes a human-readable tree of bodies, world positions,
axis-aligned bounding boxes, and flagged overlaps — the `--inspect` CLI
mode hits this to eyeball scene layout without running physics.

The module is intentionally scene-agnostic: inputs are a compiled
`mj_forward`-ready `MjModel` / `MjData` pair plus a few scene-declared
descriptors (grippable names, allow-listed overlapping geom pairs, the
attachment-constraint registry). Scene modules (`scenes/data_center.py`)
expose these via attributes the runner reads with `getattr`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from arm_handles import ArmHandles, ArmSide

ViolationKind = Literal[
    "static_overlap",
    "tcp_clip",
    "unreachable",
    "missing_body",
    "eq_type_mismatch",
    "connect_anchor_oob",
    "cable_short",
]


@dataclass(frozen=True)
class SceneCheckViolation:
    kind: ViolationKind
    detail: str


class SceneCheckError(RuntimeError):
    """Raised when `check_scene` finds one or more violations.

    Carries the structured `violations` list so callers can inspect the
    individual failures; `str(err)` renders the whole sorted list.
    """

    def __init__(self, violations: list[SceneCheckViolation]) -> None:
        self.violations = violations
        lines = [f"scene_check found {len(violations)} violation(s):"]
        for v in violations:
            lines.append(f"  [{v.kind}] {v.detail}")
        super().__init__("\n".join(lines))


# -----------------------------------------------------------------------------
# Attachment-constraint descriptor (scene-agnostic)
# -----------------------------------------------------------------------------
# Scenes with mjEQ_WELD / mjEQ_CONNECT equalities can expose a tuple of these
# so `check_scene` can validate each entry (body refs resolve, compiled eq
# type matches, CONNECT anchors lie inside body_a). The scene's own
# `_AttachmentWeldSpec` maps trivially to this shape; the runner adapts.


@dataclass(frozen=True)
class AttachmentConstraint:
    name: str
    body_a: str
    body_b: str
    kind: Literal["weld", "connect"]
    connect_anchor_in_a: tuple[float, float, float] = (0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# AABB helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _GeomBox:
    """Tight-as-cheap world-space AABB for one geom."""

    geom_id: int
    name: str
    body_id: int
    body_name: str
    min_xyz: np.ndarray  # shape (3,)
    max_xyz: np.ndarray  # shape (3,)


def _geom_world_aabb(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> _GeomBox:
    """Compute a conservative world-space AABB for one geom.

    Primitives (box/sphere/cylinder/capsule/ellipsoid) use `model.geom_size`
    as the half-extent along each local axis; we project those to world via
    `data.geom_xmat`, so the box is axis-aligned in world even after the
    geom has been rotated. Meshes fall back to `geom_rbound` (a conservative
    sphere bound) — loose but correct, and our scenes have at most one mesh.
    """
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        # rbound is the radius of the enclosing sphere — conservative AABB.
        r = float(model.geom_rbound[geom_id])
        half_world = np.array([r, r, r])
    else:
        # For primitives, half-extents along local axes; for capsule/cylinder
        # the full half-length is size[1] along local z (extended from base
        # radius size[0]). size already carries the right per-axis half for
        # box/ellipsoid; for sphere/cylinder/capsule we use max(size) as a
        # cheap conservative bound in any orientation.
        if gtype == mujoco.mjtGeom.mjGEOM_BOX or gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            local_half = size.copy()
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            local_half = np.array([r, r, r])
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER or gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(size[0])
            half_len = float(size[1])
            local_half = np.array([r, r, half_len + r])
        else:  # PLANE or other — treat as a point (harmless in overlap checks)
            local_half = np.zeros(3)
        # Project the local half-extents onto world axes via |xmat| · local_half.
        # This gives the axis-aligned world bound of the (rotated) local box.
        half_world = np.abs(xmat) @ local_half

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"g{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"b{body_id}"
    return _GeomBox(
        geom_id=geom_id,
        name=name,
        body_id=body_id,
        body_name=body_name,
        min_xyz=xpos - half_world,
        max_xyz=xpos + half_world,
    )


def _aabb_overlap(a: _GeomBox, b: _GeomBox, eps: float = 1e-6) -> bool:
    """True iff a's and b's axis-aligned boxes intersect (strict, with eps)."""
    return bool(np.all(a.min_xyz < b.max_xyz - eps) and np.all(b.min_xyz < a.max_xyz - eps))


def _point_in_aabb(p: np.ndarray, box: _GeomBox, eps: float = 1e-6) -> bool:
    return bool(np.all(p > box.min_xyz + eps) and np.all(p < box.max_xyz - eps))


def _body_is_static(model: mujoco.MjModel, body_id: int) -> bool:
    """A body is static iff its weld-equivalence root is the worldbody."""
    return int(model.body_weldid[body_id]) == 0


def _collect_static_geom_boxes(model: mujoco.MjModel, data: mujoco.MjData) -> list[_GeomBox]:
    boxes: list[_GeomBox] = []
    for g in range(model.ngeom):
        body_id = int(model.geom_bodyid[g])
        if not _body_is_static(model, body_id):
            continue
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue  # planes have infinite extent; skip
        boxes.append(_geom_world_aabb(model, data, g))
    return boxes


def _body_local_aabb(model: mujoco.MjModel, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Union of all child-geom AABBs in `body_id`'s local frame.

    Used by the CONNECT anchor check — the anchor is given in body_a's local
    frame and should fall inside the physical extent of body_a's geoms. An
    anchor outside this local AABB pins body_b's origin to empty space.
    """
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        gtype = int(model.geom_type[g])
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = np.asarray(model.geom_pos[g], dtype=float)
        size = np.asarray(model.geom_size[g], dtype=float)
        if gtype == mujoco.mjtGeom.mjGEOM_BOX or gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            half = size.copy()
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            half = np.array([r, r, r])
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER or gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(size[0])
            hl = float(size[1])
            half = np.array([r, r, hl + r])
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            r = float(model.geom_rbound[g])
            half = np.array([r, r, r])
        else:
            half = np.zeros(3)
        mins = np.minimum(mins, pos - half)
        maxs = np.maximum(maxs, pos + half)
    if not np.all(np.isfinite(mins)):
        # Body carries no geoms (TIAGo links often have none); return a tiny
        # box at origin so the anchor check becomes trivially "anchor at origin
        # is fine, anywhere else is out of bounds" — intentional: connect
        # anchors should live on a body with actual geometry.
        return np.zeros(3), np.zeros(3)
    return mins, maxs


# -----------------------------------------------------------------------------
# Reach-envelope pre-filter
# -----------------------------------------------------------------------------

_PIPER_REACH_RADIUS_M = 0.75
"""Loose reach pre-filter. A Piper's strict reach envelope is ~0.55-0.65 m to
the gripper centre, but grippable bodies are measured to their body *centre*
while the arm actually reaches for a handle offset from that centre — so a
grippable with centre at 0.67 m might still be reachable via a handle at
0.60 m. The real unreachable-target guard is `_snap_factory`'s 2 cm IK-
residual abort; this radius just catches obviously-misplaced bodies (rack
sitting 2 m away) before we bother with IK at all."""


def _arm_base_world_pos(
    model: mujoco.MjModel, data: mujoco.MjData, side: ArmSide
) -> np.ndarray | None:
    """World position of `{side}base_link`, or None if the body isn't present."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}base_link")
    if bid < 0:
        return None
    return np.asarray(data.xpos[bid], dtype=float)


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------


@dataclass
class _CheckContext:
    """Things we compute once and reuse across multiple checks."""

    static_boxes: list[_GeomBox] = field(default_factory=list)


def check_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    arms: dict[ArmSide, ArmHandles],
    grippable_names: tuple[str, ...] = (),
    allowed_static_overlaps: tuple[tuple[str, str], ...] = (),
    attachment_constraints: tuple[AttachmentConstraint, ...] = (),
) -> None:
    """Run every invariant; raise `SceneCheckError` if any failed.

    Intended to be called once from the runner after `apply_initial_state`
    and before `make_task_plan` / viser. All inputs are already resolved by
    the runner — no scene-specific logic lives in this module.
    """
    violations: list[SceneCheckViolation] = []
    ctx = _CheckContext(static_boxes=_collect_static_geom_boxes(model, data))

    violations.extend(_check_static_overlaps(ctx, allowed_static_overlaps))
    violations.extend(_check_tcp_not_in_static_geom(data, arms, ctx))
    violations.extend(_check_grippables_reachable(model, data, grippable_names, arms))
    violations.extend(_check_attachment_constraints(model, data, attachment_constraints))

    if violations:
        raise SceneCheckError(violations)


def _check_static_overlaps(
    ctx: _CheckContext,
    allowed: tuple[tuple[str, str], ...],
) -> list[SceneCheckViolation]:
    # Canonicalise allow-list pairs so (a, b) and (b, a) both match.
    allow_set: set[tuple[str, str]] = set()
    for a, b in allowed:
        allow_set.add((a, b))
        allow_set.add((b, a))

    out: list[SceneCheckViolation] = []
    boxes = ctx.static_boxes
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            # Skip pairs within the same body — the body's own geoms are
            # allowed to share edges by design (rack panels meet at seams).
            if a.body_id == b.body_id:
                continue
            if not _aabb_overlap(a, b):
                continue
            if (a.name, b.name) in allow_set or (a.body_name, b.body_name) in allow_set:
                continue
            out.append(
                SceneCheckViolation(
                    kind="static_overlap",
                    detail=(
                        f"{a.name!r} (body {a.body_name!r}) overlaps "
                        f"{b.name!r} (body {b.body_name!r}); "
                        f"add ({a.name!r}, {b.name!r}) or "
                        f"({a.body_name!r}, {b.body_name!r}) to "
                        f"ALLOWED_STATIC_OVERLAPS if intentional"
                    ),
                )
            )
    return out


def _check_tcp_not_in_static_geom(
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    ctx: _CheckContext,
) -> list[SceneCheckViolation]:
    out: list[SceneCheckViolation] = []
    for side, arm in arms.items():
        tcp = np.asarray(data.site_xpos[arm.tcp_site_id], dtype=float)
        for box in ctx.static_boxes:
            if _point_in_aabb(tcp, box):
                out.append(
                    SceneCheckViolation(
                        kind="tcp_clip",
                        detail=(
                            f"{side}tcp at ({tcp[0]:.3f}, {tcp[1]:.3f}, "
                            f"{tcp[2]:.3f}) is inside static geom {box.name!r} "
                            f"(body {box.body_name!r})"
                        ),
                    )
                )
    return out


def _check_grippables_reachable(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grippable_names: tuple[str, ...],
    arms: dict[ArmSide, ArmHandles],
) -> list[SceneCheckViolation]:
    out: list[SceneCheckViolation] = []
    arm_bases: list[tuple[ArmSide, np.ndarray]] = []
    for side in arms:
        bp = _arm_base_world_pos(model, data, side)
        if bp is not None:
            arm_bases.append((side, bp))
    if not arm_bases:
        return out  # no arms declared — nothing to reach-check

    for name in grippable_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            out.append(
                SceneCheckViolation(
                    kind="missing_body",
                    detail=f"grippable {name!r} not found in compiled model",
                )
            )
            continue
        gpos = np.asarray(data.xpos[bid], dtype=float)
        dists = [(side, float(np.linalg.norm(gpos - bp))) for side, bp in arm_bases]
        if all(d > _PIPER_REACH_RADIUS_M for _, d in dists):
            reach_str = ", ".join(f"{s} {d:.2f} m" for s, d in dists)
            out.append(
                SceneCheckViolation(
                    kind="unreachable",
                    detail=(
                        f"grippable {name!r} at ({gpos[0]:.3f}, {gpos[1]:.3f}, "
                        f"{gpos[2]:.3f}) is farther than {_PIPER_REACH_RADIUS_M} m "
                        f"from every arm base ({reach_str})"
                    ),
                )
            )
    return out


def _check_attachment_constraints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constraints: tuple[AttachmentConstraint, ...],
) -> list[SceneCheckViolation]:
    out: list[SceneCheckViolation] = []
    for c in constraints:
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, c.name)
        if eq_id < 0:
            out.append(
                SceneCheckViolation(
                    kind="missing_body",
                    detail=f"attachment equality {c.name!r} not found in compiled model",
                )
            )
            continue
        body_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c.body_a)
        body_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c.body_b)
        if body_a_id < 0:
            out.append(
                SceneCheckViolation(
                    kind="missing_body",
                    detail=f"attachment {c.name!r} body_a={c.body_a!r} not found",
                )
            )
        if body_b_id < 0:
            out.append(
                SceneCheckViolation(
                    kind="missing_body",
                    detail=f"attachment {c.name!r} body_b={c.body_b!r} not found",
                )
            )
        # Verify compiled equality type matches the declared kind.
        compiled_type = int(model.eq_type[eq_id])
        expected = mujoco.mjtEq.mjEQ_WELD if c.kind == "weld" else mujoco.mjtEq.mjEQ_CONNECT
        if compiled_type != int(expected):
            out.append(
                SceneCheckViolation(
                    kind="eq_type_mismatch",
                    detail=(
                        f"attachment {c.name!r} declared kind={c.kind!r} "
                        f"but compiled eq type={compiled_type} "
                        f"(expected {int(expected)})"
                    ),
                )
            )
            continue
        # For CONNECT entries, the anchor must lie inside body_a's local AABB.
        if c.kind == "connect" and body_a_id >= 0:
            mins, maxs = _body_local_aabb(model, body_a_id)
            ax = np.asarray(c.connect_anchor_in_a, dtype=float)
            # Use <= / >= rather than strict to accept boundary anchors (ports
            # sit exactly on the server's front face by design).
            if not bool(np.all(ax >= mins - 1e-6) and np.all(ax <= maxs + 1e-6)):
                out.append(
                    SceneCheckViolation(
                        kind="connect_anchor_oob",
                        detail=(
                            f"attachment {c.name!r} anchor ({ax[0]:.3f}, "
                            f"{ax[1]:.3f}, {ax[2]:.3f}) in body_a={c.body_a!r} "
                            f"local frame is outside body AABB "
                            f"[{mins[0]:.3f},{maxs[0]:.3f}] × "
                            f"[{mins[1]:.3f},{maxs[1]:.3f}] × "
                            f"[{mins[2]:.3f},{maxs[2]:.3f}]"
                        ),
                    )
                )
    return out


# -----------------------------------------------------------------------------
# Schematic printer (--inspect)
# -----------------------------------------------------------------------------


def print_schematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    arms: dict[ArmSide, ArmHandles],
    grippable_names: tuple[str, ...] = (),
    attachment_constraints: tuple[AttachmentConstraint, ...] = (),
) -> None:
    """Print a body/geom/attachment schematic to stdout. Never raises; meant
    to be paired with `check_scene` (which does) on the `--inspect` path."""
    print(
        f"Scene: nbody={model.nbody} njnt={model.njnt} nu={model.nu} "
        f"neq={model.neq} ngeom={model.ngeom}"
    )
    static_boxes = _collect_static_geom_boxes(model, data)
    print()
    print(f"Static bodies (weldid=0): {len(static_boxes)} geoms")
    # Group by body for readability.
    by_body: dict[int, list[_GeomBox]] = {}
    for b in static_boxes:
        by_body.setdefault(b.body_id, []).append(b)
    for body_id in sorted(by_body):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"b{body_id}"
        body_pos = np.asarray(data.xpos[body_id], dtype=float)
        geoms = by_body[body_id]
        body_min = np.min([g.min_xyz for g in geoms], axis=0)
        body_max = np.max([g.max_xyz for g in geoms], axis=0)
        print(
            f"  {name:<22} pos=({body_pos[0]:+.3f}, {body_pos[1]:+.3f}, {body_pos[2]:+.3f})"
            f"  AABB=[{body_min[0]:+.3f},{body_max[0]:+.3f}]"
            f"×[{body_min[1]:+.3f},{body_max[1]:+.3f}]"
            f"×[{body_min[2]:+.3f},{body_max[2]:+.3f}]"
            f"  geoms={len(geoms)}"
        )

    print()
    print("Arms at home pose:")
    for side, arm in arms.items():
        base = _arm_base_world_pos(model, data, side)
        tcp = np.asarray(data.site_xpos[arm.tcp_site_id], dtype=float)
        if base is not None:
            print(
                f"  [{side}] base=({base[0]:+.3f}, {base[1]:+.3f}, {base[2]:+.3f})"
                f"  tcp=({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f})"
            )

    if grippable_names:
        print()
        print(f"Grippables ({len(grippable_names)}):")
        for name in grippable_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                print(f"  {name:<22} ⚠ MISSING")
                continue
            gpos = np.asarray(data.xpos[bid], dtype=float)
            dists = []
            for side in arms:
                bp = _arm_base_world_pos(model, data, side)
                if bp is not None:
                    dists.append(f"{side}{np.linalg.norm(gpos - bp):.2f}m")
            print(
                f"  {name:<22} pos=({gpos[0]:+.3f}, {gpos[1]:+.3f}, {gpos[2]:+.3f})"
                f"  reach=[{', '.join(dists)}]"
            )

    if attachment_constraints:
        print()
        weld_count = sum(1 for c in attachment_constraints if c.kind == "weld")
        connect_count = sum(1 for c in attachment_constraints if c.kind == "connect")
        print(
            f"Attachment constraints ({len(attachment_constraints)}): "
            f"{connect_count}× CONNECT, {weld_count}× WELD"
        )
        for c in attachment_constraints:
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, c.name)
            status = "✓" if eq_id >= 0 else "✗ MISSING"
            print(f"  {c.name:<28} {c.kind.upper():<8} {c.body_a!r} ↔ {c.body_b!r}  {status}")
