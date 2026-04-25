"""PAL TIAGo loader — Menagerie-adapter for the data-center scene's mobile base.

Menagerie's `pal_tiago/tiago.xml` stays untouched on disk; everything our
scene needs to change about it is declared here as a typed
`TiagoConfig` + applied by `load_tiago(config)`. Scenes import only
`load_tiago` and `torso_world_pos_at_zero` from this module — not
`TIAGO_XML` directly — so there's a single place to look when Menagerie
updates and something needs to adjust.

Customizations we currently apply:

* **Strip the upstream single arm** (`arm_1_link` subtree) — the data-
  center scene replaces it with two Pipers attached via `robots/piper.py`.
* **Strip the upstream head** (`head_1_link` subtree) — unactuated in
  our scenes; the camera it carried is replaced by a rigid pole + D435i
  welded to base_link.
* **Delete the `reference` freejoint on base_link** — mink's IK is
  purely kinematic and ignores equality constraints, so leaving a
  floating base pinned only by an `mjEQ_WELD` lets the solver slide the
  robot under a world-frame target during planning. That produced the
  "grabs at mid-air" bug (IK reports near-zero error while runtime TCP
  is 40 cm off). Removing the joint keeps planning and runtime kinematics
  in agreement.

Boundary assertions (see `_assert_menagerie_shape`) check the upstream
names we depend on exist BEFORE we try to delete or reference them. If
Menagerie renames `arm_1_link` → `left_arm_link` the load fails here
with a clear message instead of silently no-op'ing and failing 400
lines later in IK-land.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import mujoco

from paths import TIAGO_XML


@dataclass(frozen=True)
class TiagoConfig:
    """Declarative customization for Menagerie's `pal_tiago/tiago.xml`.

    Only the deltas our scenes need are listed here — everything else
    (meshes, inertia, joint damping, actuator-less torso_lift_joint)
    stays as upstream. Keeping the diff this small is deliberate: every
    override is a drift risk on the next Menagerie bump.
    """

    strip_subtrees: tuple[str, ...] = ("arm_1_link", "head_1_link")
    remove_freejoint_name: str | None = "reference"


# Names we dereference at load time; see `_assert_menagerie_shape`.
_TIAGO_REQUIRED_BODIES: tuple[str, ...] = (
    "base_link",
    "torso_lift_link",
    "arm_1_link",
    "head_1_link",
)


def _iter_body_subtree(root: mujoco.MjsBody) -> Iterator[mujoco.MjsBody]:
    """Pre-order traversal yielding `root` and every descendant body."""
    yield root
    for child in root.bodies:
        yield from _iter_body_subtree(child)


def _collect_dead_body_names(spec: mujoco.MjSpec, subtree_roots: tuple[str, ...]) -> set[str]:
    """Names of every body inside each named subtree, collected BEFORE
    `spec.delete(...)` runs so downstream orphan-reference pruning
    (contact excludes) knows which names just disappeared."""
    dead: set[str] = set()
    for root_name in subtree_roots:
        root = spec.body(root_name)
        for b in _iter_body_subtree(root):
            if b.name:
                dead.add(b.name)
    return dead


def _assert_menagerie_shape(spec: mujoco.MjSpec, config: TiagoConfig) -> None:
    """Fail at load time if the upstream names we plan to touch are
    missing. Cheaper to triage here than as a mysterious MuJoCo compile
    error 400 lines into `build_spec`.
    """
    # Everything the scene downstream assumes exists:
    for name in _TIAGO_REQUIRED_BODIES:
        if spec.body(name) is None:
            raise RuntimeError(
                f"TIAGo upstream XML missing expected body {name!r}. "
                "Menagerie's pal_tiago/tiago.xml may have changed shape — "
                "update robots/tiago.py if the new naming is intentional."
            )
    # Each strip target must actually be there:
    for subtree in config.strip_subtrees:
        if spec.body(subtree) is None:
            raise RuntimeError(
                f"TiagoConfig.strip_subtrees references {subtree!r} but it's "
                "not in TIAGo's upstream XML."
            )
    # Freejoint to remove must exist where we expect it:
    if config.remove_freejoint_name is not None:
        base = spec.body("base_link")
        names = {j.name for j in base.joints if j.name}
        if config.remove_freejoint_name not in names:
            raise RuntimeError(
                f"TiagoConfig.remove_freejoint_name={config.remove_freejoint_name!r} "
                f"but base_link's joints are {sorted(names)!r}."
            )


def load_tiago(config: TiagoConfig = TiagoConfig()) -> mujoco.MjSpec:  # noqa: B008 — frozen dataclass, no shared state
    """Return a customized, uncompiled TIAGo MjSpec per `config`.

    Callers typically treat this as the root spec and add more bodies to
    its worldbody before `spec.compile()` — that's what
    `scenes/data_center.py::build_spec` does.
    """
    spec = mujoco.MjSpec.from_file(str(TIAGO_XML))
    _assert_menagerie_shape(spec, config)

    dead_bodies = _collect_dead_body_names(spec, config.strip_subtrees)
    for root_name in config.strip_subtrees:
        spec.delete(spec.body(root_name))  # cascades to descendants

    if config.remove_freejoint_name is not None:
        base = spec.body("base_link")
        for j in list(base.joints):
            if j.name == config.remove_freejoint_name:
                spec.delete(j)
                break

    # Prune orphaned `<exclude>` entries (TIAGo's upstream declares 3, all
    # referencing arm/gripper bodies that are now gone). TIAGo has no
    # `<actuator>` or `<equality>` block upstream, so nothing else to prune.
    for exc in list(spec.excludes):
        if exc.bodyname1 in dead_bodies or exc.bodyname2 in dead_bodies:
            spec.delete(exc)

    return spec


def torso_world_pos_at_zero() -> tuple[float, float, float]:
    """World pos of `torso_lift_link` at qpos=0, read directly from
    Menagerie's tiago.xml.

    Referenced by `scenes/data_center_layout` so bin heights / IK targets
    derive from the one authoritative source — no hardcoded `(0, 0, 0.8885)`
    copy to drift silently if Menagerie updates the torso attach point.

    In TIAGo's upstream file `torso_lift_link` hangs off `torso_fixed_link`
    whose own pos is (0, 0, 0), so the local pos is the world pos at qpos=0.
    """
    spec = mujoco.MjSpec.from_file(str(TIAGO_XML))
    body = spec.body("torso_lift_link")
    if body is None:
        raise RuntimeError(
            "TIAGo upstream XML missing torso_lift_link — "
            "did Menagerie's pal_tiago/tiago.xml change shape?"
        )
    pos = body.pos
    return (float(pos[0]), float(pos[1]), float(pos[2]))
