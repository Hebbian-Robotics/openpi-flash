"""PAL TIAGo loader — Menagerie-adapter for the data-center scene's mobile base.

Menagerie's `pal_tiago/tiago.xml` stays untouched on disk; everything our
scene needs to change about it is declared here as a typed `TiagoConfig`
+ applied by `load_tiago(config)`. Scenes import only `load_tiago` and
`torso_world_pos_at_zero` from this module — not `TIAGO_XML` directly —
so there's a single place to look when Menagerie updates and something
needs to adjust.

Built atop `dm_control.mjcf`: `load_tiago` returns an `mjcf.RootElement`
which the scene then composes (Pipers, cameras, rack, cart, cables) by
attaching at sites. Compilation happens once, at the scene's
`build_scene()`, via `MjModel.from_xml_string(root.to_xml_string(),
root.get_assets())`.

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

from dm_control import mjcf

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


def _iter_body_subtree(root: mjcf.Element) -> Iterator[mjcf.Element]:
    """Pre-order traversal yielding `root` and every descendant body."""
    yield root
    for child in root.all_children():
        if child.tag == "body":
            yield from _iter_body_subtree(child)


def _collect_dead_body_names(root: mjcf.RootElement, subtree_roots: tuple[str, ...]) -> set[str]:
    """Names of every body inside each named subtree, collected BEFORE
    `body.remove()` runs so downstream orphan-reference pruning (contact
    excludes) knows which names just disappeared."""
    dead: set[str] = set()
    for root_name in subtree_roots:
        body = root.find("body", root_name)
        if body is None:
            continue
        for b in _iter_body_subtree(body):
            if b.name:
                dead.add(b.name)
    return dead


def _assert_menagerie_shape(root: mjcf.RootElement, config: TiagoConfig) -> None:
    """Fail at load time if the upstream names we plan to touch are
    missing. Cheaper to triage here than as a mysterious MuJoCo compile
    error 400 lines into `build_scene`."""
    for name in _TIAGO_REQUIRED_BODIES:
        if root.find("body", name) is None:
            raise RuntimeError(
                f"TIAGo upstream XML missing expected body {name!r}. "
                "Menagerie's pal_tiago/tiago.xml may have changed shape — "
                "update robots/tiago.py if the new naming is intentional."
            )
    for subtree in config.strip_subtrees:
        if root.find("body", subtree) is None:
            raise RuntimeError(
                f"TiagoConfig.strip_subtrees references {subtree!r} but it's "
                "not in TIAGo's upstream XML."
            )
    if (
        config.remove_freejoint_name is not None
        and root.find("joint", config.remove_freejoint_name) is None
    ):
        raise RuntimeError(
            f"TiagoConfig.remove_freejoint_name={config.remove_freejoint_name!r} "
            "but no such joint in TIAGo's upstream XML."
        )


def load_tiago(config: TiagoConfig = TiagoConfig()) -> mjcf.RootElement:  # noqa: B008 — frozen dataclass, no shared state
    """Return a customized, uncompiled TIAGo `mjcf.RootElement` per `config`.

    Callers typically treat this as the root and compose more children
    (Pipers, cameras, rack, cart, cables) before compiling — that's
    what `scenes/data_center.py::build_scene` does.

    The returned root has `.model = ""` (no namespace) so TIAGo's body
    names compile unprefixed (`base_link`, `torso_lift_link`); attached
    children carry their own namespace prefixes via their `.model`
    attribute (e.g. `left/`, `cable1/`).
    """
    root = mjcf.from_path(str(TIAGO_XML))
    # TIAGo's upstream MJCF declares `<mujoco model="tiago">`; clearing the
    # model attribute keeps body names like `torso_lift_link` unprefixed in
    # the compiled scene (rather than `tiago/torso_lift_link`).
    root.model = ""
    _assert_menagerie_shape(root, config)

    # Collect dead-body names BEFORE removing subtrees so orphan-exclude
    # pruning below can match against the deleted set.
    dead_bodies = _collect_dead_body_names(root, config.strip_subtrees)

    # Prune `<exclude>` entries referencing dead bodies first — body Element
    # references would dangle if we removed the bodies first.
    if root.contact is not None:
        for exc in list(root.contact.all_children()):
            if exc.tag != "exclude":
                continue
            b1 = getattr(exc.body1, "name", None) if exc.body1 is not None else None
            b2 = getattr(exc.body2, "name", None) if exc.body2 is not None else None
            if b1 in dead_bodies or b2 in dead_bodies:
                exc.remove()

    for subtree in config.strip_subtrees:
        body = root.find("body", subtree)
        if body is not None:
            body.remove()

    if config.remove_freejoint_name is not None:
        joint = root.find("joint", config.remove_freejoint_name)
        if joint is not None:
            joint.remove()

    return root


def torso_world_pos_at_zero() -> tuple[float, float, float]:
    """World pos of `torso_lift_link` at qpos=0, read directly from
    Menagerie's tiago.xml.

    Referenced by `scenes/data_center_layout` so bin heights / IK targets
    derive from the one authoritative source — no hardcoded `(0, 0, 0.8885)`
    copy to drift silently if Menagerie updates the torso attach point.

    In TIAGo's upstream file `torso_lift_link` hangs off `torso_fixed_link`
    whose own pos is (0, 0, 0), so the local pos is the world pos at qpos=0.
    """
    root = mjcf.from_path(str(TIAGO_XML))
    body = root.find("body", "torso_lift_link")
    if body is None:
        raise RuntimeError(
            "TIAGo upstream XML missing torso_lift_link — "
            "did Menagerie's pal_tiago/tiago.xml change shape?"
        )
    pos = body.pos
    return (float(pos[0]), float(pos[1]), float(pos[2]))
