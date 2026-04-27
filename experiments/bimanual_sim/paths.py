"""Filesystem paths used by scenes.

Resolved lazily so running the code on a new machine (different user, laptop
vs. EC2, etc.) doesn't require editing Python. Convention: clone
`mujoco_menagerie` into `$HOME/mujoco_menagerie`; override with the
`MENAGERIE_PATH` env var if you put it elsewhere.

Each menagerie-relative path validates the file exists at *first access*, not
at module import. This matters because `paths` is imported transitively by
every scene module: eager validation would force the Piper / TIAGo legacy
assets to be present even when the live UR10e + Mobile-ALOHA scene doesn't
touch them. Lazy access localises the failure to "this scene actually uses
this asset and the asset is missing", which is the only useful failure mode.

Downstream code consumes the refined `MenagerieXml` / `MenagerieMesh` types,
which carry the proof that the file exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, NewType

# A menagerie-relative XML path whose existence has been verified at
# construction time. Produced only by `parse_menagerie_xml`.
MenagerieXml = NewType("MenagerieXml", Path)
# Same refinement pattern, for mesh assets (.stl / .obj). Kept as a distinct
# NewType from MenagerieXml so we can't accidentally hand an STL to code that
# expects a compilable MJCF.
MenagerieMesh = NewType("MenagerieMesh", Path)


def _resolve_menagerie() -> Path:
    env = os.environ.get("MENAGERIE_PATH")
    if env:
        return Path(env).expanduser()
    # Default: ~/mujoco_menagerie — same path works for the ubuntu user on the
    # EC2 instance and for a developer's laptop.
    return Path.home() / "mujoco_menagerie"


def parse_menagerie_xml(*parts: str) -> MenagerieXml:
    """Resolve a menagerie-relative MJCF path; raise if it doesn't exist.

    The raise-at-access gives one clear failure site ("menagerie not cloned"
    or "that XML path is wrong") instead of the opaque MuJoCo parse error
    you'd get when `MjSpec.from_file` is eventually called on a missing file.
    """
    path = _resolve_menagerie().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"menagerie MJCF not found: {path}. "
            "Clone https://github.com/google-deepmind/mujoco_menagerie into "
            "$HOME/mujoco_menagerie or set MENAGERIE_PATH."
        )
    return MenagerieXml(path)


def parse_menagerie_mesh(*parts: str) -> MenagerieMesh:
    """Resolve a menagerie-relative mesh path (.stl/.obj); raise if missing."""
    path = _resolve_menagerie().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"menagerie mesh not found: {path}. "
            "Clone https://github.com/google-deepmind/mujoco_menagerie into "
            "$HOME/mujoco_menagerie or set MENAGERIE_PATH."
        )
    return MenagerieMesh(path)


# Lazy menagerie-relative paths. `from paths import UR10E_XML` triggers
# `__getattr__("UR10E_XML")` exactly once on first import; the resolved path
# is then bound on the consumer module's namespace, so subsequent accesses
# are free. Adding a new entry is two lines: append to the right registry
# below, and declare the type under `TYPE_CHECKING`.
_MENAGERIE_XMLS: dict[str, tuple[str, ...]] = {
    "PIPER_XML": ("agilex_piper", "piper.xml"),
    "UR10E_XML": ("universal_robots_ur10e", "ur10e.xml"),
    "ROBOTIQ_2F85_XML": ("robotiq_2f85", "2f85.xml"),
    "TIAGO_XML": ("pal_tiago", "tiago.xml"),
    "D435I_XML": ("realsense_d435i", "d435i.xml"),
}

_MENAGERIE_MESHES: dict[str, tuple[str, ...]] = {
    "D405_MESH_STL": ("aloha", "assets", "d405_solid.stl"),
}

if TYPE_CHECKING:
    # Type-only declarations so static analysis sees the names with the
    # right refined type. At runtime they're produced by `__getattr__`.
    PIPER_XML: MenagerieXml
    UR10E_XML: MenagerieXml
    ROBOTIQ_2F85_XML: MenagerieXml
    TIAGO_XML: MenagerieXml
    D435I_XML: MenagerieXml
    D405_MESH_STL: MenagerieMesh


def __getattr__(name: str) -> MenagerieXml | MenagerieMesh:
    if name in _MENAGERIE_XMLS:
        return parse_menagerie_xml(*_MENAGERIE_XMLS[name])
    if name in _MENAGERIE_MESHES:
        return parse_menagerie_mesh(*_MENAGERIE_MESHES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Stanford Mobile ALOHA body — combined chassis + lift column + top
# platform extracted from the project-page CAD download. The original
# STL ships as a single monolithic mesh with all four ViperX arms
# attached; the four-arm subtrees are stripped (centroid-based filter,
# see `tools/strip_aloha_arms.py`) leaving only the central rolling
# platform. Units are MILLIMETRES — geom must scale by 1e-3 to compile
# in our metres-everywhere scene. Vendored at `assets/mobile_aloha/`
# so the demo is self-contained.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent
_MOBILE_ALOHA_MESHES: Path = _PROJECT_ROOT / "assets" / "mobile_aloha"


def _mobile_aloha_mesh(filename: str) -> MenagerieMesh:
    """Resolve a vendored Mobile ALOHA mesh path; raise if missing.

    Same parse-don't-validate pattern as the menagerie helpers — the
    returned `MenagerieMesh` carries the proof the file exists.
    """
    path = _MOBILE_ALOHA_MESHES / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"vendored Mobile ALOHA mesh not found: {path}. "
            "Re-extract from the Stanford Mobile ALOHA project-page CAD "
            "(see tools/strip_aloha_arms.py)."
        )
    return MenagerieMesh(path)


MOBILE_ALOHA_STANFORD_BODY_STL: MenagerieMesh = _mobile_aloha_mesh("aloha_body_no_arms.stl")
