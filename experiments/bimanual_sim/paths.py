"""Filesystem paths used by scenes.

Resolved lazily so running the code on a new machine (different user, laptop
vs. EC2, etc.) doesn't require editing Python. Convention: clone
`mujoco_menagerie` into `$HOME/mujoco_menagerie`; override with the
`MENAGERIE_PATH` env var if you put it elsewhere.

External filesystem paths are parsed at import via `parse_menagerie_xml`: a
missing MJCF raises a clear error here rather than later as a cryptic MuJoCo
load failure downstream. Downstream code consumes the refined `MenagerieXml`
type, which carries the proof that the file exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NewType

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

    The raise-at-import gives one clear failure site ("menagerie not cloned"
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


MENAGERIE: Path = _resolve_menagerie()
PIPER_XML: MenagerieXml = parse_menagerie_xml("agilex_piper", "piper.xml")
# Universal Robots UR10e — 6-DoF revolute arm with 1.3 m reach. Used by
# scenes/data_center.py as the bimanual workhorse; the upstream MJCF
# ships standalone (no built-in gripper) so we attach a Robotiq 2F-85
# at its `attachment_site` on `wrist_3_link`.
UR10E_XML: MenagerieXml = parse_menagerie_xml("universal_robots_ur10e", "ur10e.xml")
# Robotiq 2F-85 parallel-jaw gripper — tendon-coupled, single actuator
# (`fingers_actuator`, ctrl 0..255). Mounts on the UR10e's wrist flange.
ROBOTIQ_2F85_XML: MenagerieXml = parse_menagerie_xml("robotiq_2f85", "2f85.xml")
FRANKA_SCENE_XML: MenagerieXml = parse_menagerie_xml("franka_emika_panda", "scene.xml")
# PAL TIAGo single-arm: differential-drive base + torso lift. Used by
# scenes/data_center.py as the mobile embodiment (its single arm gets stripped
# by robots.tiago.load_tiago).
TIAGO_XML: MenagerieXml = parse_menagerie_xml("pal_tiago", "tiago.xml")
# Intel RealSense D435i package (mesh assets + standalone MJCF).
D435I_XML: MenagerieXml = parse_menagerie_xml("realsense_d435i", "d435i.xml")
# D405 wrist cam mesh lives inside ALOHA's asset dir. Referenced directly
# (no separate D405 package in Menagerie).
D405_MESH_STL: MenagerieMesh = parse_menagerie_mesh("aloha", "assets", "d405_solid.stl")

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
