"""Annotated 6-view render of the Stanford Mobile ALOHA base.

Produces a set of PNGs that label the surviving CAD geometry so the
user (and future-me) can disambiguate "front" vs "back" pairs without
guessing. Each render shows:

* The stripped body mesh as a semi-transparent steel-blue surface.
* The four mount-post column tops marked with **labelled coloured
  spheres** — green = front pair (highest y_cad), red = back pair.
* A CAD-frame axis triad at the mesh origin
  (red=+x_cad, green=+y_cad, blue=+z_cad) so the orientation is
  unambiguous.
* Each labelled point shows its CAD coordinates AND the body-frame
  coordinates that come out post -π/2 z-rotation — i.e. exactly the
  numbers that get plugged into `_ARM_MOUNT_X / _ARM_MOUNT_Y_ABS / _ARM_MOUNT_Z`
  in `robots/mobile_aloha.py`.

Six standard views (iso, top, front, back, left, right) so the
geometry can be confirmed from every angle.

Run: `uv run python tools/inspect_aloha_body.py [--out-dir /tmp]`
Output: `aloha_inspect_iso.png`, `…_top.png`, `…_front.png`, …
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STL = PROJECT_ROOT / "assets" / "mobile_aloha" / "aloha_body_no_arms.stl"


def find_mount_posts(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Connected components matching the mount-post signature: tall
    (z-extent > 200 mm), narrow xy footprint (< 200 mm each), and
    reaching near the platform top (cmax z > 800 mm). Returns the four
    Stanford CAD mount-post columns."""
    posts: list[trimesh.Trimesh] = []
    for c in mesh.split(only_watertight=False):
        cmin, cmax = c.bounds
        dx, dy, dz = cmax - cmin
        if dz > 200 and dx < 200 and dy < 200 and cmax[2] > 800:
            posts.append(c)
    return posts


def split_front_back(
    posts: list[trimesh.Trimesh],
) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Split posts into the higher-y_cad pair (front) and lower-y_cad pair (back)."""
    posts_sorted = sorted(posts, key=lambda c: -c.centroid[1])
    return posts_sorted[:2], posts_sorted[2:]


def cad_to_body(p_cad: np.ndarray) -> tuple[float, float, float]:
    """CAD-frame (mm) → body-frame (m) under the +π/2 z-rotation that
    `robots/mobile_aloha.py` applies to the geom: (x, y, z) → (-y, x, z).

    This sign matches the convention where ARM #3, #4 (the user's
    chosen follower pair, at CAD -y) end up pointing in the +x_world
    direction (toward the rack)."""
    x, y, z = p_cad / 1000.0
    return (-y, x, z)


def render_view(
    mesh: trimesh.Trimesh,
    front: list[trimesh.Trimesh],
    back: list[trimesh.Trimesh],
    elev: float,
    azim: float,
    title: str,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    body_collection = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.18,
        facecolor="steelblue",
        edgecolor="navy",
        linewidths=0.08,
    )
    ax.add_collection3d(body_collection)

    def _mark(c: trimesh.Trimesh, *, color: str, tag: str) -> None:
        ctr = c.centroid
        top_z = c.bounds[1, 2]
        body_xyz = cad_to_body(np.array([ctr[0], ctr[1], top_z]))
        label = (
            f"{tag}\n"
            f"CAD ({ctr[0]:+.0f}, {ctr[1]:+.0f}, {top_z:+.0f}) mm\n"
            f"body ({body_xyz[0]:+.3f}, {body_xyz[1]:+.3f}, {body_xyz[2]:+.3f}) m"
        )
        ax.scatter(
            [ctr[0]],
            [ctr[1]],
            [top_z],
            c=color,
            s=180,
            edgecolor="black",
            linewidth=1.0,
            zorder=10,
        )
        ax.text(
            ctr[0],
            ctr[1],
            top_z + 70,
            label,
            fontsize=7,
            ha="center",
            color="black",
            bbox={"facecolor": color, "alpha": 0.55, "edgecolor": "black", "pad": 1.5},
        )

    for c in front:
        _mark(c, color="lime", tag="FRONT")
    for c in back:
        _mark(c, color="salmon", tag="BACK")

    # CAD-frame axis triad at origin (mm).
    L = 250.0
    ax.quiver(0, 0, 0, L, 0, 0, color="red", arrow_length_ratio=0.18, linewidth=2)
    ax.quiver(0, 0, 0, 0, L, 0, color="green", arrow_length_ratio=0.18, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, L, color="blue", arrow_length_ratio=0.18, linewidth=2)
    ax.text(L * 1.12, 0, 0, "+x_cad", color="red", fontsize=10, weight="bold")
    ax.text(0, L * 1.12, 0, "+y_cad", color="green", fontsize=10, weight="bold")
    ax.text(0, 0, L * 1.12, "+z_cad", color="blue", fontsize=10, weight="bold")

    cmin, cmax = mesh.bounds
    ax.text2D(
        0.02,
        0.98,
        f"CAD bbox (mm)\n"
        f"  x [{cmin[0]:+.0f}, {cmax[0]:+.0f}]\n"
        f"  y [{cmin[1]:+.0f}, {cmax[1]:+.0f}]\n"
        f"  z [{cmin[2]:+.0f}, {cmax[2]:+.0f}]\n"
        f"faces: {len(mesh.faces)}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    ax.text2D(
        0.02,
        0.02,
        "post-rotation maps CAD (x, y, z) → body (y, -x, z)\n"
        "FRONT pair = the +y_cad pair (matches the 'squiggle' arms\n"
        "in the user's annotated reference image).",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        bbox={"facecolor": "lightyellow", "alpha": 0.85, "edgecolor": "gray"},
    )

    ax.set_xlabel("x_cad (mm)")
    ax.set_ylabel("y_cad (mm)")
    ax.set_zlabel("z_cad (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12)

    extents = mesh.extents
    max_ext = float(max(extents)) / 2.0 * 1.15
    ctr = mesh.centroid
    ax.set_xlim(ctr[0] - max_ext, ctr[0] + max_ext)
    ax.set_ylim(ctr[1] - max_ext, ctr[1] + max_ext)
    ax.set_zlim(-50, ctr[2] + max_ext)
    with contextlib.suppress(AttributeError):
        ax.set_box_aspect((1, 1, 1))

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded_geometry = trimesh.load(args.stl)
    if not isinstance(loaded_geometry, trimesh.Trimesh):
        raise TypeError(f"expected {args.stl} to load as a Trimesh, got {type(loaded_geometry)}")
    mesh = loaded_geometry
    print(f"loaded {args.stl}")
    print(f"  faces={len(mesh.faces)}  bbox_mm={mesh.bounds.tolist()}")

    posts = find_mount_posts(mesh)
    front, back = split_front_back(posts)
    print(f"found {len(posts)} mount posts ({len(front)} front, {len(back)} back)")
    for c in front:
        body = cad_to_body(np.array([c.centroid[0], c.centroid[1], c.bounds[1, 2]]))
        print(
            f"  FRONT  CAD top=({c.centroid[0]:+.0f},{c.centroid[1]:+.0f},{c.bounds[1, 2]:+.0f}) mm  "
            f"→ body=({body[0]:+.3f},{body[1]:+.3f},{body[2]:+.3f}) m"
        )
    for c in back:
        body = cad_to_body(np.array([c.centroid[0], c.centroid[1], c.bounds[1, 2]]))
        print(
            f"  BACK   CAD top=({c.centroid[0]:+.0f},{c.centroid[1]:+.0f},{c.bounds[1, 2]:+.0f}) mm  "
            f"→ body=({body[0]:+.3f},{body[1]:+.3f},{body[2]:+.3f}) m"
        )

    views: list[tuple[float, float, str, str]] = [
        (22, -55, "iso (default)", "iso"),
        (89, -90, "top-down (x_cad right, y_cad up)", "top"),
        (5, -90, "looking from +y_cad (CAD's 'front')", "front"),
        (5, 90, "looking from -y_cad (CAD's 'back')", "back"),
        (5, 0, "looking from +x_cad (right side)", "right"),
        (5, 180, "looking from -x_cad (left side)", "left"),
    ]
    for elev, azim, title, tag in views:
        out = args.out_dir / f"aloha_inspect_{tag}.png"
        render_view(mesh, front, back, elev, azim, title, out)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
