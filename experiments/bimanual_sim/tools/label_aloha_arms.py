"""Render the *original* Stanford Mobile ALOHA CAD with each of the
four ViperX arms colour-coded and numbered, so the user can point at
the diagram and say which two to replace with UR10e arms.

The original CAD ships as a single monolithic STL with all four arms
baked in (54 k faces total; arms are 47 k of those). This script:

1. Loads `assets/mobile_aloha_stanford.stl`.
2. Splits into connected components, groups arm components (centroid
   z > 1000 mm) into four clusters by xy quadrant — one cluster per
   ViperX arm.
3. Renders the full body in steel-blue, with the four arm clusters in
   four distinct colours: ARM #1 = red, #2 = orange, #3 = green,
   #4 = magenta. Each arm gets a big numeric label at its centroid
   plus a coordinate readout.

Output: 4 PNGs (iso, top, front, back) in --out-dir.

Run: `uv run --with matplotlib python tools/label_aloha_arms.py`
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
DEFAULT_STL = PROJECT_ROOT / "assets" / "mobile_aloha_stanford.stl"

# Distinct, high-contrast colours for the four arms. Order corresponds
# to ARM #1, #2, #3, #4 — the script assigns numbers below by xy
# quadrant so the labels are stable across runs.
ARM_COLOURS = ("#e41a1c", "#ff7f00", "#4daf4a", "#984ea3")  # red, orange, green, purple


def split_arms_by_quadrant(
    arm_components: list[trimesh.Trimesh],
) -> dict[int, list[trimesh.Trimesh]]:
    """Group arm components into 4 clusters by sign of (x_cad, y_cad).

    Returns dict: arm_number (1..4) -> list of components in that arm.
    Numbering convention:
        ARM #1: x_cad > 0, y_cad > 0  (front-right in CAD)
        ARM #2: x_cad < 0, y_cad > 0  (front-left in CAD)
        ARM #3: x_cad < 0, y_cad < 0  (back-left in CAD)
        ARM #4: x_cad > 0, y_cad < 0  (back-right in CAD)
    """
    clusters: dict[int, list[trimesh.Trimesh]] = {1: [], 2: [], 3: [], 4: []}
    for c in arm_components:
        cx, cy, _ = c.centroid
        if cx > 0 and cy > 0:
            clusters[1].append(c)
        elif cx < 0 and cy > 0:
            clusters[2].append(c)
        elif cx < 0 and cy < 0:
            clusters[3].append(c)
        else:
            clusters[4].append(c)
    return clusters


def cluster_centroid(comps: list[trimesh.Trimesh]) -> np.ndarray:
    """Mean centroid of a cluster of components, weighted by face count."""
    if not comps:
        return np.array([0.0, 0.0, 0.0])
    weights = np.array([len(c.faces) for c in comps], dtype=float)
    centroids = np.array([c.centroid for c in comps])
    return np.average(centroids, axis=0, weights=weights)


def cluster_bounds(comps: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    """Combined min/max bounds over a cluster of components."""
    all_verts = np.vstack([c.vertices for c in comps])
    return all_verts.min(axis=0), all_verts.max(axis=0)


def render_view(
    body_components: list[trimesh.Trimesh],
    arm_clusters: dict[int, list[trimesh.Trimesh]],
    elev: float,
    azim: float,
    title: str,
    out: Path,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    body_mesh = trimesh.util.concatenate(body_components)
    ax.add_collection3d(
        Poly3DCollection(
            body_mesh.vertices[body_mesh.faces],
            alpha=0.35,
            facecolor="steelblue",
            edgecolor="navy",
            linewidths=0.08,
        )
    )

    for arm_num, comps in arm_clusters.items():
        if not comps:
            continue
        arm_mesh = trimesh.util.concatenate(comps)
        colour = ARM_COLOURS[arm_num - 1]
        ax.add_collection3d(
            Poly3DCollection(
                arm_mesh.vertices[arm_mesh.faces],
                alpha=0.65,
                facecolor=colour,
                edgecolor="black",
                linewidths=0.05,
            )
        )

        # Small labelled marker offset above the arm so it doesn't
        # occlude the actual mesh geometry.
        ctr = cluster_centroid(comps)
        _cmin, cmax = cluster_bounds(comps)
        # Push label OUTWARD horizontally + above so it doesn't sit on top of the arm.
        label_x = ctr[0] + (250 if ctr[0] > 0 else -250)
        label_y = ctr[1] + (250 if ctr[1] > 0 else -250)
        label_z = cmax[2] + 250
        ax.text(
            label_x,
            label_y,
            label_z,
            f"#{arm_num}",
            fontsize=16,
            ha="center",
            weight="bold",
            color="white",
            bbox={"facecolor": colour, "edgecolor": "black", "pad": 4},
            zorder=21,
        )
        # Thin guide-line from label to the arm centroid.
        ax.plot(
            [label_x, ctr[0]],
            [label_y, ctr[1]],
            [label_z, ctr[2]],
            color=colour,
            linewidth=1.0,
            alpha=0.7,
            zorder=19,
        )

    # CAD-frame axis triad at origin (mm).
    L = 300.0
    ax.quiver(0, 0, 0, L, 0, 0, color="red", arrow_length_ratio=0.18, linewidth=2.5)
    ax.quiver(0, 0, 0, 0, L, 0, color="green", arrow_length_ratio=0.18, linewidth=2.5)
    ax.quiver(0, 0, 0, 0, 0, L, color="blue", arrow_length_ratio=0.18, linewidth=2.5)
    ax.text(L * 1.10, 0, 0, "+x_cad", color="red", fontsize=11, weight="bold")
    ax.text(0, L * 1.10, 0, "+y_cad", color="green", fontsize=11, weight="bold")
    ax.text(0, 0, L * 1.10, "+z_cad", color="blue", fontsize=11, weight="bold")

    ax.text2D(
        0.02,
        0.98,
        "Stanford Mobile ALOHA — original CAD\n"
        "Tell me which two arms to replace with UR10e\n"
        "(e.g. 'use #1 and #2' to keep the front pair).",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"facecolor": "lightyellow", "alpha": 0.9, "edgecolor": "gray"},
    )

    ax.set_xlabel("x_cad (mm)")
    ax.set_ylabel("y_cad (mm)")
    ax.set_zlabel("z_cad (mm)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=13)

    full_mesh = trimesh.util.concatenate(
        body_components + [c for comps in arm_clusters.values() for c in comps]
    )
    extents = full_mesh.extents
    max_ext = float(max(extents)) / 2.0 * 1.10
    ctr = full_mesh.centroid
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
    parser.add_argument(
        "--arm-z-threshold",
        type=float,
        default=1000.0,
        help="centroid-z (mm) cutoff above which a component is part of an arm",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading: {args.stl}")
    mesh = trimesh.load(args.stl)
    print(f"  faces={len(mesh.faces)}  bounds_mm={mesh.bounds.tolist()}")

    components = mesh.split(only_watertight=False)
    body = [c for c in components if c.centroid[2] <= args.arm_z_threshold]
    arms = [c for c in components if c.centroid[2] > args.arm_z_threshold]
    print(f"  {len(components)} comps → {len(body)} body, {len(arms)} arm")

    clusters = split_arms_by_quadrant(arms)
    for n, comps in clusters.items():
        if not comps:
            continue
        ctr = cluster_centroid(comps)
        _cmin, cmax = cluster_bounds(comps)
        print(
            f"  ARM #{n}: {len(comps):2d} comps, "
            f"centroid ({ctr[0]:+5.0f}, {ctr[1]:+5.0f}, {ctr[2]:+5.0f}) mm  "
            f"y_sign={'+y' if ctr[1] > 0 else '-y'} "
            f"x_sign={'+x' if ctr[0] > 0 else '-x'}  "
            f"z_top={cmax[2]:.0f} mm"
        )

    views: list[tuple[float, float, str, str]] = [
        (22, -55, "iso (default 3D view)", "iso"),
        (89, -90, "top-down (x_cad right, y_cad up)", "top"),
        (5, -90, "looking from +y_cad direction", "front_y"),
        (5, 0, "looking from +x_cad direction", "front_x"),
    ]
    for elev, azim, title, tag in views:
        out = args.out_dir / f"aloha_arms_labelled_{tag}.png"
        render_view(body, clusters, elev, azim, title, out)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
