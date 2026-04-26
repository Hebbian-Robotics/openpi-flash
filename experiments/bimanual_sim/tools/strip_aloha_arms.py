"""Extract the Stanford Mobile ALOHA *base* from the project-page CAD.

The CAD ships as a single monolithic STL containing the chassis,
central column, front + rear mount-post columns, dual top platforms,
and all four ViperX arms with their grippers and wrist cameras. The
user's annotated screenshot (red squiggles on the two front arms, red
X's on the two rear arms) makes the intent explicit: drop only the
four ViperX arm assemblies — keep the entire base, including the four
red mount-post columns, so the UR10e arms can sit on top of the front
pair.

Filter rule: split the STL into connected components, drop every
component whose centroid is above z = 1050 mm. Empirically the four
ViperX arms together contribute 46 k of the 54 k faces and live
exclusively above z = 1000 mm (their shoulder bases sit at z ≈ 1000;
the arm links extend up to z ≈ 1500). The cutoff is 1050 mm so
shoulder-base mounting plates just above 1000 mm stay with the base.
Everything below the cutoff is base structure: chassis, drive bay,
vertical column legs, the four red mount-post columns, and the two
horizontal top platforms (yellow front, green rear).

Run: `uv run python tools/strip_aloha_arms.py`. Output overwrites
`assets/mobile_aloha/aloha_body_no_arms.stl`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_STL = PROJECT_ROOT / "assets" / "mobile_aloha_stanford.stl"
OUTPUT_STL = PROJECT_ROOT / "assets" / "mobile_aloha" / "aloha_body_no_arms.stl"

# All four ViperX arms have shoulder bases at z ≈ 1000 mm and project
# upward; nothing else in the CAD is above this height. Threshold set
# slightly higher (1050) so we keep any shoulder-base mounting plates
# whose centroids sit just above 1000 mm but are visually part of the
# mount post, not the arm.
ARM_Z_THRESHOLD_MM = 1050.0


def main() -> None:
    print(f"loading: {INPUT_STL}")
    mesh = trimesh.load(INPUT_STL)
    print(f"  faces={len(mesh.faces)}  bounds_mm={mesh.bounds.tolist()}")

    components = mesh.split(only_watertight=False)
    print(f"  {len(components)} connected components")

    keep = [c for c in components if c.centroid[2] <= ARM_Z_THRESHOLD_MM]
    drop = [c for c in components if c.centroid[2] > ARM_Z_THRESHOLD_MM]
    print(
        f"  keep: {len(keep)} comps / {sum(len(c.faces) for c in keep)} faces "
        f"(centroid z ≤ {ARM_Z_THRESHOLD_MM:.0f} mm)"
    )
    print(
        f"  drop: {len(drop)} comps / {sum(len(c.faces) for c in drop)} faces "
        f"(centroid z > {ARM_Z_THRESHOLD_MM:.0f} mm — the four ViperX arms)"
    )

    body = trimesh.util.concatenate(keep)
    body_bounds = body.bounds
    print(f"  output faces={len(body.faces)}  bounds_mm={body_bounds.tolist()}")

    OUTPUT_STL.parent.mkdir(parents=True, exist_ok=True)
    body.export(OUTPUT_STL)
    print(f"wrote: {OUTPUT_STL}")

    # Quick analysis: find the four mount-post column centres at the top
    # of the post (z near max). These are where UR10e arms ought to bolt on.
    top_z = body_bounds[1, 2]
    # Slab right under the top: that's where the post tops + platforms live.
    slab_lo = top_z - 60
    mask = (body.vertices[:, 2] >= slab_lo) & (body.vertices[:, 2] <= top_z)
    print(f"\ntop-slab z=[{slab_lo:.0f}, {top_z:.0f}] mm has {mask.sum()} verts")
    if mask.any():
        v = body.vertices[mask]
        print(f"  x range: [{v[:, 0].min():+.0f}, {v[:, 0].max():+.0f}]")
        print(f"  y range: [{v[:, 1].min():+.0f}, {v[:, 1].max():+.0f}]")

    # Re-find the four post-top centres by clustering top-slab vertices into 4
    # quadrants by sign of (x, y) and reporting each cluster's mean.
    if mask.sum() >= 4:
        v = body.vertices[mask]
        for sx in (-1, +1):
            for sy in (-1, +1):
                q = v[(np.sign(v[:, 0]) == sx) & (np.sign(v[:, 1]) == sy)]
                if len(q) > 0:
                    print(
                        f"  quadrant sign(x)={sx:+d} sign(y)={sy:+d}: "
                        f"n={len(q):4d} mean=({q[:, 0].mean():+.0f}, {q[:, 1].mean():+.0f}, {q[:, 2].mean():+.0f})"
                    )


if __name__ == "__main__":
    main()
