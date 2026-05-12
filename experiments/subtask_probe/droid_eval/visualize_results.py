#!/usr/bin/env python3
"""Generate a self-contained HTML report for DROID evaluation results.

Produces a single HTML file with embedded images (base64) that can be
opened in any browser and shared without dependencies.

Sections:
  1. Summary metrics — overall L2, cosine sim, gripper accuracy
  2. Pairwise comparisons — Wilcoxon tests, % better
  3. Subtask gallery — every frame with images + generated subtask text
  4. Action trajectories — per-dimension action plots for sample frames

Usage:
    uv run python experiments/subtask_probe/droid_eval/visualize_results.py \
        --samples_dir ./.experiments_cache/droid_eval \
        --output ./.experiments_cache/droid_eval/report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from .constants import CONDITION_COLORS, JOINT_NAMES
from .utils import load_manifest, load_subtask_entries, load_subtask_records

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ImageFormat = Literal["jpeg", "png"]


def _is_coherent_subtask(text: str) -> bool:
    """Check if a subtask string is coherent (non-empty ASCII text, not Unicode garbage)."""
    return bool(text.strip()) and text.isascii()


def image_to_base64(image_array: np.ndarray, fmt: ImageFormat = "jpeg", quality: int = 80) -> str:
    """Convert a numpy image array (HWC, uint8) to a base64-encoded data URI."""
    from PIL import Image

    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    if fmt == "jpeg":
        img.save(buffer, format="JPEG", quality=quality)
    else:
        img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "jpeg" else "image/png"
    return f"data:{mime};base64,{encoded}"


def make_action_svg(
    ground_truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    dim_idx: int,
    dim_name: str,
    width: int = 280,
    height: int = 120,
) -> str:
    """Generate an inline SVG showing action trajectories for one dimension."""
    margin_left = 35
    margin_right = 10
    margin_top = 20
    margin_bottom = 25
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Collect all values for axis scaling
    all_values = [ground_truth[:, dim_idx]]
    for pred in predictions.values():
        all_values.append(pred[:, dim_idx])
    all_flat = np.concatenate(all_values)
    y_min = float(np.min(all_flat))
    y_max = float(np.max(all_flat))
    y_range = y_max - y_min
    if y_range < 1e-6:
        y_range = 1.0
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    y_range = y_max - y_min

    num_steps = ground_truth.shape[0]

    def to_svg_coords(step: int, value: float) -> tuple[float, float]:
        x = margin_left + (step / max(num_steps - 1, 1)) * plot_width
        y = margin_top + (1 - (value - y_min) / y_range) * plot_height
        return x, y

    def make_polyline(values: np.ndarray, color: str, dashed: bool = False) -> str:
        points = " ".join(
            f"{to_svg_coords(i, v)[0]:.1f},{to_svg_coords(i, v)[1]:.1f}"
            for i, v in enumerate(values)
        )
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        return (
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>'
        )

    lines = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    lines.append(f'<rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>')

    # Title
    lines.append(
        f'<text x="{width // 2}" y="14" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#333">{dim_name}</text>'
    )

    # Y-axis labels
    for frac in [0, 0.5, 1.0]:
        y_val = y_min + frac * y_range
        _, svg_y = to_svg_coords(0, y_val)
        lines.append(
            f'<text x="{margin_left - 4}" y="{svg_y + 3:.0f}" text-anchor="end" font-size="9" font-family="sans-serif" fill="#999">{y_val:.2f}</text>'
        )
        lines.append(
            f'<line x1="{margin_left}" y1="{svg_y:.0f}" x2="{margin_left + plot_width}" y2="{svg_y:.0f}" stroke="#eee" stroke-width="0.5"/>'
        )

    # Ground truth (dashed gray)
    lines.append(
        make_polyline(ground_truth[:, dim_idx], CONDITION_COLORS["ground_truth"], dashed=True)
    )

    # Predictions
    for condition_name, pred in predictions.items():
        lines.append(make_polyline(pred[:, dim_idx], CONDITION_COLORS[condition_name]))

    lines.append("</svg>")
    return "\n".join(lines)


def generate_html(
    samples_dir: Path,
    max_gallery_frames: int,
    sample_action_frames: int,
    subtasks_path: Path | None = None,
) -> str:
    """Generate the full HTML report.

    ``subtasks_path`` overrides the default ``samples_dir/subtasks.json`` so the
    same report template can visualize subtasks from different backends (e.g.
    pi0.5 vs Gemini) pointing at the same cache.
    """
    # Load all data
    manifest = load_manifest(samples_dir)
    resolved_subtasks_path = subtasks_path or (samples_dir / "subtasks.json")
    subtask_index = load_subtask_entries(resolved_subtasks_path)

    # Load raw subtask list for unique count in header
    subtasks = load_subtask_records(resolved_subtasks_path)

    results_path = samples_dir / "results.json"
    has_results = results_path.exists()
    results = {}
    if has_results:
        with results_path.open() as f:
            results = json.load(f)

    predictions_dir = samples_dir / "predictions"
    has_predictions = predictions_dir.exists()

    total_frames = sum(ep["num_frames"] for ep in manifest)

    # --- Start HTML ---
    html_parts: list[str] = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DROID Subtask Evaluation Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; line-height: 1.5; padding: 20px; }
  .container { max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  h2 { font-size: 18px; margin: 30px 0 12px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h3 { font-size: 15px; margin: 20px 0 8px; color: #555; }
  .subtitle { color: #666; font-size: 14px; margin-bottom: 20px; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; background: white; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th, td { padding: 8px 12px; text-align: right; font-size: 13px; border-bottom: 1px solid #eee; }
  th { background: #f8f8f8; font-weight: 600; color: #555; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  tr:hover { background: #f9f9f9; }
  .card { background: white; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .episode-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }
  .episode-header .ep-id { font-weight: 600; font-size: 14px; }
  .episode-header .task-type { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: #e8f0fe; color: #1a73e8; }
  .episode-header .task-type.multi { background: #fce8e6; color: #d93025; }
  .instruction { font-style: italic; color: #555; font-size: 13px; margin-bottom: 12px; }
  .frame-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
  .frame-card { background: #fafafa; border-radius: 6px; padding: 10px; border: 1px solid #eee; }
  .frame-card .images { display: flex; gap: 6px; margin-bottom: 8px; }
  .frame-card img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }
  .frame-card .subtask { font-size: 13px; }
  .frame-card .subtask-text { font-weight: 600; color: #1a73e8; }
  .frame-card .frame-meta { font-size: 11px; color: #999; margin-top: 4px; }
  .legend { display: flex; gap: 16px; margin: 10px 0; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 4px; font-size: 12px; }
  .legend-swatch { width: 20px; height: 3px; border-radius: 2px; }
  .legend-swatch.dashed { background: repeating-linear-gradient(90deg, currentColor 0, currentColor 4px, transparent 4px, transparent 7px); height: 2px; }
  .action-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 8px; }
  .stat-highlight { font-size: 28px; font-weight: 700; }
  .stat-label { font-size: 12px; color: #888; }
  .stats-row { display: flex; gap: 30px; margin: 16px 0; flex-wrap: wrap; }
  .stat-box { text-align: center; }
  .sig { color: #d93025; font-weight: 600; }
  .not-sig { color: #999; }
</style>
</head>
<body>
<div class="container">
""")

    # --- Header ---
    html_parts.append(f"""
<h1>DROID Subtask Evaluation Report</h1>
<p class="subtitle">{len(manifest)} episodes &middot; {total_frames} frames &middot; {len({s["subtask_text"] for s in subtasks})} unique subtasks</p>
""")

    # --- Summary stats ---
    if has_results:
        html_parts.append("<h2>Summary Metrics</h2>")

        # Stat boxes
        baseline_l2 = results["baseline"]["overall"]["l2_distance"]["mean"]
        subtask_l2 = results["subtask"]["overall"]["l2_distance"]["mean"]
        p_val = results.get("pairwise", {}).get("baseline_vs_subtask", {}).get("wilcoxon_p", 1.0)

        html_parts.append('<div class="stats-row">')
        for label, l2_val in [("Baseline L2", baseline_l2), ("Subtask L2", subtask_l2)]:
            html_parts.append(
                f'<div class="stat-box"><div class="stat-highlight">{l2_val:.3f}</div><div class="stat-label">{label}</div></div>'
            )
        cls = "sig" if p_val < 0.05 else "not-sig"
        sig_label = "significant" if p_val < 0.05 else "not significant"
        html_parts.append(
            f'<div class="stat-box"><div class="stat-highlight {cls}">p={p_val:.4f}</div><div class="stat-label">Baseline vs Subtask ({sig_label})</div></div>'
        )
        html_parts.append("</div>")

        # Overall metrics table
        html_parts.append("""
<table>
<tr><th>Condition</th><th>L2 Distance</th><th>Cosine Similarity</th><th>Gripper Accuracy</th><th>N</th></tr>
""")
        for condition in ["baseline", "subtask"]:
            o = results[condition]["overall"]
            l2 = o["l2_distance"]
            cos = o["cosine_similarity"]
            grip = o["gripper_accuracy"]
            html_parts.append(
                f"<tr><td><strong>{condition}</strong></td>"
                f"<td>{l2['mean']:.4f} &plusmn; {l2['std']:.4f}</td>"
                f"<td>{cos['mean']:.4f} &plusmn; {cos['std']:.4f}</td>"
                f"<td>{grip['mean']:.4f} &plusmn; {grip['std']:.4f}</td>"
                f"<td>{o['n_frames']}</td></tr>"
            )
        html_parts.append("</table>")

        # Per-dimension MAE table
        html_parts.append("<h3>Per-Dimension MAE</h3>")
        html_parts.append("<table><tr><th>Condition</th>")
        for name in JOINT_NAMES:
            html_parts.append(f"<th>{name}</th>")
        html_parts.append("</tr>")
        for condition in ["baseline", "subtask"]:
            per_dim = results[condition]["overall"]["per_dim_mae"]
            html_parts.append(f"<tr><td><strong>{condition}</strong></td>")
            for name in JOINT_NAMES:
                val = per_dim.get(name, 0)
                html_parts.append(f"<td>{val:.4f}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")

        # Pairwise comparisons
        html_parts.append("<h3>Pairwise Comparisons</h3>")
        html_parts.append(
            "<table><tr><th>Comparison</th><th>L2 Diff (mean)</th><th>% B Better</th><th>Wilcoxon p</th></tr>"
        )
        for key, data in results.get("pairwise", {}).items():
            p_val = data.get("wilcoxon_p", None)
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            cls = "sig" if p_val is not None and p_val < 0.05 else "not-sig"
            html_parts.append(
                f"<tr><td>{key}</td>"
                f"<td>{data['l2_diff_mean']:.4f}</td>"
                f"<td>{data['pct_b_better']:.1f}%</td>"
                f'<td class="{cls}">{p_str}</td></tr>'
            )
        html_parts.append("</table>")

    # --- Subtask Gallery ---
    html_parts.append("<h2>Subtask Gallery</h2>")
    html_parts.append(
        '<p class="subtitle">Click an episode to expand. Coherent/garbage counts shown in the header.</p>'
    )

    for episode in manifest:
        episode_id = episode["episode_id"]
        task_type_cls = "multi" if episode["task_type"] == "multi_step" else ""

        # Count coherent vs garbage subtasks for this episode
        ep_subtasks = [subtask_index.get((episode_id, f["frame_idx"])) for f in episode["frames"]]
        ep_subtasks = [s for s in ep_subtasks if s is not None]
        coherent_count = sum(1 for s in ep_subtasks if _is_coherent_subtask(s["subtask_text"]))
        garbage_count = len(ep_subtasks) - coherent_count
        quality_label = f"{coherent_count}/{len(ep_subtasks)} coherent"
        if garbage_count > 0:
            quality_label += f", {garbage_count} garbage"
        quality_color = (
            "#4caf50"
            if garbage_count == 0
            else ("#ff9800" if coherent_count > garbage_count else "#f44336")
        )

        html_parts.append(f"""
<details class="card" {"open" if coherent_count > garbage_count else ""}>
<summary style="cursor:pointer;user-select:none">
  <div class="episode-header" style="display:inline-flex">
    <span class="ep-id">{episode_id}</span>
    <span class="task-type {task_type_cls}">{episode["task_type"].replace("_", " ")}</span>
    <span style="font-size:12px;color:{quality_color};font-weight:600">{quality_label}</span>
    <span style="font-size:12px;color:#999">{episode["num_frames"]} frames</span>
  </div>
  <div class="instruction">&ldquo;{episode["instruction"]}&rdquo;</div>
</summary>
<div class="frame-grid">
""")

        for frame_info in episode["frames"]:
            frame_idx = frame_info["frame_idx"]
            subtask_entry = subtask_index.get((episode_id, frame_idx))

            # Load images
            frame_path = samples_dir / frame_info["file"]
            if frame_path.exists():
                frame_data = np.load(frame_path)
                ext_uri = image_to_base64(frame_data["exterior_image"])
                wrist_uri = image_to_base64(frame_data["wrist_image"])
            else:
                ext_uri = ""
                wrist_uri = ""

            subtask_text = subtask_entry["subtask_text"] if subtask_entry else "(no subtask)"
            gen_time = subtask_entry.get("generation_time_s", 0) if subtask_entry else 0
            is_garbage = not _is_coherent_subtask(subtask_text)
            border_color = "#f44336" if is_garbage else "#eee"

            html_parts.append(f"""
<div class="frame-card" style="border-color:{border_color}">
  <div class="images">
    <img src="{ext_uri}" alt="exterior" title="Exterior camera">
    <img src="{wrist_uri}" alt="wrist" title="Wrist camera">
  </div>
  <div class="subtask">Subtask: <span class="subtask-text" {'style="color:#f44336"' if is_garbage else ""}>{subtask_text}</span></div>
  <div class="frame-meta">Frame {frame_idx} &middot; {gen_time:.2f}s</div>
</div>
""")

        html_parts.append("</div></details>")  # frame-grid, details

    # --- Action Trajectories ---
    if has_predictions:
        html_parts.append("<h2>Action Trajectories (Sample Frames)</h2>")
        html_parts.append("""
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#888;"></div> Ground truth</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#4a90d9;"></div> Baseline</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#d94a4a;"></div> Subtask</div>
</div>
""")

        action_frame_count = 0
        for episode in manifest:
            episode_id = episode["episode_id"]
            # Pick evenly spaced frames from this episode
            frames = episode["frames"]
            if len(frames) <= 2:
                selected_frames = frames
            else:
                step = max(1, len(frames) // 2)
                selected_frames = frames[::step][:3]

            for frame_info in selected_frames:
                if action_frame_count >= sample_action_frames:
                    break

                frame_idx = frame_info["frame_idx"]
                pred_file = predictions_dir / f"{episode_id}_frame_{frame_idx:05d}.npz"
                frame_file = samples_dir / frame_info["file"]

                if not pred_file.exists() or not frame_file.exists():
                    continue

                pred_data = np.load(pred_file)
                frame_data = np.load(frame_file)
                ground_truth = frame_data["ground_truth_actions"]

                predictions = {
                    "baseline": pred_data["baseline"],
                    "subtask": pred_data["subtask"],
                }

                # Trim to common horizon
                min_h = min(ground_truth.shape[0], *(p.shape[0] for p in predictions.values()))
                ground_truth_trimmed = ground_truth[:min_h]
                predictions_trimmed = {k: v[:min_h] for k, v in predictions.items()}

                subtask_entry = subtask_index.get((episode_id, frame_idx))
                subtask_text = subtask_entry["subtask_text"] if subtask_entry else ""
                ext_uri = image_to_base64(frame_data["exterior_image"])

                html_parts.append(f"""
<div class="card">
  <div class="episode-header">
    <span class="ep-id">{episode_id} / frame {frame_idx}</span>
    <span style="font-size:12px;color:#999">Subtask: <strong>{subtask_text}</strong></span>
  </div>
  <div style="display:flex;gap:12px;align-items:flex-start">
    <img src="{ext_uri}" style="width:120px;height:120px;object-fit:cover;border-radius:6px;flex-shrink:0">
    <div class="action-grid" style="flex:1">
""")
                num_dims = min(ground_truth_trimmed.shape[1], len(JOINT_NAMES))
                for dim_idx in range(num_dims):
                    svg = make_action_svg(
                        ground_truth_trimmed, predictions_trimmed, dim_idx, JOINT_NAMES[dim_idx]
                    )
                    html_parts.append(svg)

                html_parts.append("</div></div></div>")
                action_frame_count += 1

            if action_frame_count >= sample_action_frames:
                break

    # --- Footer ---
    html_parts.append("""
<div style="margin-top:40px;padding:16px 0;border-top:1px solid #ddd;color:#999;font-size:12px">
  Generated by experiments/subtask_probe/droid_eval/visualize_results.py
</div>
</div>
</body>
</html>
""")

    return "".join(html_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML evaluation report")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file (default: samples_dir/report.html)",
    )
    parser.add_argument(
        "--max_gallery_frames", type=int, default=100, help="Max frames to show in subtask gallery"
    )
    parser.add_argument(
        "--sample_action_frames",
        type=int,
        default=15,
        help="Number of frames to show action trajectories for",
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        default=None,
        help="Path to the subtasks JSON (defaults to samples_dir/subtasks.json)",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_path = Path(args.output) if args.output else samples_dir / "report.html"
    subtasks_path = Path(args.subtasks) if args.subtasks else None

    logger.info("Generating report from %s", samples_dir)
    html = generate_html(
        samples_dir,
        max_gallery_frames=args.max_gallery_frames,
        sample_action_frames=args.sample_action_frames,
        subtasks_path=subtasks_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Report saved to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
