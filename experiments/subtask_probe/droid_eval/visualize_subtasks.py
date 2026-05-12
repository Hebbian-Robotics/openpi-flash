#!/usr/bin/env python3
"""Render frames + per-frame subtask text without running the action eval.

Two output modes:

  * **HTML** (default): one self-contained HTML file per episode with the
    exterior frame and its subtask caption inlined side-by-side. Good for
    scrolling through a full episode.

  * **Video** (``--video``): one mp4 per episode with the subtask string
    drawn onto the exterior frame, played back at ``--fps`` (default 2 Hz,
    matching the DROID cache subsample rate so wall-clock ≈ real time).

Usage::

    uv run python -m experiments.subtask_probe.droid_eval.visualize_subtasks \\
        --samples_dir ./.experiments_cache/droid_eval_ah15 \\
        --subtasks ./.experiments_cache/droid_eval_ah15/subtasks_comet_qwen30b.json \\
        --output_dir ./.experiments_cache/droid_eval_ah15/subtask_videos \\
        --video --fps 2
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
from pathlib import Path

import imageio.v3 as iio3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.subtask_probe.droid_eval.utils import load_manifest, load_subtask_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_frame_images(frame_path: Path) -> dict[str, np.ndarray]:
    """Load both camera views from a cached .npz as uint8 HxWx3 arrays.

    Returns a dict keyed by camera name (``exterior``, ``wrist``). Both views
    exist on every cached DROID frame; surfacing both makes plan-tracking
    failures easier to diagnose (e.g. gripper is clearly closed in the wrist
    view but the reasoner still emits "grasp the cube").
    """
    data = np.load(frame_path)
    return {
        "exterior": np.asarray(data["exterior_image"], dtype=np.uint8),
        "wrist": np.asarray(data["wrist_image"], dtype=np.uint8),
    }


def _composite_side_by_side(views: dict[str, np.ndarray], separator_px: int = 4) -> np.ndarray:
    """Concatenate multiple same-height views horizontally with a dark divider.

    Returns a single uint8 HxWx3 array so downstream caption/encode code can
    treat the multi-view frame as one image.
    """
    images = list(views.values())
    heights = {img.shape[0] for img in images}
    if len(heights) != 1:
        raise ValueError(f"camera views must share a height; got {heights}")
    h = images[0].shape[0]
    divider = np.full((h, separator_px, 3), 24, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, img in enumerate(images):
        if i > 0:
            pieces.append(divider)
        pieces.append(img)
    return np.concatenate(pieces, axis=1)


def _draw_caption(
    image: np.ndarray,
    caption: str,
    footer: str | None = None,
    camera_labels: list[tuple[int, str]] | None = None,
) -> np.ndarray:
    """Overlay a subtask caption on the bottom of the frame plus a dark banner.

    ``camera_labels`` is an optional list of ``(x_offset, label)`` pairs drawn
    in the top-left of each view, useful when the image is a composite of
    multiple camera feeds stitched together by ``_composite_side_by_side``.
    """
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Try a bundled TrueType font; fall back to default if unavailable.
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(16, h // 18))
    except OSError:
        font = ImageFont.load_default()
    label_font = ImageFont.load_default()

    if camera_labels:
        for x_offset, label in camera_labels:
            draw.rectangle(
                [(x_offset, 0), (x_offset + 8 + 6 * len(label), 18)],
                fill=(0, 0, 0, 180),
            )
            draw.text((x_offset + 4, 3), label, fill=(220, 220, 220, 255), font=label_font)

    banner_h = int(h * 0.22)
    draw.rectangle([(0, h - banner_h), (w, h)], fill=(0, 0, 0, 180))

    text = caption or "<empty>"
    pad = 10
    draw.text((pad, h - banner_h + pad), text, fill=(255, 255, 255, 255), font=font)

    if footer:
        draw.text((pad, h - 14), footer, fill=(200, 200, 200, 255), font=label_font)

    return np.asarray(img)


def _encode_png_base64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _write_html(
    out_path: Path,
    episode_id: str,
    instruction: str,
    items: list[tuple[int, str, dict[str, np.ndarray]]],
) -> None:
    """Emit a self-contained HTML file with both camera views + subtask per step."""
    rows = []
    for frame_idx, subtask, views in items:
        ext_b64 = _encode_png_base64(views["exterior"])
        wrist_b64 = _encode_png_base64(views["wrist"])
        rows.append(
            f"<tr>"
            f"<td class='idx'>{frame_idx}</td>"
            f"<td class='img'>"
            f"<div class='cam-label'>exterior</div>"
            f"<img src='data:image/png;base64,{ext_b64}' width='320'/>"
            f"</td>"
            f"<td class='img'>"
            f"<div class='cam-label'>wrist</div>"
            f"<img src='data:image/png;base64,{wrist_b64}' width='320'/>"
            f"</td>"
            f"<td class='sub'>{subtask or '<em>(empty)</em>'}</td>"
            f"</tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Subtasks — {episode_id}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;background:#0b0f14;color:#e6edf3}"
        "h1{font-size:1.1rem;margin-bottom:.25rem}"
        "p.meta{color:#8b949e;margin-top:0}"
        "table{border-collapse:collapse;margin-top:1rem}"
        "td{padding:8px;vertical-align:middle;border-bottom:1px solid #30363d}"
        "td.idx{color:#8b949e;font-family:ui-monospace,Menlo,Consolas,monospace;width:60px}"
        "td.sub{font-size:1rem;max-width:420px}"
        "td.img{padding:8px}"
        ".cam-label{color:#8b949e;font-size:.75rem;margin-bottom:4px;"
        "text-transform:uppercase;letter-spacing:.05em}"
        "img{border-radius:6px;display:block}"
        "</style></head><body>"
        f"<h1>{episode_id}</h1>"
        f"<p class='meta'>Instruction: <code>{instruction}</code> &middot; {len(items)} frames</p>"
        "<table>"
        "<thead><tr><th>frame</th><th>exterior</th><th>wrist</th><th>subtask</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
        "</body></html>"
    )
    out_path.write_text(html)


def _write_mp4(
    out_path: Path,
    frames_with_captions: list[np.ndarray],
    fps: int,
) -> None:
    """Write an mp4 of the given captioned frames using imageio-ffmpeg."""
    iio3.imwrite(
        out_path,
        np.stack(frames_with_captions),
        fps=fps,
        codec="libx264",
        macro_block_size=1,  # Allow odd dims without forced rescaling
    )


def _process_episode(
    samples_dir: Path,
    episode: dict,
    subtask_index: dict[tuple[str, int], str],
    output_dir: Path,
    video: bool,
    fps: int,
) -> None:
    episode_id = episode["episode_id"]
    instruction = episode["instruction"]

    items: list[tuple[int, str, dict[str, np.ndarray]]] = []
    captioned: list[np.ndarray] = []

    for frame_info in episode["frames"]:
        frame_idx = frame_info["frame_idx"]
        views = _load_frame_images(samples_dir / frame_info["file"])
        subtask = subtask_index.get((episode_id, frame_idx), "")
        items.append((frame_idx, subtask, views))
        if video:
            composite = _composite_side_by_side(views)
            ext_w = views["exterior"].shape[1]
            captioned.append(
                _draw_caption(
                    composite,
                    caption=subtask,
                    footer=f"{episode_id} frame {frame_idx}  |  task: {instruction}",
                    camera_labels=[(4, "EXTERIOR"), (ext_w + 8, "WRIST")],
                )
            )

    if video:
        mp4_path = output_dir / f"{episode_id}.mp4"
        _write_mp4(mp4_path, captioned, fps=fps)
        logger.info("%s -> %s (%d frames, %d fps)", episode_id, mp4_path, len(captioned), fps)
    else:
        html_path = output_dir / f"{episode_id}.html"
        _write_html(html_path, episode_id, instruction, items)
        logger.info("%s -> %s (%d frames)", episode_id, html_path, len(items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render subtasks over DROID frames")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--subtasks", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write per-episode HTML or mp4 files into.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Emit mp4 per episode (caption drawn on each frame) instead of HTML.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames-per-second for mp4 output. Default 2 (~DROID cache subsample rate).",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(samples_dir)
    subtask_index = load_subtask_index(Path(args.subtasks))
    logger.info("Loaded %d episodes, %d subtask records", len(manifest), len(subtask_index))

    for episode in manifest:
        _process_episode(
            samples_dir=samples_dir,
            episode=episode,
            subtask_index=subtask_index,
            output_dir=output_dir,
            video=args.video,
            fps=args.fps,
        )

    logger.info("Done. Output: %s", output_dir)


if __name__ == "__main__":
    main()
