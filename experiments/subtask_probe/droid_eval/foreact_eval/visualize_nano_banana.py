#!/usr/bin/env python3
"""Render the nano-banana foresight chain as a side-by-side mp4.

``actual | foresight`` per frame, subtask caption overlaid on the bottom,
episode/frame footer under it. Runs at 2 fps by default to match the
ForeAct v2 golden chain mp4.

Caption/composite drawing is shared with ``visualize_subtasks.py`` so this
visualizer and ``visualize_foreact.py`` produce stylistically matched
videos. Episode-phase subtask labels come from
``generate_foresight_nano_banana.EPISODE_PHASES`` so the label shown in the
mp4 is literally the same string the generator wrote into its manifest.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio.v3 as iio3
import numpy as np
from PIL import Image, ImageDraw

from experiments.subtask_probe.droid_eval.foreact_eval._io import (
    foresight_path,
    iter_source_frames,
)
from experiments.subtask_probe.droid_eval.foreact_eval.generate_foresight_nano_banana import (
    lookup_phase,
)
from experiments.subtask_probe.droid_eval.visualize_subtasks import (
    _composite_side_by_side,
    _draw_caption,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FRAME_W = 480
FRAME_H = 270
SEPARATOR_PX = 4


def _load_resized(path: Path, *, w: int, h: int) -> np.ndarray:
    with Image.open(path) as img:
        resized = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)


def _missing_placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = Image.new("RGB", (w, h), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    tw = draw.textlength(text)
    draw.text(((w - tw) / 2, h / 2 - 8), text, fill=(200, 200, 200))
    return np.asarray(img, dtype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render nano-banana chain as mp4")
    parser.add_argument(
        "--v2_root",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/foresight_chain_eggplant_v2"),
        help="Source 'actual' frames dir (episode_*/actual/frame_*.png).",
    )
    parser.add_argument(
        "--foresight_dir",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/foresight_nano_banana_chain"),
        help="Nano-banana foresight dir (episode_*/frame_*.png).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(
            ".experiments_cache/foreact_eval/foresight_nano_banana_chain/chain_nano_banana.mp4"
        ),
    )
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_frames = iter_source_frames(args.v2_root)
    phased = [(f, lookup_phase(f.episode_index, f.frame_idx)) for f in all_frames]
    in_phase = [(f, p) for (f, p) in phased if p is not None]
    if not in_phase:
        raise SystemExit(f"No in-phase frames found in {args.v2_root}")
    logger.info(
        "Rendering %d in-phase frames (trimmed %d) @ %d fps",
        len(in_phase),
        len(all_frames) - len(in_phase),
        args.fps,
    )

    composites: list[np.ndarray] = []
    for frame, phase in in_phase:
        actual = _load_resized(frame.actual_path, w=FRAME_W, h=FRAME_H)
        foresight_file = foresight_path(args.foresight_dir, frame.episode_index, frame.frame_idx)
        foresight = (
            _load_resized(foresight_file, w=FRAME_W, h=FRAME_H)
            if foresight_file.exists()
            else _missing_placeholder(FRAME_W, FRAME_H, "(no foresight)")
        )
        composite = _composite_side_by_side(
            {"actual": actual, "foresight": foresight}, separator_px=SEPARATOR_PX
        )
        footer = f"episode {frame.episode_index:03d}  \u00b7  frame {frame.frame_idx:05d}"
        composites.append(
            _draw_caption(
                composite,
                caption=phase.subtask_label,
                footer=footer,
                camera_labels=[
                    (4, "ACTUAL"),
                    (FRAME_W + SEPARATOR_PX + 4, "FORESIGHT"),
                ],
            )
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    iio3.imwrite(
        args.output_path,
        np.stack(composites),
        fps=args.fps,
        codec="libx264",
        macro_block_size=1,
    )
    logger.info("Wrote %s", args.output_path)


if __name__ == "__main__":
    main()
