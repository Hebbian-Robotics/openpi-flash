#!/usr/bin/env python3
"""Run the ForeAct planner (Qwen3-VL) over the v2 eggplant chain.

Walks the v2 chain's ``episode_*/actual/frame_*.png`` frames in temporal
order, keeps the planner stateful across frames (no reset between linked
episodes since they form one continuous task), and writes per-frame subtask
predictions to JSON.

This is the "use the paper's VLM correctly" companion to
``generate_foresight_nano_banana.py`` — we've been hardcoding subtask
labels in ``CHAIN_PHASES`` for both foresight generators; this script
replaces that with what a real Qwen3-VL-8B planner would say frame by
frame.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.subtask_probe.droid_eval.foreact_eval._io import iter_source_frames
from experiments.subtask_probe.droid_eval.foreact_eval.planner import OpenAICompatPlanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_TASK = "Pick up the eggplant and place it into the blue plate."
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL planner over the v2 chain")
    parser.add_argument(
        "--v2_root",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/foresight_chain_eggplant_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/subtasks_qwen3_v2chain.json"),
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--no_schema",
        action="store_true",
        help="Disable JSON schema enforcement on the VLM response (closer to Table 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frames = iter_source_frames(args.v2_root)
    logger.info(
        "Running Qwen3-VL planner over %d frames (task=%r, base_url=%s)",
        len(frames),
        args.task,
        args.base_url,
    )

    planner = OpenAICompatPlanner(
        base_url=args.base_url, model=args.model, use_schema=not args.no_schema
    )
    logger.info("use_schema=%s", not args.no_schema)

    records: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        image = _load_rgb(frame.actual_path)
        try:
            result = planner.generate_subtask(args.task, image)
        except Exception as exc:
            logger.warning("ep%d f%05d failed: %s", frame.episode_index, frame.frame_idx, exc)
            result = {
                "subtask": "",
                "previous_finished": False,
                "prompt_phase": "error",
                "error": str(exc),
            }
        records.append(
            {
                "episode_index": frame.episode_index,
                "frame_idx": frame.frame_idx,
                **result,
            }
        )
        logger.info(
            "[%d/%d] ep%d f%05d (%s): previous_finished=%s subtask=%r",
            i + 1,
            len(frames),
            frame.episode_index,
            frame.frame_idx,
            result.get("prompt_phase", "?"),
            result.get("previous_finished"),
            result.get("subtask"),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "task": args.task,
                "model": args.model,
                "base_url": args.base_url,
                "results": records,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
