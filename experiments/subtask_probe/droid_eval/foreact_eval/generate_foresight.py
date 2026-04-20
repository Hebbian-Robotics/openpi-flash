#!/usr/bin/env python3
"""Run the ForeAct foresight image generator across DROID cache frames.

IMPORTANT: this driver is designed to run **inside the foreact conda env on a
remote GPU box**, not inside our hosting project's Python env. It imports
``pipeline.VisualForesightPipeline`` and ``utils.trainer_utils`` from the
``/Users/kkuan/openpi/foreact/`` repo — those modules live in the foreact
env, not ours, which is why our linter ignores the imports.

Typical workflow:

1. On the remote box, clone the foreact repo and run its ``environment_setup.sh``.
2. ``huggingface-cli download mit-han-lab/foreact-pretrained --local-dir ~/foreact_ckpt``.
3. Copy our DROID cache + subtasks JSON to the box.
4. ``scp`` this script into the foreact repo directory.
5. ``conda activate foreact && python generate_foresight.py ...``.

Uses the paper's recommended inference hparams (``foreact/app_cli.py``):
``guidance_scale=4.5``, ``image_guidance_scale=1.5``, ``num_inference_steps=8``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# These imports resolve inside the foreact conda env, not ours. This script
# is meant to be run on the remote GPU box after ``conda activate foreact``.
from pipeline import VisualForesightPipeline  # ty: ignore[unresolved-import]
from utils.trainer_utils import find_newest_checkpoint  # ty: ignore[unresolved-import]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_manifest(samples_dir: Path) -> list[dict[str, Any]]:
    with (samples_dir / "manifest.json").open() as f:
        return json.load(f)


def _load_subtask_records(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognized subtask JSON shape in {path}")


def _index_subtasks(records: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    return {(r["episode_id"], r["frame_idx"]): r["subtask_text"] for r in records}


def _process_frame(
    pipeline: VisualForesightPipeline,
    exterior_image: np.ndarray,
    subtask_text: str,
    *,
    guidance_scale: float,
    image_guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> tuple[Image.Image, float]:
    """Generate one foresight image and return (image, elapsed_seconds)."""
    pil_in = Image.fromarray(exterior_image).convert("RGB")
    generator = torch.Generator().manual_seed(seed)
    start = time.time()
    out = pipeline(
        caption=subtask_text,
        image=pil_in,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
        generator=generator,
    ).images
    elapsed = time.time() - start
    if not out:
        raise RuntimeError("VisualForesightPipeline returned no images")
    return out[0], elapsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForeAct foresight generator over DROID cache frames"
    )
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument(
        "--subtasks",
        type=str,
        required=True,
        help="Path to a subtasks_*.json (e.g. from foreact_eval.generate_subtasks).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write foresight PNGs and foresight_manifest.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local path to the foreact-pretrained checkpoint directory.",
    )
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Only process the first N episodes (for smoke tests).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(samples_dir)
    if args.max_episodes is not None:
        manifest = manifest[: args.max_episodes]
    logger.info("Loaded manifest: %d episodes", len(manifest))

    records = _load_subtask_records(Path(args.subtasks))
    subtask_index = _index_subtasks(records)
    logger.info("Loaded %d subtask records", len(records))

    logger.info("Loading VisualForesightPipeline from %s ...", args.checkpoint)
    pipeline = VisualForesightPipeline.from_pretrained(
        find_newest_checkpoint(args.checkpoint),
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    logger.info("Pipeline loaded.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    generation_records: list[dict[str, Any]] = []
    for episode in manifest:
        episode_id = episode["episode_id"]
        episode_dir = output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting episode %s (%d frames)", episode_id, len(episode["frames"]))

        for step_idx, frame_info in enumerate(episode["frames"]):
            frame_idx = frame_info["frame_idx"]
            subtask_text = subtask_index.get((episode_id, frame_idx), "")
            if not subtask_text:
                logger.warning("No subtask for %s frame %d; skipping", episode_id, frame_idx)
                continue

            frame_path = samples_dir / frame_info["file"]
            frame_data = np.load(frame_path)
            exterior_image = np.asarray(frame_data["exterior_image"], dtype=np.uint8)

            # Per-frame seed so the generation is reproducible but each frame
            # has independent noise.
            frame_seed = (args.seed, episode_id, frame_idx).__hash__() & 0x7FFFFFFF
            try:
                image, elapsed = _process_frame(
                    pipeline,
                    exterior_image,
                    subtask_text,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=frame_seed,
                )
            except Exception as exc:
                logger.warning(
                    "Foresight generation failed for %s frame %d: %s",
                    episode_id,
                    frame_idx,
                    exc,
                )
                continue

            out_path = episode_dir / f"frame_{frame_idx:05d}.png"
            image.save(out_path)
            generation_records.append(
                {
                    "episode_id": episode_id,
                    "frame_idx": frame_idx,
                    "subtask_text": subtask_text,
                    "output": str(out_path.relative_to(output_dir)),
                    "generation_time_s": round(elapsed, 3),
                    "seed": frame_seed,
                }
            )

            if step_idx % 5 == 0 or step_idx == len(episode["frames"]) - 1:
                logger.info(
                    "[%s] frame %d/%d: %r (%.2fs)",
                    episode_id,
                    step_idx + 1,
                    len(episode["frames"]),
                    subtask_text,
                    elapsed,
                )

    manifest_path = output_dir / "foresight_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "hparams": {
                    "guidance_scale": args.guidance_scale,
                    "image_guidance_scale": args.image_guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "seed": args.seed,
                    "checkpoint": args.checkpoint,
                },
                "records": generation_records,
            },
            f,
            indent=2,
        )
    logger.info(
        "Wrote %d foresight images + manifest to %s",
        len(generation_records),
        output_dir,
    )

    latencies = [r["generation_time_s"] for r in generation_records]
    if latencies:
        logger.info(
            "Foresight latency: mean=%.2fs min=%.2fs max=%.2fs (n=%d)",
            float(np.mean(latencies)),
            float(np.min(latencies)),
            float(np.max(latencies)),
            len(latencies),
        )


if __name__ == "__main__":
    main()
