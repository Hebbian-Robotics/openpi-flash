#!/usr/bin/env python3
"""Run the ForeAct foresight generator across a LeRobot-format dataset episode.

Same generator, same hparams as ``generate_foresight.py``, but reads input
frames from LeRobot-v2.1 shards (mp4 videos + parquet) instead of our DROID
.npz cache. Used to test the pretrained generator on a dataset it was
*pretrained on* (Galaxea R1 Lite, `mit-han-lab/ForeActDataset`).

This script runs inside the foreact conda env on the remote GPU box:

    ssh us-west-2 "cd ~/foreact && source .venv/bin/activate && \\
        python generate_foresight_lerobot.py \\
            --dataset_root ~/foreact_dataset/20251102_Pick_Veg \\
            --camera_key observation.images.head_left_rgb \\
            --output_dir ~/foresight_foreact_picksveg \\
            --checkpoint ~/foreact_ckpt \\
            --episode_indices 0,1,2,3,4 \\
            --stride 15"
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image

# Resolved inside the foreact conda env on the remote box.
from pipeline import VisualForesightPipeline  # ty: ignore[unresolved-import]
from utils.trainer_utils import find_newest_checkpoint  # ty: ignore[unresolved-import]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_episode_index(dataset_root: Path) -> dict[int, dict[str, Any]]:
    """Parse meta/episodes.jsonl into {episode_index: {tasks: [...], length: int}}."""
    by_idx: dict[int, dict[str, Any]] = {}
    with (dataset_root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            by_idx[row["episode_index"]] = row
    return by_idx


def _decode_mp4_frames(video_path: Path, stride: int) -> list[np.ndarray]:
    """Return every `stride`-th frame from an mp4 as HxWx3 uint8 RGB arrays.

    LeRobot stores each camera as a single mp4 per episode; we decode all
    frames sequentially and pick out the strided ones. For Galaxea R1 Lite at
    15 fps, stride=15 gives ~1 Hz — matches the paper's pretraining sampling
    cadence (§3.2: "sample condition frames at 1-second intervals").
    """
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if idx % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
    return frames


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


def _parse_episode_indices(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForeAct foresight over a LeRobot dataset")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to a LeRobot-v2.1 dataset dir containing meta/, data/, videos/.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="observation.images.head_left_rgb",
        help="Camera feature name. Default matches foreact/configs/finetune.yaml.",
    )
    parser.add_argument(
        "--episode_indices",
        type=str,
        default="0",
        help="Comma-separated episode indices to process (e.g. '0,1,2,3,4').",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Decode every Nth frame (15 Hz dataset / stride=15 = ~1 Hz).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_meta = _load_episode_index(dataset_root)
    episode_indices = _parse_episode_indices(args.episode_indices)
    for idx in episode_indices:
        if idx not in episode_meta:
            raise SystemExit(f"episode {idx} not in meta/episodes.jsonl")
    logger.info(
        "Dataset root=%s, processing %d episodes: %s",
        dataset_root,
        len(episode_indices),
        episode_indices,
    )

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
    for ep_idx in episode_indices:
        meta = episode_meta[ep_idx]
        tasks: list[str] = meta.get("tasks") or []
        # Skip episodes whose label is the "Finish." filler used as a between-
        # task reset signal (task_index=1 in tasks.jsonl).
        if not tasks or tasks[0].strip().lower().rstrip(".") == "finish":
            logger.info("Skipping episode %d (task=%r)", ep_idx, tasks)
            continue
        subtask_text = tasks[0]
        video_path = (
            dataset_root / "videos" / "chunk-000" / args.camera_key / f"episode_{ep_idx:06d}.mp4"
        )
        if not video_path.exists():
            logger.warning("missing video: %s", video_path)
            continue

        logger.info("Decoding %s ...", video_path)
        frames = _decode_mp4_frames(video_path, stride=args.stride)
        logger.info(
            "Episode %d: %d strided frames (length=%d) — subtask=%r",
            ep_idx,
            len(frames),
            meta.get("length"),
            subtask_text,
        )

        episode_dir = output_dir / f"episode_{ep_idx:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Also save the actual source frames so the downstream HTML / eyeball
        # comparison doesn't need to re-decode the mp4 on the local box.
        src_dir = episode_dir / "actual"
        src_dir.mkdir(parents=True, exist_ok=True)

        for step_idx, frame_rgb in enumerate(frames):
            frame_idx = step_idx * args.stride
            Image.fromarray(frame_rgb).save(src_dir / f"frame_{frame_idx:05d}.png")
            seed = (args.seed, ep_idx, frame_idx).__hash__() & 0x7FFFFFFF
            try:
                image, elapsed = _process_frame(
                    pipeline,
                    frame_rgb,
                    subtask_text,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=seed,
                )
            except Exception as exc:
                logger.warning("episode %d frame %d failed: %s", ep_idx, frame_idx, exc)
                continue

            out_path = episode_dir / f"frame_{frame_idx:05d}.png"
            image.save(out_path)
            generation_records.append(
                {
                    "episode_index": ep_idx,
                    "frame_idx": frame_idx,
                    "subtask_text": subtask_text,
                    "output": str(out_path.relative_to(output_dir)),
                    "generation_time_s": round(elapsed, 3),
                    "seed": seed,
                }
            )

            if step_idx % 3 == 0 or step_idx == len(frames) - 1:
                logger.info(
                    "[ep%d] step %d/%d (frame %d): %.2fs",
                    ep_idx,
                    step_idx + 1,
                    len(frames),
                    frame_idx,
                    elapsed,
                )

    manifest_path = output_dir / "foresight_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "dataset_root": str(dataset_root),
                "camera_key": args.camera_key,
                "stride": args.stride,
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
    logger.info("Wrote %d foresight images to %s", len(generation_records), output_dir)

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
