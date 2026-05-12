#!/usr/bin/env python3
"""Phase 2: Run action generation under 2 prompt conditions via deployed server.

Sends each cached DROID frame to the deployed server with mode="action_only"
under two prompt conditions:
  1. Baseline:  original task instruction only
  2. Subtask:   "{instruction}. Subtask: {subtask}" (with generated subtask)

Requires the server to be running with pi05_droid config and the DROID
checkpoint, since the action policy's transforms and normalization must
match the DROID embodiment.

Usage:
    uv run python experiments/subtask_probe/droid_eval/run_action_eval.py \
        --samples_dir ./.experiments_cache/droid_eval \
        --subtasks ./.experiments_cache/droid_eval/subtasks.json \
        --output_dir ./.experiments_cache/droid_eval/predictions \
        --server 43.200.36.250
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from hosting.flash_transport_policy import FlashTransportPolicy

from .constants import DEFAULT_QUIC_PORT, DROID_ACTION_DIM
from .utils import (
    build_action_observation,
    build_warmup_observation,
    generate_frame_noise,
    load_manifest,
    load_subtask_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run action eval with 2 prompt conditions")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--subtasks", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--server",
        type=str,
        required=True,
        help="Server address (e.g., 43.200.36.250)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_QUIC_PORT, help="Server QUIC port")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(samples_dir)
    subtask_index = load_subtask_index(Path(args.subtasks))

    # Connect to server
    policy = FlashTransportPolicy(args.server, port=args.port)
    logger.info("Connected to server at %s:%d via QUIC", args.server, args.port)

    policy.infer(build_warmup_observation(mode="action_only"))
    logger.info("Server warmup complete")

    # Process all frames
    all_predictions = []
    total_frames = sum(ep["num_frames"] for ep in manifest)
    processed = 0

    for episode in manifest:
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]

        for frame_info in episode["frames"]:
            frame_idx = frame_info["frame_idx"]
            frame_file = samples_dir / frame_info["file"]
            frame_data = np.load(frame_file)

            exterior_image = frame_data["exterior_image"]
            wrist_image = frame_data["wrist_image"]
            raw_state = frame_data["state"]

            subtask_text = subtask_index.get((episode_id, frame_idx), "")
            if not subtask_text:
                logger.warning(
                    "No subtask found for %s frame %d, using empty string", episode_id, frame_idx
                )

            # Same noise for all conditions so the only difference is the prompt
            frame_noise = generate_frame_noise(episode_id, frame_idx)

            # Run all conditions
            conditions = {
                "baseline": instruction,
                "subtask": f"{instruction}. Subtask: {subtask_text}",
            }
            frame_predictions = {}
            for condition_name, prompt in conditions.items():
                obs = build_action_observation(
                    exterior_image, wrist_image, raw_state, prompt, noise=frame_noise
                )
                result = policy.infer(obs)
                frame_predictions[condition_name] = np.array(result["actions"])[
                    :, :DROID_ACTION_DIM
                ]

            # Save predictions
            pred_file = output_dir / f"{episode_id}_frame_{frame_idx:05d}.npz"
            np.savez_compressed(pred_file, **frame_predictions)

            all_predictions.append(
                {
                    "episode_id": episode_id,
                    "frame_idx": frame_idx,
                    "instruction": instruction,
                    "subtask_text": subtask_text,
                    "prediction_file": str(pred_file.relative_to(output_dir)),
                }
            )

            processed += 1
            if processed % 5 == 0 or processed == total_frames:
                logger.info(
                    "[%d/%d] Processed %s frame %d", processed, total_frames, episode_id, frame_idx
                )

    # Save prediction manifest
    pred_manifest_path = output_dir / "prediction_manifest.json"
    with pred_manifest_path.open("w") as f:
        json.dump(all_predictions, f, indent=2)

    logger.info(
        "Action evaluation complete: %d predictions saved to %s", len(all_predictions), output_dir
    )


if __name__ == "__main__":
    main()
