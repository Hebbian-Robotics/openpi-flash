#!/usr/bin/env python3
"""Phase 0: Extract DROID samples from RLDS for evaluation.

Streams episodes from gs://gresearch/robotics/droid/1.0.1, extracts frames
with images, proprioceptive state, language instructions, and ground truth
action chunks. Caches to local .npz files.

Two selection modes:

* **First-K** (default): take the first ``--num_episodes`` matches in stream
  order. Fast and useful when the dataset is already well-shaped.

* **Top-K longest** (``--scan_episodes N``): scan the first N episodes in the
  stream, buffer qualifiers in an in-memory min-heap, save the
  ``--num_episodes`` longest. Use for long-horizon evals where the raw stream
  is dominated by short demos. Works well paired with ``--require_multi_step``.

Usage:
    # First-K (legacy behavior):
    uv run python -m experiments.subtask_probe.droid_eval.extract_droid_samples \\
        --num_episodes 10 \\
        --output_dir ./.experiments_cache/droid_eval

    # Top-K longest with multi-step filter (long-horizon eval):
    uv run python -m experiments.subtask_probe.droid_eval.extract_droid_samples \\
        --num_episodes 5 --scan_episodes 5000 \\
        --min_duration_s 60 --require_multi_step \\
        --output_dir ./.experiments_cache/droid_eval_2min
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from .constants import ACTION_HORIZON
from .utils import decode_droid_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# DROID is recorded at 15 Hz — used to convert --min_duration_s to a step count.
DROID_FPS = 15

# Multi-step task keywords — tasks containing these are likely long-horizon
MULTI_STEP_KEYWORDS = re.compile(
    r"\b(and then|then|after|pick.{1,20}place|put.{1,20}in|open.{1,20}put|grab.{1,20}move)\b",
    re.IGNORECASE,
)


def is_multi_step_task(instruction: str) -> bool:
    """Heuristic: does the instruction describe a multi-step task?"""
    return bool(MULTI_STEP_KEYWORDS.search(instruction))


def _decode_str(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _extract_traj_fields(traj: dict[str, Any]) -> dict[str, Any] | None:
    """Pull metadata + arrays from a raw numpy traj.

    Returns None if the episode is unusable (empty instruction, too short).
    """
    file_path = _decode_str(traj["traj_metadata"]["episode_metadata"]["file_path"][0])
    instruction = _decode_str(traj["language_instruction"][0]).strip()
    if not instruction or instruction.lower() in ("nan", "none"):
        logger.debug("Skipping episode with empty instruction: %s", file_path)
        return None

    actions_joint = traj["action_dict"]["joint_position"]  # [T, 7]
    actions_gripper = traj["action_dict"]["gripper_position"]  # [T, 1]
    actions = np.concatenate([actions_joint, actions_gripper], axis=-1)  # [T, 8]

    traj_len = len(actions)
    if traj_len < ACTION_HORIZON:
        logger.debug("Skipping short episode (%d frames): %s", traj_len, file_path)
        return None

    return {
        "file_path": file_path,
        "instruction": instruction,
        "traj_len": int(traj_len),
        "actions": actions,
        "exterior_images": traj["observation"]["exterior_image_1_left"],
        "wrist_images": traj["observation"]["wrist_image_left"],
        "joint_positions": traj["observation"]["joint_position"],
        "gripper_positions": traj["observation"]["gripper_position"],
    }


def _passes_filters(
    fields: dict[str, Any], min_duration_s: float, require_multi_step: bool
) -> bool:
    min_traj_len = int(min_duration_s * DROID_FPS)
    if fields["traj_len"] < min_traj_len:
        return False
    return not (require_multi_step and not is_multi_step_task(fields["instruction"]))


def _save_episode(
    fields: dict[str, Any],
    episode_id: str,
    output_dir: Path,
    frame_subsample: int,
) -> dict[str, Any]:
    """Decode images + save npz frames for one episode. Returns the manifest entry."""
    episode_dir = output_dir / episode_id
    episode_dir.mkdir(exist_ok=True)

    traj_len = fields["traj_len"]
    actions = fields["actions"]
    frame_indices = list(range(0, traj_len, frame_subsample))

    frame_records = []
    for frame_idx in frame_indices:
        # Decode images from JPEG bytes (kept at native DROID res; server pads to 224x224).
        exterior_img = decode_droid_image(fields["exterior_images"][frame_idx])
        wrist_img = decode_droid_image(fields["wrist_images"][frame_idx])

        # Ground truth action chunk: next ACTION_HORIZON actions, pad tail with last.
        action_chunk_indices = np.minimum(
            np.arange(frame_idx, frame_idx + ACTION_HORIZON), traj_len - 1
        )
        ground_truth_action_chunk = actions[action_chunk_indices]  # [15, 8]

        state = np.concatenate(
            [fields["joint_positions"][frame_idx], fields["gripper_positions"][frame_idx]]
        )  # [8]

        frame_file = episode_dir / f"frame_{frame_idx:05d}.npz"
        np.savez_compressed(
            frame_file,
            exterior_image=exterior_img,
            wrist_image=wrist_img,
            state=state,
            ground_truth_actions=ground_truth_action_chunk,
            frame_idx=frame_idx,
        )

        frame_records.append(
            {"frame_idx": int(frame_idx), "file": str(frame_file.relative_to(output_dir))}
        )

    return {
        "episode_id": episode_id,
        "instruction": fields["instruction"],
        "task_type": "multi_step" if is_multi_step_task(fields["instruction"]) else "single_step",
        "file_path": fields["file_path"],
        "traj_len": traj_len,
        "num_frames": len(frame_records),
        "frames": frame_records,
    }


def _write_manifest(manifest: list[dict[str, Any]], output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    multi_step_count = sum(1 for ep in manifest if ep["task_type"] == "multi_step")
    single_step_count = sum(1 for ep in manifest if ep["task_type"] == "single_step")
    total_frames = sum(ep["num_frames"] for ep in manifest)
    logger.info("Extraction complete:")
    logger.info(
        "  Episodes: %d (%d multi-step, %d single-step)",
        len(manifest),
        multi_step_count,
        single_step_count,
    )
    logger.info("  Total frames: %d", total_frames)
    logger.info("  Manifest: %s", manifest_path)


def _extract_first_k(
    dataset: Any,
    num_episodes: int,
    output_dir: Path,
    frame_subsample: int,
    min_duration_s: float,
    require_multi_step: bool,
) -> None:
    """Stream-scan mode: save the first ``num_episodes`` qualifying episodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for traj in tqdm.tqdm(dataset.as_numpy_iterator(), desc="Extracting", total=num_episodes):
        if len(manifest) >= num_episodes:
            break
        fields = _extract_traj_fields(traj)
        if fields is None or not _passes_filters(fields, min_duration_s, require_multi_step):
            continue
        entry = _save_episode(fields, f"ep_{len(manifest):04d}", output_dir, frame_subsample)
        manifest.append(entry)
        logger.info(
            "Episode %s (%d steps / %.1fs, %s): %r",
            entry["episode_id"],
            entry["traj_len"],
            entry["traj_len"] / DROID_FPS,
            entry["task_type"],
            entry["instruction"][:80],
        )

    _write_manifest(manifest, output_dir)


def _extract_top_k(
    dataset: Any,
    num_episodes: int,
    scan_episodes: int,
    output_dir: Path,
    frame_subsample: int,
    min_duration_s: float,
    require_multi_step: bool,
) -> None:
    """Top-K longest mode: scan ``scan_episodes`` and keep the longest ``num_episodes`` matches.

    Uses a size-``num_episodes`` min-heap keyed on traj_len so memory stays bounded
    (roughly ``num_episodes`` x episode_size in RAM). Episode size is ~20-40 MB for
    a 1-2 minute DROID episode, so keeping 5 in memory is ~150 MB — fine.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # heap: (traj_len, insertion_index, fields_dict). insertion_index is a tiebreaker
    # so heapq never tries to compare raw dicts when traj_len matches.
    heap: list[tuple[int, int, dict[str, Any]]] = []
    scanned = 0
    qualifying = 0

    for i, traj in enumerate(
        tqdm.tqdm(dataset.as_numpy_iterator(), desc="Scanning", total=scan_episodes)
    ):
        if i >= scan_episodes:
            break
        scanned = i + 1
        fields = _extract_traj_fields(traj)
        if fields is None or not _passes_filters(fields, min_duration_s, require_multi_step):
            continue
        qualifying += 1
        key = (fields["traj_len"], i, fields)
        if len(heap) < num_episodes:
            heapq.heappush(heap, key)
        elif fields["traj_len"] > heap[0][0]:
            heapq.heapreplace(heap, key)

    logger.info(
        "Scan complete: %d episodes read, %d passed filters, keeping top %d by duration",
        scanned,
        qualifying,
        len(heap),
    )

    # Sort longest-first so ep_0000 is the longest — easier to eyeball the manifest.
    selected = sorted(heap, key=lambda entry: -entry[0])

    manifest: list[dict[str, Any]] = []
    for episode_count, (_, _, fields) in enumerate(selected):
        entry = _save_episode(fields, f"ep_{episode_count:04d}", output_dir, frame_subsample)
        manifest.append(entry)
        logger.info(
            "Saved %s (%d steps / %.1fs, %s): %r",
            entry["episode_id"],
            entry["traj_len"],
            entry["traj_len"] / DROID_FPS,
            entry["task_type"],
            entry["instruction"][:80],
        )

    _write_manifest(manifest, output_dir)


def extract_episodes(
    data_dir: str,
    num_episodes: int,
    output_dir: Path,
    frame_subsample: int = 10,
    min_duration_s: float = 0.0,
    require_multi_step: bool = False,
    scan_episodes: int | None = None,
) -> None:
    """Stream DROID episodes from GCS and cache selected frames."""
    # Lazy imports — tensorflow is heavy and optional
    import dlimp as dl  # ty: ignore[unresolved-import]
    import tensorflow as tf  # ty: ignore[unresolved-import]
    import tensorflow_datasets as tfds  # ty: ignore[unresolved-import]

    # Prevent TF from grabbing GPU
    tf.config.set_visible_devices([], "GPU")

    logger.info("Building DROID RLDS dataset from %s ...", data_dir)
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=4)

    # Filter for successful episodes only
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0],
            ".*success.*",
        )
    )

    if scan_episodes is None:
        _extract_first_k(
            dataset=dataset,
            num_episodes=num_episodes,
            output_dir=output_dir,
            frame_subsample=frame_subsample,
            min_duration_s=min_duration_s,
            require_multi_step=require_multi_step,
        )
    else:
        _extract_top_k(
            dataset=dataset,
            num_episodes=num_episodes,
            scan_episodes=scan_episodes,
            output_dir=output_dir,
            frame_subsample=frame_subsample,
            min_duration_s=min_duration_s,
            require_multi_step=require_multi_step,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DROID samples for evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gs://gresearch/robotics",
        help=(
            "GCS path that contains the `droid/` TFDS tree. The public DROID v1.0.1 "
            "lives at gs://gresearch/robotics/droid/1.0.1, so the right data_dir is "
            "`gs://gresearch/robotics`."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to save (start small to validate pipeline)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./.experiments_cache/droid_eval",
        help="Local directory to cache extracted frames",
    )
    parser.add_argument(
        "--frame_subsample",
        type=int,
        default=10,
        help="Take every Nth frame from each episode",
    )
    parser.add_argument(
        "--min_duration_s",
        type=float,
        default=0.0,
        help=(
            "Minimum episode duration in seconds. DROID runs at 15 Hz, so "
            "e.g. 60 keeps only episodes >= 1:00. Default 0 keeps all."
        ),
    )
    parser.add_argument(
        "--require_multi_step",
        action="store_true",
        help=(
            "Keep only episodes whose language instruction matches the multi-step "
            "keyword heuristic (and-then, pick…place, put…in, etc.)."
        ),
    )
    parser.add_argument(
        "--scan_episodes",
        type=int,
        default=None,
        help=(
            "If set, enable top-K longest mode: scan this many stream episodes, buffer "
            "qualifying ones in a min-heap, save the --num_episodes longest. If omitted, "
            "use legacy first-K streaming."
        ),
    )
    args = parser.parse_args()

    extract_episodes(
        data_dir=args.data_dir,
        num_episodes=args.num_episodes,
        output_dir=Path(args.output_dir),
        frame_subsample=args.frame_subsample,
        min_duration_s=args.min_duration_s,
        require_multi_step=args.require_multi_step,
        scan_episodes=args.scan_episodes,
    )


if __name__ == "__main__":
    main()
