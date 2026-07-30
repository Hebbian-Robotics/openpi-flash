#!/usr/bin/env python3
"""Phase 1 (alt): Generate subtask text for DROID frames via Gemini Robotics-ER.

Drop-in alternative to generate_subtasks.py that swaps the deployed pi0.5 server
for Google's Gemini Robotics-ER 1.6 Preview model, called directly via the
google-genai SDK. The on-disk JSON schema is identical so compare_subtask_outputs.py
and run_action_eval.py work unchanged.

Requires GEMINI_API_KEY in the environment (or in a .env file loaded via
python-dotenv, which is already a project dep).

Usage:
    uv run python experiments/subtask_probe/droid_eval/generate_subtasks_gemini.py \\
        --samples_dir ./.experiments_cache/droid_eval \\
        --output ./.experiments_cache/droid_eval/subtasks_gemini.json

    # Override the prompt template:
    uv run python experiments/subtask_probe/droid_eval/generate_subtasks_gemini.py \\
        --samples_dir ./.experiments_cache/droid_eval \\
        --prompt_format 'Task: {task}. What is the robot doing right now? Reply in 4 words.' \\
        --output ./.experiments_cache/droid_eval/subtasks_gemini_terse.json
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .comet_style._gemini_utils import call_with_retry, encode_png
from .utils import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

DEFAULT_PROMPT_FORMAT = (
    'You are observing a robot performing: "{task}". Looking at the exterior and '
    "wrist camera views of the current moment, describe the immediate next subtask "
    "the robot should do in 3 to 6 words, as a lowercase imperative phrase with no "
    "trailing period. Respond with only that phrase."
)


def _generate_one(
    client: genai.Client,
    model: str,
    prompt_text: str,
    exterior_png: bytes,
    wrist_png: bytes,
    thinking_budget: int,
    max_retries: int,
) -> tuple[str, float]:
    """Call Gemini with retry-on-429. Returns (subtask_text, elapsed_seconds)."""
    start_time = time.time()

    def _call() -> str:
        response = client.models.generate_content(
            model=model,
            contents=[
                "Exterior camera view:",
                types.Part.from_bytes(data=exterior_png, mime_type="image/png"),
                "Wrist camera view:",
                types.Part.from_bytes(data=wrist_png, mime_type="image/png"),
                prompt_text,
            ],
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        )
        return (response.text or "").strip()

    text = call_with_retry(_call, max_retries=max_retries)
    elapsed = time.time() - start_time
    return text, elapsed


def _validate_prompt_format(prompt_format: str) -> None:
    """Reject prompt formats that do not contain the required {task} placeholder.

    Mirrors the check in admin_server.py so CLI errors surface early.
    """
    if "{task}" not in prompt_format:
        raise ValueError(
            f"--prompt_format must contain the literal '{{task}}' placeholder, got: "
            f"{prompt_format!r}"
        )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate subtasks for DROID frames via Gemini Robotics-ER"
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        required=True,
        help="Directory with extracted DROID samples (from extract_droid_samples.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for subtask cache",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default=DEFAULT_PROMPT_FORMAT,
        help=(
            "Prompt template sent to Gemini. Must contain the literal '{task}' "
            "placeholder, which is replaced with the episode's language instruction."
        ),
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Gemini thinking budget (0 disables thinking, higher values allow more reasoning tokens)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help=(
            "Max concurrent Gemini API requests. Default 1 keeps you under the "
            "free-tier 5 RPM cap; raise it on a paid tier for throughput."
        ),
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Per-frame retry budget for 429 (RESOURCE_EXHAUSTED) responses",
    )
    args = parser.parse_args()

    _validate_prompt_format(args.prompt_format)

    samples_dir = Path(args.samples_dir)
    manifest = load_manifest(samples_dir)
    logger.info("Loaded manifest: %d episodes", len(manifest))

    client = genai.Client()

    # Build the flat task list up-front so results can be collected in manifest order
    # regardless of which worker finishes first.
    tasks: list[dict[str, Any]] = []
    for episode in manifest:
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]
        for frame_info in episode["frames"]:
            tasks.append(
                {
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "frame_idx": frame_info["frame_idx"],
                    "frame_path": samples_dir / frame_info["file"],
                }
            )
    total_frames = len(tasks)
    logger.info(
        "Dispatching %d frames to %s (max_workers=%d)", total_frames, args.model, args.max_workers
    )

    progress_lock = threading.Lock()
    progress = {"done": 0}

    def process(task: dict[str, Any]) -> dict[str, Any]:
        frame_data = np.load(task["frame_path"])
        prompt_text = args.prompt_format.format(task=task["instruction"])
        try:
            exterior_png = encode_png(frame_data["exterior_image"])
            wrist_png = encode_png(frame_data["wrist_image"])
            subtask_text, elapsed = _generate_one(
                client=client,
                model=args.model,
                prompt_text=prompt_text,
                exterior_png=exterior_png,
                wrist_png=wrist_png,
                thinking_budget=args.thinking_budget,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            logger.warning(
                "Gemini call failed for %s frame %d: %s",
                task["episode_id"],
                task["frame_idx"],
                exc,
            )
            subtask_text = ""
            elapsed = 0.0

        with progress_lock:
            progress["done"] += 1
            done = progress["done"]
        if done % 5 == 0 or done == total_frames:
            logger.info(
                "[%d/%d] %s frame %d: %r (%.1fs)",
                done,
                total_frames,
                task["episode_id"],
                task["frame_idx"],
                subtask_text,
                elapsed,
            )

        return {
            "episode_id": task["episode_id"],
            "frame_idx": task["frame_idx"],
            "instruction": task["instruction"],
            "subtask_text": subtask_text,
            "generation_time_s": round(elapsed, 2),
            "server_subtask_ms": round(elapsed * 1000, 1),
        }

    if args.max_workers <= 1:
        subtask_results = [process(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # executor.map preserves input order, which matches the manifest.
            subtask_results = list(executor.map(process, tasks))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "prompt_format": args.prompt_format,
                "backend": args.model,
                "results": subtask_results,
            },
            f,
            indent=2,
        )

    logger.info(
        "Subtask generation complete: %d results saved to %s (backend=%s)",
        len(subtask_results),
        output_path,
        args.model,
    )

    # Summary stats — mirrors generate_subtasks.py so output looks familiar.
    gen_times = [r["generation_time_s"] for r in subtask_results if r["generation_time_s"] > 0]
    if gen_times:
        logger.info(
            "Latency: mean=%.2fs, min=%.2fs, max=%.2fs",
            float(np.mean(gen_times)),
            float(np.min(gen_times)),
            float(np.max(gen_times)),
        )
    unique_subtasks = {r["subtask_text"] for r in subtask_results if r["subtask_text"]}
    logger.info(
        "Unique subtask texts: %d out of %d frames", len(unique_subtasks), len(subtask_results)
    )
    failed = sum(1 for r in subtask_results if not r["subtask_text"])
    if failed:
        logger.warning("Failed frames: %d / %d", failed, len(subtask_results))


if __name__ == "__main__":
    main()
