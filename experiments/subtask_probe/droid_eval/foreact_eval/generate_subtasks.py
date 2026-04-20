#!/usr/bin/env python3
"""Run the ForeAct VLM planner across DROID cache frames.

Produces ``subtasks_foreact_*.json`` with the same per-frame schema other
subtask generators in this project emit, so ``visualize_foreact.py`` and any
future tooling consume it uniformly.

Usage (OpenAI-compatible backend, paper's Qwen3-VL-8B-Instruct on local vLLM)::

    uv run python -m experiments.subtask_probe.droid_eval.foreact_eval.generate_subtasks \\
        --samples_dir ./.experiments_cache/droid_eval_ah15 \\
        --output ./.experiments_cache/droid_eval_ah15/subtasks_foreact_qwen8b.json \\
        --backend openai_compat \\
        --base_url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal, assert_never, cast, get_args

import numpy as np
from dotenv import load_dotenv

from experiments.subtask_probe.droid_eval.foreact_eval.planner import (
    DEFAULT_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    BasePlanner,
    GeminiPlanner,
    OpenAICompatPlanner,
)
from experiments.subtask_probe.droid_eval.utils import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


Backend = Literal["openai_compat", "gemini"]
BACKEND_CHOICES: tuple[Backend, ...] = get_args(Backend)


def _parse_backend(raw: str) -> Backend:
    if raw in BACKEND_CHOICES:
        return cast(Backend, raw)
    raise ValueError(f"Unknown backend: {raw!r} (expected one of {BACKEND_CHOICES})")


def _build_planner(backend: Backend, args: argparse.Namespace) -> tuple[BasePlanner, str]:
    """Return (planner, backend_label) where backend_label goes into the output JSON."""
    match backend:
        case "openai_compat":
            base_url = args.base_url or DEFAULT_BASE_URL
            model = args.model or DEFAULT_OPENAI_MODEL
            planner: BasePlanner = OpenAICompatPlanner(
                base_url=base_url,
                model=model,
                api_key=args.api_key,
            )
            return planner, f"{model}@{base_url}"
        case "gemini":
            model = args.model or DEFAULT_GEMINI_MODEL
            planner = GeminiPlanner(
                model=model,
                thinking_budget=args.thinking_budget,
                max_retries=args.max_retries,
            )
            return planner, model
        case _ as unreachable:
            assert_never(unreachable)


def _process_episode(
    planner: BasePlanner,
    samples_dir: Path,
    episode: dict[str, Any],
    replan_every: int,
) -> list[dict[str, Any]]:
    """Run the planner across one episode, reset()-ing state at the start.

    On replan frames, one VLM call is issued. On non-replan frames, the last
    subtask is reused without a VLM call. Non-replan latency is recorded as 0s
    so the output JSON clearly distinguishes replan frames from reused frames.
    """
    planner.reset()
    records: list[dict[str, Any]] = []
    episode_id = episode["episode_id"]
    instruction = episode["instruction"]
    last_subtask = ""
    last_previous_finished = False
    last_prompt_phase = ""

    for step_idx, frame_info in enumerate(episode["frames"]):
        frame_idx = frame_info["frame_idx"]
        frame_path = samples_dir / frame_info["file"]
        frame_data = np.load(frame_path)
        exterior_image = np.asarray(frame_data["exterior_image"], dtype=np.uint8)

        is_replan = step_idx % replan_every == 0
        elapsed = 0.0
        if is_replan:
            start = time.time()
            try:
                result = planner.generate_subtask(instruction, exterior_image)
                last_subtask = result["subtask"]
                last_previous_finished = result["previous_finished"]
                last_prompt_phase = result["prompt_phase"]
            except Exception as exc:
                logger.warning(
                    "Planner call failed for %s frame %d: %s",
                    episode_id,
                    frame_idx,
                    exc,
                )
                last_subtask = ""
                last_previous_finished = False
                last_prompt_phase = "error"
            elapsed = time.time() - start

        records.append(
            {
                "episode_id": episode_id,
                "frame_idx": frame_idx,
                "instruction": instruction,
                "subtask_text": last_subtask,
                "generation_time_s": round(elapsed, 2),
                "server_subtask_ms": round(elapsed * 1000, 1),
                "previous_finished": last_previous_finished,
                "prompt_phase": last_prompt_phase,
                "is_replan": is_replan,
            }
        )

        if step_idx % 5 == 0 or step_idx == len(episode["frames"]) - 1 or is_replan:
            logger.info(
                "[%s] frame %d/%d phase=%s finished=%s: %r (%.1fs)",
                episode_id,
                step_idx + 1,
                len(episode["frames"]),
                last_prompt_phase or "reuse",
                last_previous_finished,
                last_subtask,
                elapsed,
            )

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForeAct VLM planner over DROID cache frames")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_CHOICES),
        default="openai_compat",
        help="VLM backend. Default matches the paper's Qwen3-VL-8B via local vLLM.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID. Defaults: Qwen/Qwen3-VL-8B-Instruct (openai_compat) / "
        "gemini-robotics-er-1.6-preview (gemini).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible server URL. Default: http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="none",
        help="API key for openai_compat backend (local vLLM ignores this).",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Gemini thinking budget (gemini backend only; 0 disables).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Per-call retry budget for 429 / transient network errors (gemini backend).",
    )
    parser.add_argument(
        "--replan_every",
        type=int,
        default=1,
        help=(
            "Issue a new planner call every N cached frames; intermediate frames "
            "reuse the previous subtask text. Default 1 — the paper implies "
            "per-observation cadence."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Only process the first N episodes (for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    samples_dir = Path(args.samples_dir)
    manifest = load_manifest(samples_dir)
    if args.max_episodes is not None:
        manifest = manifest[: args.max_episodes]
    logger.info("Loaded manifest: %d episodes", len(manifest))

    backend = _parse_backend(args.backend)
    planner, backend_label = _build_planner(backend, args)
    logger.info(
        "Backend=%s label=%s replan_every=%d",
        backend,
        backend_label,
        args.replan_every,
    )

    all_records: list[dict[str, Any]] = []
    for episode in manifest:
        logger.info(
            "Starting episode %s (%d frames): %r",
            episode["episode_id"],
            len(episode["frames"]),
            episode["instruction"],
        )
        all_records.extend(
            _process_episode(planner, samples_dir, episode, replan_every=args.replan_every)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "prompt_format": "foreact_two_turn",
                "backend": backend_label,
                "results": all_records,
            },
            f,
            indent=2,
        )
    logger.info("Saved %d records to %s (backend=%s)", len(all_records), output_path, backend_label)

    gen_times = [r["generation_time_s"] for r in all_records if r["generation_time_s"] > 0]
    if gen_times:
        logger.info(
            "Planner latency: mean=%.2fs min=%.2fs max=%.2fs (replan calls only, n=%d)",
            float(np.mean(gen_times)),
            float(np.min(gen_times)),
            float(np.max(gen_times)),
            len(gen_times),
        )
    unique_subtasks = {r["subtask_text"] for r in all_records if r["subtask_text"]}
    logger.info("Unique subtask texts: %d / %d frames", len(unique_subtasks), len(all_records))
    empty = sum(1 for r in all_records if not r["subtask_text"])
    if empty:
        logger.warning("Empty subtasks: %d / %d frames", empty, len(all_records))
    advances = sum(1 for r in all_records if r["previous_finished"])
    logger.info("Planner reported previous_finished=True on %d frames", advances)


if __name__ == "__main__":
    main()
