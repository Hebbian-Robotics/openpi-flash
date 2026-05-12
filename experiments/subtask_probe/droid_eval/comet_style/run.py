#!/usr/bin/env python3
"""Phase 1 (alt 2): Comet-style hierarchical subtask generation for DROID frames.

Runs a stateful plan -> critique -> subtask loop per episode, ported from
``openpi-comet/src/openpi/shared/client.py``. Supports two backends:

  * ``--backend gemini``          — Gemini Robotics-ER 1.6 Preview (default).
  * ``--backend openai_compat``   — any OpenAI-compatible chat-completions
                                     server (e.g. a local vLLM hosting
                                     Qwen3-VL-30B-A3B-Instruct).

Output JSON schema matches ``generate_subtasks_gemini.py`` so ``run_action_eval``,
``compute_metrics`` and ``visualize_results`` consume it unchanged.

Usage:
    # Gemini backend (requires GEMINI_API_KEY)
    uv run python -m experiments.subtask_probe.droid_eval.comet_style.run \\
        --samples_dir ./.experiments_cache/droid_eval_2min \\
        --output ./.experiments_cache/droid_eval_2min/subtasks_comet_gemini.json \\
        --backend gemini

    # OpenAI-compatible backend (vLLM hosting Qwen3-VL-30B)
    uv run python -m experiments.subtask_probe.droid_eval.comet_style.run \\
        --samples_dir ./.experiments_cache/droid_eval_2min \\
        --output ./.experiments_cache/droid_eval_2min/subtasks_comet_qwen.json \\
        --backend openai_compat \\
        --base_url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-VL-30B-A3B-Instruct
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

from experiments.subtask_probe.droid_eval.utils import load_manifest

from .reasoner_base import BaseReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

Backend = Literal["gemini", "openai_compat"]
BACKEND_CHOICES: tuple[Backend, ...] = get_args(Backend)


def _parse_backend(raw: str) -> Backend:
    """Narrow the argparse string into the ``Backend`` literal union.

    argparse's ``choices=`` already rejects bad values at runtime; this
    function exists to carry that proof into the type system.
    """
    if raw in BACKEND_CHOICES:
        return cast(Backend, raw)
    raise ValueError(f"Unknown backend: {raw!r} (expected one of {BACKEND_CHOICES})")


def _build_reasoner(
    backend: Backend,
    args: argparse.Namespace,
) -> tuple[BaseReasoner, str]:
    """Instantiate the requested backend and return (reasoner, backend_label).

    ``backend_label`` gets written into the output JSON's ``backend`` field so
    downstream tooling can tell runs apart.
    """
    match backend:
        case "gemini":
            from .gemini_reasoner import DEFAULT_MODEL as GEMINI_DEFAULT
            from .gemini_reasoner import GeminiReasoner

            model = args.model or GEMINI_DEFAULT
            reasoner: BaseReasoner = GeminiReasoner(
                model=model,
                thinking_budget=args.thinking_budget,
                max_retries=args.max_retries,
                history_maxlen=args.history_maxlen,
                sampled_images_max=args.sampled_images_max,
                history_stride=args.history_stride,
            )
            return reasoner, model
        case "openai_compat":
            from .openai_compat_reasoner import DEFAULT_BASE_URL, OpenAICompatReasoner
            from .openai_compat_reasoner import DEFAULT_MODEL as OAI_DEFAULT

            base_url = args.base_url or DEFAULT_BASE_URL
            model = args.model or OAI_DEFAULT
            reasoner = OpenAICompatReasoner(
                base_url=base_url,
                model=model,
                api_key=args.api_key,
                history_maxlen=args.history_maxlen,
                sampled_images_max=args.sampled_images_max,
                history_stride=args.history_stride,
            )
            return reasoner, f"{model}@{base_url}"
        case _ as unreachable:
            assert_never(unreachable)


def _process_episode(
    reasoner: BaseReasoner,
    samples_dir: Path,
    episode: dict[str, Any],
    replan_every: int,
) -> list[dict[str, Any]]:
    """Run the reasoner across all frames of one episode.

    Calls ``reasoner.reset()`` at the start. On replan frames the reasoner
    issues 2 VLM calls (plan/critique + subtask); non-replan frames reuse
    the last subtask text and make no VLM calls.
    """
    reasoner.reset()
    records: list[dict[str, Any]] = []
    episode_id = episode["episode_id"]
    instruction = episode["instruction"]
    last_subtask = ""

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
                last_subtask = reasoner.generate_subtask(instruction, [exterior_image])
            except Exception as exc:
                logger.warning(
                    "Reasoner call failed for %s frame %d: %s",
                    episode_id,
                    frame_idx,
                    exc,
                )
                last_subtask = ""
            elapsed = time.time() - start

        records.append(
            {
                "episode_id": episode_id,
                "frame_idx": frame_idx,
                "instruction": instruction,
                "subtask_text": last_subtask,
                "generation_time_s": round(elapsed, 2),
                "server_subtask_ms": round(elapsed * 1000, 1),
            }
        )

        if step_idx % 5 == 0 or step_idx == len(episode["frames"]) - 1:
            logger.info(
                "[%s] frame %d/%d: %r (%.1fs, replan=%s)",
                episode_id,
                step_idx + 1,
                len(episode["frames"]),
                last_subtask,
                elapsed,
                is_replan,
            )

    logger.info(
        "[%s] done — final plan_status:\n%s",
        episode_id,
        reasoner.plan_status or "<empty>",
    )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comet-style hierarchical subtask generation for DROID frames"
    )
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_CHOICES),
        default="gemini",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID. Defaults: gemini-robotics-er-1.6-preview (gemini) / "
        "Qwen/Qwen3-VL-30B-A3B-Instruct (openai_compat).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible server URL (openai_compat backend only). "
        "Default: http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="none",
        help="API key for openai_compat backend (local vLLM ignores this)",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Gemini thinking budget (gemini backend only; 0 disables thinking)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Per-call retry budget for 429 (gemini backend only)",
    )
    parser.add_argument(
        "--replan_every",
        type=int,
        default=1,
        help=(
            "Issue a new plan+critique+subtask every N cached frames; "
            "intermediate frames reuse the previous subtask text. Default 1."
        ),
    )
    parser.add_argument(
        "--history_maxlen",
        type=int,
        default=640,
        help="Per-episode image history deque size. Default matches Comet's 64*10.",
    )
    parser.add_argument(
        "--sampled_images_max",
        type=int,
        default=64,
        help="Max images sampled from history per VLM call.",
    )
    parser.add_argument(
        "--history_stride",
        type=int,
        default=5,
        help=(
            "Stride used when sampling history for each VLM call. Default "
            "5 matches Comet's original hardcoded value and empirically "
            "gives the best plan-stability on our 1 Hz cache because the "
            "wider temporal span provides the contrast signal the reasoner "
            "needs to detect progression. stride=1 over-interprets "
            "frame-to-frame motion on slow tasks — see FINDINGS.md."
        ),
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Only process the first N episodes from the manifest (for triage runs).",
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
    reasoner, backend_label = _build_reasoner(backend, args)
    logger.info(
        "Backend=%s model=%s replan_every=%d history_maxlen=%d sampled_images_max=%d",
        backend,
        backend_label,
        args.replan_every,
        args.history_maxlen,
        args.sampled_images_max,
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
            _process_episode(reasoner, samples_dir, episode, replan_every=args.replan_every)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "prompt_format": "comet_style_hierarchical",
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
            "Latency: mean=%.2fs, min=%.2fs, max=%.2fs (replan calls only)",
            float(np.mean(gen_times)),
            float(np.min(gen_times)),
            float(np.max(gen_times)),
        )
    unique_subtasks = {r["subtask_text"] for r in all_records if r["subtask_text"]}
    logger.info("Unique subtask texts: %d out of %d frames", len(unique_subtasks), len(all_records))
    failed = sum(1 for r in all_records if not r["subtask_text"])
    if failed:
        logger.warning("Empty subtasks: %d / %d frames", failed, len(all_records))


if __name__ == "__main__":
    main()
