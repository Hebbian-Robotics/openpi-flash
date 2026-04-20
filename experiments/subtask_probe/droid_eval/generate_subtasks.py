#!/usr/bin/env python3
"""Phase 1: Generate subtask text for DROID frames via deployed server.

Sends each cached DROID frame to the deployed server with mode="subtask_only"
to get subtask text, then caches the results for Phase 2.

Optionally configures the subtask prompt format on the server via the admin
HTTP endpoint before generation, so the same script can drive prompt-format
A/B tests against a single deployment.

Usage:
    # Use whichever subtask prompt format the server is currently configured with:
    uv run python experiments/subtask_probe/droid_eval/generate_subtasks.py \
        --samples_dir ./.experiments_cache/droid_eval \
        --output ./.experiments_cache/droid_eval/subtasks.json \
        --server 43.200.36.250

    # Override the subtask prompt format on the server before running:
    uv run python experiments/subtask_probe/droid_eval/generate_subtasks.py \
        --samples_dir ./.experiments_cache/droid_eval \
        --prompt_format '{task}' \
        --output ./.experiments_cache/droid_eval/subtasks_raw.json \
        --server 43.200.36.250
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from hosting.admin_server import DEFAULT_ADMIN_PORT
from hosting.flash_transport_policy import FlashTransportPolicy

from .constants import DEFAULT_QUIC_PORT
from .utils import build_subtask_observation, build_warmup_observation, load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def _admin_request(
    server: str,
    admin_port: int,
    method: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a request to the server's admin HTTP endpoint and return the parsed JSON response."""
    url = f"http://{server}:{admin_port}/config"
    try:
        response = httpx.request(method, url, json=body, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Admin endpoint returned {exc.response.status_code} for {method} {url}: "
            f"{exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach admin endpoint at {url}: {exc}") from exc
    return response.json()


def _set_server_prompt_format(server: str, admin_port: int, prompt_format: str) -> str:
    """PATCH the server's subtask prompt format and verify it was applied."""
    payload = _admin_request(
        server, admin_port, method="PATCH", body={"generation_prompt_format": prompt_format}
    )
    actual = payload.get("generation_prompt_format")
    if actual != prompt_format:
        raise RuntimeError(
            f"Admin endpoint did not apply prompt format. Requested {prompt_format!r}, "
            f"server reports {actual!r}."
        )
    return prompt_format


def _get_server_prompt_format(server: str, admin_port: int) -> str:
    """Fetch the server's currently-configured subtask prompt format."""
    payload = _admin_request(server, admin_port, method="GET")
    prompt_format = payload.get("generation_prompt_format")
    if not isinstance(prompt_format, str):
        raise RuntimeError(
            f"Admin endpoint returned no generation_prompt_format field: {payload!r}"
        )
    return prompt_format


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtasks for DROID frames via server")
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
        "--server",
        type=str,
        required=True,
        help="Server address (e.g., 43.200.36.250)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_QUIC_PORT, help="Server QUIC port")
    parser.add_argument(
        "--admin_host",
        type=str,
        default=None,
        help=(
            "Host for the admin HTTP endpoint. Defaults to --server. Use 127.0.0.1 with "
            "an SSH tunnel when the deployed server binds admin to localhost."
        ),
    )
    parser.add_argument(
        "--admin_port",
        type=int,
        default=DEFAULT_ADMIN_PORT,
        help="Admin HTTP port for runtime config (used to set/read --prompt_format)",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default=None,
        help=(
            "Subtask prompt format to install on the server before generation, "
            "e.g. 'Task: {task}. Subtask: ' or '{task}'. Must contain the literal "
            "'{{task}}' placeholder. If omitted, the server's current format is used."
        ),
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    manifest = load_manifest(samples_dir)
    logger.info("Loaded manifest: %d episodes", len(manifest))

    # Set or read the active subtask prompt format on the server before any inference.
    # The runtime config is read on every generate() call server-side, so this takes
    # effect on the next request.
    admin_host = args.admin_host or args.server
    if args.prompt_format is not None:
        active_prompt_format = _set_server_prompt_format(
            admin_host, args.admin_port, args.prompt_format
        )
        logger.info("Set server subtask prompt format to %r", active_prompt_format)
    else:
        active_prompt_format = _get_server_prompt_format(admin_host, args.admin_port)
        logger.info("Using server's current subtask prompt format: %r", active_prompt_format)

    # Connect to server via QUIC
    policy = FlashTransportPolicy(args.server, port=args.port)
    logger.info("Connected to server at %s:%d via QUIC", args.server, args.port)

    # Warm up connection
    policy.infer(build_warmup_observation(mode="subtask_only"))
    logger.info("Server warmup complete")

    # Process all frames
    subtask_results = []
    total_frames = sum(ep["num_frames"] for ep in manifest)
    processed = 0

    for episode in manifest:
        episode_id = episode["episode_id"]
        instruction = episode["instruction"]

        for frame_info in episode["frames"]:
            frame_file = samples_dir / frame_info["file"]
            frame_data = np.load(frame_file)

            # Send raw uint8 images — the server's _normalize_image() handles
            # conversion to float32 [-1, 1] and resize_with_pad to 224x224.
            obs = build_subtask_observation(
                exterior_image=frame_data["exterior_image"],
                wrist_image=frame_data["wrist_image"],
                prompt=instruction,
            )

            start_time = time.time()
            result = policy.infer(obs)
            elapsed = time.time() - start_time

            subtask_info = result.get("subtask", {})
            subtask_text = subtask_info.get("text", "") or result.get("subtask_text", "")
            subtask_ms = subtask_info.get("ms", elapsed * 1000)

            subtask_results.append(
                {
                    "episode_id": episode_id,
                    "frame_idx": frame_info["frame_idx"],
                    "instruction": instruction,
                    "subtask_text": subtask_text,
                    "generation_time_s": round(elapsed, 2),
                    "server_subtask_ms": round(subtask_ms, 1),
                }
            )

            processed += 1
            if processed % 5 == 0 or processed == total_frames:
                logger.info(
                    "[%d/%d] %s frame %d: '%s' (%.1fs)",
                    processed,
                    total_frames,
                    episode_id,
                    frame_info["frame_idx"],
                    subtask_text,
                    elapsed,
                )

    # Save results — wrap in a self-describing dict so consumers know which prompt
    # format produced these subtasks. load_subtask_records() handles both this shape
    # and the legacy bare-list shape for backward compatibility.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {"prompt_format": active_prompt_format, "results": subtask_results},
            f,
            indent=2,
        )

    logger.info(
        "Subtask generation complete: %d results saved to %s (prompt_format=%r)",
        len(subtask_results),
        output_path,
        active_prompt_format,
    )

    # Summary stats
    gen_times = [r["generation_time_s"] for r in subtask_results]
    logger.info(
        "Latency: mean=%.2fs, min=%.2fs, max=%.2fs",
        np.mean(gen_times),
        np.min(gen_times),
        np.max(gen_times),
    )
    unique_subtasks = {r["subtask_text"] for r in subtask_results}
    logger.info(
        "Unique subtask texts: %d out of %d frames", len(unique_subtasks), len(subtask_results)
    )


if __name__ == "__main__":
    main()
