#!/usr/bin/env python3
"""Foresight image generation via Gemini 3.1 Flash Image ("Nano Banana").

Drop-in alternative to the ForeAct SANA+Gemma generator. Doesn't require a
GPU or any pretrained robot-specific weights — just the google-genai SDK and
a careful generic scene-rules prompt.

Two structural choices worth knowing:

1. **Generic scene-rules prompt + subtask slot.** A single ``SCENE_RULES_TEMPLATE``
   encodes the scene inventory, robot anatomy, grasp physics, and preservation
   constraints. Per-frame subtask labels (from ``CHAIN_PHASES`` or, eventually,
   a Qwen3-VL planner) slot into ``{subtask}``. This replaces the earlier
   per-phase hand-crafted prompts and makes short planner output pluggable.

2. **Two-image conditioning.** Each API call passes (reference_frame,
   current_observation, prompt). The reference frame (ep0 f00, all objects on
   the table, nothing occluded) gives the stateless image generator a visual
   anchor for object identity. This is specifically to combat the "eggplant
   morphs to apple during carry frames" failure mode we saw with single-image
   conditioning when the eggplant is occluded inside the closed gripper.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

from google import genai
from google.genai import types
from PIL import Image

from experiments.subtask_probe.droid_eval.foreact_eval._io import (
    ForesightStatus,
    foresight_path,
    iter_source_frames,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


# ---------------------------------------------------------------------------
# Generic scene-rules prompt + per-phase subtask slot.
#
# Rationale: the earlier version had a bespoke ~200-word descriptive prompt
# per phase (PICK_UP_PROMPT / PLACE_PROMPT / RETURN_HOME_PROMPT). That made
# each phase's prompt a one-shot artifact — you couldn't plug in a short
# subtask label from a VLM planner (Qwen3-VL) without rewriting the prompt.
# The new design splits out (a) a stable SCENE_RULES prompt that encodes the
# physics and scene inventory once, and (b) a short ``subtask`` string that
# slots in per frame. Now any short subtask label (hardcoded from
# CHAIN_PHASES or dynamic from a planner) drives the generator.
#
# Multi-image input: we also pass a REFERENCE FRAME (ep0 f00, all objects on
# the table with nothing occluded) alongside the current observation. With a
# stateless single-image generator, the eggplant's identity collapses during
# carry frames where it's occluded behind the closed gripper. The reference
# frame gives the model an anchor for what the eggplant looks like.
# ---------------------------------------------------------------------------

SCENE_RULES_TEMPLATE = """\
You will be given TWO images followed by a subtask instruction.

IMAGE 1 — REFERENCE FRAME. The scene in its starting state with all objects \
clearly visible. Use this STRICTLY as a reference for OBJECT IDENTITY — \
especially what the small dark-purple eggplant looks like, since it will be \
occluded in later frames. Do NOT output this frame as the future; it is \
only for identity anchoring.

IMAGE 2 — CURRENT OBSERVATION. This is the frame from which you will \
predict the scene's FUTURE state. The future image must preserve this \
frame's camera viewpoint, lighting, table surface, and background walls \
EXACTLY; only what the subtask requires should change.

SCENE INVENTORY (left to right, as shown in the reference frame): green \
leek with white base on the far left; small red chili pepper next to the \
leek; empty blue plate in the middle of the table; small dark-purple \
eggplant (oval, glossy dark-purple skin, with a short stem); green \
zucchini; yellow corn on the far right.

ROBOT: Galaxea R1 Lite right arm. Enters from the UPPER-RIGHT EDGE of the \
frame and is always continuous to that edge — no floating grippers. \
Orange-red upper-arm segment connected to the top-right corner, black \
forearm, black parallel-finger gripper with exactly two parallel fingers \
(not a claw). Grasps oblong objects by their stem, so a held object hangs \
BELOW the closed fingers as a visible silhouette.

SUBTASK SEMANTICS: subtask instructions name actions the ROBOT ARM takes, \
not actions that undo previous work. Specifically, "return to home \
position" or "go home" means the ARM RETRACTS upward/out of frame — it \
does NOT mean moving any object back. Once an object is in the blue \
plate, it stays in the plate unless a future subtask explicitly says to \
move it somewhere else. Only the subtask in "CURRENT SUBTASK" below is \
active; do not reverse earlier subtasks.

WORLD PRESERVATION RULE (strict): anything the current subtask does NOT \
explicitly act on must appear in the output image EXACTLY as it does in \
IMAGE 2 — same position, same orientation, same color, same condition. \
Do not delete, duplicate, relocate, or alter any object the subtask does \
not touch. If an object (including the eggplant, the blue plate, or any \
vegetable) is visible in IMAGE 2, it must be visible in the same place \
in the output unless the subtask explicitly moves it. The world is \
static except where the robot arm is actively manipulating something.

CURRENT SUBTASK: {subtask}

TASK: Predict the scene at roughly the half-subtask-ahead point of this \
subtask (the paper's "t + 0.5 times subtask duration" target). If the subtask \
involves moving the eggplant, the predicted frame should show that motion \
well underway or complete; if the eggplant ends up inside the blue plate, \
render it as the SAME dark-purple eggplant shown in IMAGE 1 (do not morph \
it to an apple, plum, or any other fruit). If the eggplant is occluded \
inside the closed gripper, do NOT draw it on the table or invent a \
substitute. The four non-target vegetables stay in their exact positions \
from IMAGE 2 unless the subtask says otherwise.
"""


class Phase(NamedTuple):
    """One subtask's extent within the chain.

    ``start_frame`` / ``end_frame`` are inclusive and refer to raw frame
    indices inside ``episode_index`` (same scale as filenames on disk —
    stride=5 between consecutive frames). Frames outside any phase are
    skipped by both the generator and the visualizer.
    """

    episode_index: int
    start_frame: int
    end_frame: int
    subtask_label: str


# Boundaries tuned against the v2 golden chain's actual physics:
#   * ep0 f00-25: arm not in frame yet → trim (no API call, no video frame).
#   * ep0 f30 → ep1 f25: approach + grasp → "Pick up" subtask.
#   * ep1 f30 → ep2 f15: eggplant in air, carried + placed → "Place" subtask.
#   * ep2 f20 → f55: arm retracting to home → "Return home" subtask.
#   * ep2 f60+: arm already home, scene static → trim.
CHAIN_PHASES: list[Phase] = [
    Phase(episode_index=0, start_frame=30, end_frame=100, subtask_label="Pick up the eggplant."),
    Phase(episode_index=1, start_frame=0, end_frame=25, subtask_label="Pick up the eggplant."),
    Phase(
        episode_index=1,
        start_frame=30,
        end_frame=75,
        subtask_label="Place the eggplant into the plate.",
    ),
    Phase(
        episode_index=2,
        start_frame=0,
        end_frame=15,
        subtask_label="Place the eggplant into the plate.",
    ),
    Phase(episode_index=2, start_frame=20, end_frame=55, subtask_label="Return to home position."),
]


# Default reference frame — ep0 f00 of the v2 chain has all objects on the
# table with nothing occluded, which is what we want as an identity anchor.
DEFAULT_REFERENCE_FRAME = Path(
    ".experiments_cache/foreact_eval/foresight_chain_eggplant_v2/episode_000000/actual/frame_00000.png"
)


def lookup_phase(episode_index: int, frame_idx: int) -> Phase | None:
    """Return the phase that owns (episode_index, frame_idx), or None if trimmed."""
    for phase in CHAIN_PHASES:
        if phase.episode_index != episode_index:
            continue
        if phase.start_frame <= frame_idx <= phase.end_frame:
            return phase
    return None


def _extract_image(response: Any) -> bytes | None:
    """Pull the first inline image from a Gemini generate_content response."""
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def _extract_text(response: Any) -> str:
    """Concatenate any text parts — Gemini sometimes narrates alongside the image."""
    texts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _generate_one(
    client: genai.Client,
    model: str,
    reference_bytes: bytes,
    current_bytes: bytes,
    prompt: str,
) -> tuple[bytes | None, str]:
    """Call Gemini with (reference_frame, current_frame, prompt) and return (image, text).

    Passing two images in order lets the model anchor object identity to the
    reference frame — critical for the "Place" phase where the eggplant is
    occluded inside the closed gripper in the current observation.
    """
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=reference_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=current_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    return _extract_image(response), _extract_text(response)


def _chain_record(
    *,
    episode_index: int,
    frame_idx: int,
    subtask_text: str,
    status: ForesightStatus,
    output: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "episode_index": episode_index,
        "frame_idx": frame_idx,
        "subtask_text": subtask_text,
        "output": output,
        "status": status,
    }
    if extra:
        record.update(extra)
    return record


def _run_chain(
    client: genai.Client,
    model: str,
    v2_root: Path,
    output_dir: Path,
    reference_frame_path: Path,
    force: bool,
) -> None:
    """Generate foresight for every in-phase frame of the v2 chain.

    Each call passes (reference_frame, current_frame, generic_prompt) to
    anchor object identity. Frames outside all CHAIN_PHASES (e.g. pre-arm
    intro frames or post-arm-home tail frames) are trimmed — no API call,
    no manifest entry. Output layout matches the ForeAct generator so
    downstream visualization tools can point at this directory:
    ``<output>/episode_{id:06d}/frame_{idx:05d}.png``.
    """
    if not reference_frame_path.exists():
        raise SystemExit(f"reference frame missing: {reference_frame_path}")
    reference_bytes = reference_frame_path.read_bytes()
    logger.info("Using reference frame: %s (%d bytes)", reference_frame_path, len(reference_bytes))

    all_source_frames = iter_source_frames(v2_root)
    frames = [
        f for f in all_source_frames if lookup_phase(f.episode_index, f.frame_idx) is not None
    ]
    trimmed = len(all_source_frames) - len(frames)
    logger.info("Chain has %d in-phase frames (trimmed %d)", len(frames), trimmed)

    records: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        phase = lookup_phase(frame.episode_index, frame.frame_idx)
        assert phase is not None  # filtered above
        out_path = foresight_path(output_dir, frame.episode_index, frame.frame_idx)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        relative_out = str(out_path.relative_to(output_dir))
        progress = f"[{i + 1}/{len(frames)}] ep{frame.episode_index} f{frame.frame_idx:05d}"

        if out_path.exists() and not force:
            logger.info("%s: already exists, skipping", progress)
            records.append(
                _chain_record(
                    episode_index=frame.episode_index,
                    frame_idx=frame.frame_idx,
                    subtask_text=phase.subtask_label,
                    status="cached",
                    output=relative_out,
                )
            )
            continue

        current_bytes = frame.actual_path.read_bytes()
        prompt = SCENE_RULES_TEMPLATE.format(subtask=phase.subtask_label)
        logger.info("%s: generating (subtask=%r)", progress, phase.subtask_label)
        try:
            out_bytes, narration = _generate_one(
                client, model, reference_bytes, current_bytes, prompt
            )
        except Exception as exc:
            logger.warning("%s: failed: %s", progress, exc)
            records.append(
                _chain_record(
                    episode_index=frame.episode_index,
                    frame_idx=frame.frame_idx,
                    subtask_text=phase.subtask_label,
                    status="failed",
                    output=None,
                    extra={"error": str(exc)},
                )
            )
            continue
        if out_bytes is None:
            logger.warning("%s: no image (narration=%r)", progress, narration[:200])
            records.append(
                _chain_record(
                    episode_index=frame.episode_index,
                    frame_idx=frame.frame_idx,
                    subtask_text=phase.subtask_label,
                    status="refused",
                    output=None,
                    extra={"narration": narration},
                )
            )
            continue
        image = Image.open(io.BytesIO(out_bytes)).convert("RGB")
        image.save(out_path)
        records.append(
            _chain_record(
                episode_index=frame.episode_index,
                frame_idx=frame.frame_idx,
                subtask_text=phase.subtask_label,
                status="generated",
                output=relative_out,
            )
        )

    manifest_path = output_dir / "foresight_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "source_dataset": str(v2_root),
                "model": model,
                "chain_phases": [phase._asdict() for phase in CHAIN_PHASES],
                "records": records,
            },
            f,
            indent=2,
        )
    logger.info("Wrote manifest -> %s", manifest_path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    logger.info("Status counts: %s", counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nano Banana foresight generator (generic scene-rules + 2-image conditioning)"
    )
    parser.add_argument(
        "--v2_root",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/foresight_chain_eggplant_v2"),
        help="Directory containing the v2 golden chain episodes with actual/ frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(".experiments_cache/foreact_eval/foresight_nano_banana_chain"),
    )
    parser.add_argument(
        "--reference_frame",
        type=Path,
        default=DEFAULT_REFERENCE_FRAME,
        help="Identity-anchor frame shown alongside each current observation. Default: ep0 f00.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if an output PNG already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client()
    _run_chain(
        client,
        args.model,
        args.v2_root,
        args.output_dir,
        args.reference_frame,
        args.force,
    )
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
