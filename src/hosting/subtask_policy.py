"""Policy wrappers for the pi0.5 subtask planner.

Two wrappers, both backed by the same ``SubtaskGenerator`` instance:

- ``SubtaskAugmentedPolicy`` — combined-mode wrapper around the action policy.
  Generates a subtask, splices it into the prompt, then runs action inference.
- ``PlannerPolicy`` — planner-only wrapper. Returns just subtask text and no
  actions, for deployments where the action model isn't loaded.

Both wrappers preserve the ``BasePolicy`` contract the transport layer expects.
The client API is unchanged — the subtask phase is transparent on the action
endpoint, and the planner endpoint returns a response without an ``actions``
field. Clients tell them apart by which port they connect to.
"""

from __future__ import annotations

import time
import traceback
from typing import Literal, TypedDict

import numpy as np
from openpi_client import base_policy as _base_policy

from hosting.config import DEFAULT_ACTION_PROMPT_TEMPLATE
from hosting.subtask_generator import SubtaskGenerator

# Finite set of inference modes sent by clients in the observation dict.
# Parsed from raw string at the infer() boundary.
InferenceMode = Literal["default", "subtask_only", "action_only"]


class SubtaskPayload(TypedDict):
    """The ``subtask`` block attached to responses on both planner-backed endpoints."""

    text: str
    ms: float


def _parse_inference_mode(raw: str) -> InferenceMode:
    """Parse a raw mode string from the client into a typed InferenceMode.

    Unknown values fall back to "default" (both subtask + action phases).
    """
    if raw == "subtask_only":
        return "subtask_only"
    if raw == "action_only":
        return "action_only"
    if raw and raw != "default":
        print(f"[subtask] Unknown inference mode '{raw}', falling back to default", flush=True)
    return "default"


def _generate_subtask_payload(
    subtask_generator: SubtaskGenerator,
    task_prompt: str,
    obs: dict,
    log_context: str,
) -> SubtaskPayload:
    """Run subtask generation with timing, shared by both endpoint wrappers.

    Owning the try/timing/response shape in one place keeps the failure
    semantics identical on the action and planner endpoints: empty text on
    failure, ``ms`` always the elapsed wall time (including failed attempts).
    """
    subtask_text = ""
    subtask_start = time.monotonic()
    try:
        client_images = _extract_images(obs)
        subtask_text = subtask_generator.generate(task_prompt, client_images)
    except Exception:
        print(
            f"[{log_context}] Subtask generation failed, returning empty subtask:\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
    subtask_generation_time_ms = (time.monotonic() - subtask_start) * 1000
    return SubtaskPayload(text=subtask_text, ms=subtask_generation_time_ms)


class SubtaskAugmentedPolicy(_base_policy.BasePolicy):
    """Wraps an action policy to transparently add subtask generation before inference.

    On each infer() call:
      1. Extracts the task prompt and images from the observation
      2. Generates a subtask via the JAX subtask generator
      3. Augments the prompt: "{task}. Subtask: {subtask}" (configurable)
      4. Passes the augmented observation to the inner (action) policy
      5. Returns the action result unchanged, with subtask metadata added

    On failure, falls back to the original prompt (graceful degradation).
    """

    def __init__(
        self,
        inner_policy: _base_policy.BasePolicy,
        subtask_generator: SubtaskGenerator,
        prompt_template: str = DEFAULT_ACTION_PROMPT_TEMPLATE,
    ) -> None:
        self._inner_policy = inner_policy
        self._subtask_generator = subtask_generator
        self._prompt_template = prompt_template

    def infer(self, obs: dict) -> dict:
        mode = _parse_inference_mode(obs.get("mode", ""))
        original_prompt = obs.get("prompt", "")

        if mode == "action_only":
            return self._inner_policy.infer(obs)

        # Generate subtask (for both default and subtask_only modes)
        subtask_payload = _generate_subtask_payload(
            self._subtask_generator, original_prompt, obs, log_context="subtask"
        )

        if mode == "subtask_only":
            return {"subtask": subtask_payload}

        # Default mode: augment prompt and run action generation. On
        # generation failure the subtask text is empty and the original
        # prompt passes through unchanged (graceful degradation).
        if subtask_payload["text"]:
            augmented_prompt = self._prompt_template.format(
                task=original_prompt, subtask=subtask_payload["text"]
            )
            obs = {**obs, "prompt": augmented_prompt}

        result = self._inner_policy.infer(obs)
        result["subtask"] = subtask_payload
        return result

    def reset(self) -> None:
        self._inner_policy.reset()


class PlannerPolicy(_base_policy.BasePolicy):
    """Planner-only policy: returns subtask text, no actions.

    Used when the server is deployed without an action slot, or when a client
    specifically wants to hit the planner endpoint in combined mode. The
    response shape is ``{"subtask": {"text": ..., "ms": ...}}`` — no
    ``actions`` field, since this endpoint doesn't generate them.
    """

    def __init__(self, subtask_generator: SubtaskGenerator) -> None:
        self._subtask_generator = subtask_generator

    def infer(self, obs: dict) -> dict:
        task_prompt = obs.get("prompt", "")
        subtask_payload = _generate_subtask_payload(
            self._subtask_generator, task_prompt, obs, log_context="planner"
        )
        return {"subtask": subtask_payload}

    def reset(self) -> None:
        return


def _extract_images(obs: dict) -> dict[str, np.ndarray] | None:
    """Extract camera images from the raw observation dict.

    Handles multiple embodiment formats:
      - ALOHA / JAX direct: obs["images"] = {"cam_high": ..., "base_0_rgb": ...}
      - DROID: obs["observation/exterior_image_1_left"] = ...

    Returns images as numpy arrays keyed by their original names, or None.
    """
    images = obs.get("images")
    if images is not None and isinstance(images, dict):
        return {key: np.asarray(value) for key, value in images.items() if value is not None}

    droid_image_keys = [k for k in obs if k.startswith("observation/") and "image" in k]
    if droid_image_keys:
        return {key: np.asarray(obs[key]) for key in droid_image_keys}

    return None
