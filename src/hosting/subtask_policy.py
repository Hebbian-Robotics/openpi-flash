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

import logging
import time
from typing import Literal

import numpy as np
from openpi_client import base_policy as _base_policy

from hosting.subtask_generator import SubtaskGenerator

logger = logging.getLogger(__name__)

# Finite set of inference modes sent by clients in the observation dict.
# Parsed from raw string at the infer() boundary.
InferenceMode = Literal["default", "subtask_only", "action_only"]

DEFAULT_ACTION_PROMPT_TEMPLATE = "{task}. Subtask: {subtask}"


def _parse_inference_mode(raw: str) -> InferenceMode:
    """Parse a raw mode string from the client into a typed InferenceMode.

    Unknown values fall back to "default" (both subtask + action phases).
    """
    if raw == "subtask_only":
        return "subtask_only"
    if raw == "action_only":
        return "action_only"
    if raw and raw != "default":
        logger.warning("Unknown inference mode '%s', falling back to default", raw)
    return "default"


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
        subtask_text = ""
        subtask_generation_time_ms = 0.0
        try:
            subtask_start = time.monotonic()
            client_images = _extract_images(obs)
            subtask_text = self._subtask_generator.generate(original_prompt, client_images)
            subtask_generation_time_ms = (time.monotonic() - subtask_start) * 1000
        except Exception:
            logger.exception("[subtask] Generation failed, falling back to original prompt")

        if mode == "subtask_only":
            return {
                "subtask": {
                    "text": subtask_text,
                    "ms": subtask_generation_time_ms,
                },
            }

        # Default mode: augment prompt and run action generation
        if subtask_text:
            augmented_prompt = self._prompt_template.format(
                task=original_prompt, subtask=subtask_text
            )
            obs = {**obs, "prompt": augmented_prompt}

        result = self._inner_policy.infer(obs)

        result["subtask"] = {
            "text": subtask_text,
            "ms": subtask_generation_time_ms,
        }

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
        subtask_text = ""
        subtask_start = time.monotonic()
        try:
            client_images = _extract_images(obs)
            subtask_text = self._subtask_generator.generate(task_prompt, client_images)
        except Exception:
            logger.exception("[planner] Generation failed")
        subtask_generation_time_ms = (time.monotonic() - subtask_start) * 1000
        return {
            "subtask": {
                "text": subtask_text,
                "ms": subtask_generation_time_ms,
            },
        }

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
