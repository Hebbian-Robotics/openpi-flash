"""Backend-agnostic scaffold for Comet-style hierarchical subtask generation.

Ports ``strip_think_tags``, ``generate_stylized_plan``, ``sample_images`` and
the three-step plan -> critique -> subtask control flow verbatim from
``openpi-comet/src/openpi/shared/client.py``. The prompt strings are copied
byte-for-byte from that file so we're testing Comet's prompting, not our
rewording.

Concrete backends subclass ``BaseReasoner`` and implement a single ``_chat``
hook that sends a VLM request with a user prompt plus a list of images and
returns the response text.
"""

from __future__ import annotations

import abc
import collections
import json
import logging
import re
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


# Comet defaults: deque(maxlen=64*10) for image history, sample at most 64 images per call
# (client.py:178, 72-79).
DEFAULT_HISTORY_MAXLEN = 64 * 10
DEFAULT_SAMPLED_IMAGES_MAX = 64

# Status markers for plan steps, matching Comet's stylized plan output:
#   [o] done, [-] in_progress, [x] not_started
PlanStepStatus = Literal["done", "in_progress", "not_started"]
_STATUS_VALUES: tuple[PlanStepStatus, ...] = ("done", "in_progress", "not_started")
_STATUS_MARKERS: dict[PlanStepStatus, str] = {
    "done": "[o]",
    "in_progress": "[-]",
    "not_started": "[x]",
}

# JSON schemas enforced via the backend's native structured-output mechanism
# (Gemini response_schema / vLLM xgrammar json_schema). Both backends translate
# these dicts to their native representation.
#
# The schemas are designed to produce short outputs compatible with the pi0.5
# action prompt budget: plans are 2-10 short step strings, subtasks are a
# single short imperative phrase (~120 chars / ~20 words).
PLAN_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 2,
    "maxItems": 10,
}

SUBTASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtask": {
            "type": "string",
            "maxLength": 120,
        },
    },
    "required": ["subtask"],
}

# plan_critique returns a parallel list of statuses — one per step of the
# original plan, in the same order. Using an enum makes structural equality
# meaningful (no prose drift) so we only reset ``subtask_history`` when the
# plan state actually changes, not when wording varies.
CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statuses": {
            "type": "array",
            "items": {"type": "string", "enum": list(_STATUS_VALUES)},
        },
    },
    "required": ["statuses"],
}


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` content or any leading ``</think>`` preamble.

    Mirrors ``openpi-comet/src/openpi/shared/client.py:82``.
    """
    lower = text.lower()
    if "<think>" in lower and "</think>" in lower:
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    elif "</think>" in lower:
        cleaned_text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    else:
        cleaned_text = text
    return re.sub(r"\n\s*\n", "\n", cleaned_text.strip())


def parse_plan_list(text: str) -> list[str] | None:
    """Parse a VLM response that should be a JSON list of plan-step strings.

    Returns the parsed list on success, or ``None`` when the response cannot
    be coerced into a non-empty list of strings (bad JSON, wrong shape,
    non-string items, empty list). Callers are expected to fall back to a
    single-step plan in that case.

    Handles the two common messy outputs from reasoning VLMs: ``<think>``
    tags around the response and markdown code fences (```json ... ```).
    Comet uses ``json_repair`` for further tolerance; we avoid the dep and
    surface parse failures as ``None`` instead.
    """
    cleaned = strip_think_tags(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            return None
        result.append(item)
    return result


def render_plan_status(plans: list[str], statuses: list[PlanStepStatus]) -> str:
    """Render a plan + per-step status list into Comet's stylized string format.

    Example output::

        [o] pick up the cube
        [-] move to the dish
        [x] release the cube

    Mirrors the shape of ``openpi-comet/src/openpi/shared/client.py:118`` but
    is driven by discrete enum statuses rather than an index + completion
    flag, which maps cleanly onto our structured-output critique responses.
    """
    if len(plans) != len(statuses):
        raise ValueError(
            f"plans/statuses length mismatch: {len(plans)} steps vs {len(statuses)} statuses"
        )
    return "\n".join(
        f"{_STATUS_MARKERS[status]} {step}" for step, status in zip(plans, statuses, strict=True)
    )


def initial_statuses(num_steps: int) -> list[PlanStepStatus]:
    """Starting state: first step in_progress, the rest not_started."""
    if num_steps <= 0:
        raise ValueError("initial_statuses requires at least one step")
    statuses: list[PlanStepStatus] = ["in_progress"]
    statuses.extend(["not_started"] * (num_steps - 1))
    return statuses


def sample_images(
    image_list: list[np.ndarray],
    max_len: int = 64,
    stride: int = 5,
) -> list[np.ndarray]:
    """Sample every ``stride``-th image in reverse from the most recent, up to ``max_len``.

    Mirrors ``openpi-comet/src/openpi/shared/client.py:72``, which hardcoded
    stride=5 because their history buffer was populated at the 30 Hz sim rate
    and they wanted ~6 Hz of temporal density per VLM call. On our DROID
    cache the buffer is populated at the *cached* rate (1 Hz with
    ``--frame_subsample=15``), so stride=5 skips over almost everything.
    Rule of thumb: ``stride ≈ cache_hz / desired_sample_hz`` (with floor 1).
    For our 1 Hz cache, stride=1 means the VLM sees the last ``max_len``
    consecutive seconds of history.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    sampled: list[np.ndarray] = []
    for i in range(len(image_list) - 1, -1, -stride):
        sampled.append(image_list[i])
        if len(sampled) >= max_len:
            break
    sampled.reverse()
    return sampled


def all_steps_done(statuses: list[PlanStepStatus] | None) -> bool:
    """Return True when every step is marked ``done``.

    DROID cached frames can continue well past the point where the robot has
    finished the instructed task. Once the plan is fully complete, the VLM
    has nothing useful to say and its degenerate outputs ("finished", "none")
    pollute the action prompt. Callers short-circuit on this condition.
    """
    return bool(statuses) and all(s == "done" for s in statuses)


class BaseReasoner(abc.ABC):
    """Stateful hierarchical reasoner shared by all backends.

    State that carries across frames within an episode:
      * ``history_multi_modals`` — ring buffer of past images
      * ``subtask_history`` — ordered list of subtask strings produced so far
      * ``plan_status`` — the current stylized plan string with status markers,
        or ``None`` when no plan has been generated yet (pre-first-call or
        post-reset)

    Call ``reset()`` at the start of each new episode.

    Design note — two-call vs merged ``plan_critique`` + ``generate_subtask``:
    Every replan fires two VLM calls back-to-back (critique + subtask select).
    These could be merged into one call with a schema like
    ``{"statuses": [...], "subtask": "..."}``, which would halve cost + latency.
    We keep them separate because Comet's original flow depends on the
    sequential dependency — the subtask prompt is built from the *updated*
    ``plan_status`` *and* a possibly-reset ``subtask_history`` (triggered when
    the statuses change). Merging loses the reset gate and forces the model to
    maintain cross-field consistency in one generation, which is fine for
    strong reasoning models but risks inconsistent outputs on non-reasoning
    models (e.g. Gemini Robotics-ER at ``thinking_budget=0``). Revisit this
    if API cost becomes the bottleneck and the model can handle it.
    """

    def __init__(
        self,
        history_maxlen: int = DEFAULT_HISTORY_MAXLEN,
        sampled_images_max: int = DEFAULT_SAMPLED_IMAGES_MAX,
        history_stride: int = 5,
    ) -> None:
        self.history_multi_modals: collections.deque[np.ndarray] = collections.deque(
            maxlen=history_maxlen
        )
        self.sampled_images_max = sampled_images_max
        # Stride used by _sampled_history when picking frames from the
        # history deque. Default 5 matches Comet's original hardcoded value
        # and empirically gives the best plan stability on our 1 Hz cache
        # because the wider temporal span (~40 s of history with max=8)
        # provides the temporal-contrast signal the reasoner needs to
        # detect plan progression. stride=1 looks tempting ("consecutive
        # frames!") but flips plan interpretation on every tiny frame
        # change on slow tasks.
        self.history_stride = history_stride
        self.subtask_history: list[str] = []
        # Canonical plan state: the immutable step texts from the one-shot
        # generate_plan call and a parallel list of per-step statuses that
        # plan_critique updates. plan_status (the rendered string) is derived.
        self.plans: list[str] | None = None
        self.plan_statuses: list[PlanStepStatus] | None = None

    def reset(self) -> None:
        self.history_multi_modals.clear()
        self.subtask_history = []
        self.plans = None
        self.plan_statuses = None

    @property
    def plan_status(self) -> str | None:
        """Rendered Comet-style plan string, or None before generate_plan runs."""
        if self.plans is None or self.plan_statuses is None:
            return None
        return render_plan_status(self.plans, self.plan_statuses)

    @abc.abstractmethod
    def _chat(
        self,
        user_prompt: str,
        images: list[np.ndarray],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send a VLM request and return the raw response string.

        When ``response_schema`` is provided the backend must enforce it via
        its native structured-output mechanism (Gemini ``response_schema`` /
        vLLM ``json_schema``); otherwise the backend should free-form generate.

        Implementations should NOT strip ``<think>`` tags — the base class does
        that where appropriate.
        """

    def _sampled_history(self) -> list[np.ndarray]:
        return sample_images(
            list(self.history_multi_modals),
            max_len=self.sampled_images_max,
            stride=self.history_stride,
        )

    def generate_plan(self, task: str, initial_image: np.ndarray) -> str:
        """One-shot plan generation from the initial observation of the episode.

        Prompt adapted from ``openpi-comet/src/openpi/shared/client.py:274``,
        with an explicit JSON-array constraint enforced via ``PLAN_SCHEMA`` so
        off-the-shelf VLMs emit the structured list Comet's scaffold expects.

        Stores the parsed plan as ``self.plans`` and initializes
        ``self.plan_statuses`` with the first step in_progress, others
        not_started. Returns the rendered plan_status string.
        """
        user_prompt = (
            f"Given the task '{task}', break it down into several concrete high-level steps. "
            "Respond with a JSON array of 2-10 short imperative step strings, nothing else."
        )
        self.history_multi_modals.append(initial_image)
        response = self._chat(user_prompt, [initial_image], response_schema=PLAN_SCHEMA)
        plans = parse_plan_list(response)
        if plans is None:
            logger.warning(
                "Plan response could not be parsed as a non-empty JSON list of strings; "
                "falling back to single-step plan. Raw response: %r",
                response,
            )
            plans = [task]
        self.plans = plans
        self.plan_statuses = initial_statuses(len(plans))
        rendered = self.plan_status
        assert rendered is not None
        return rendered

    def plan_critique(self, task: str) -> list[PlanStepStatus]:
        """Ask the VLM to update per-step statuses given the image history.

        Returns a parallel list of ``PlanStepStatus`` values, one per step of
        ``self.plans`` in the same order. Enforced via ``CRITIQUE_SCHEMA`` so
        the backend returns discrete enum values rather than prose — this is
        load-bearing for the reset-on-change logic in ``generate_subtask``,
        which previously thrashed on tiny wording variations in free-form
        critique output.
        """
        if self.plans is None or self.plan_statuses is None:
            raise RuntimeError("plan_critique called before generate_plan — no plan to critique")

        last_subtask = self.subtask_history[-1] if self.subtask_history else "None"
        numbered_plan = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(self.plans))
        current_state = "\n".join(
            f"  {i + 1}. {status}" for i, status in enumerate(self.plan_statuses)
        )
        user_prompt = (
            f"You are given the task of '{task}'. The plan has these steps in order:\n"
            f"{numbered_plan}\n"
            f"\nTheir current statuses are:\n"
            f"{current_state}\n"
            f"\nThe last high-level objective given to the robot was '{last_subtask}'. "
            "Looking at the images, update each step's status to one of "
            f"{list(_STATUS_VALUES)}. Respond with a JSON object "
            '{"statuses": [...]} containing exactly '
            f"{len(self.plans)} status values in the same order as the steps."
        )
        response = self._chat(user_prompt, self._sampled_history(), response_schema=CRITIQUE_SCHEMA)
        new_statuses = _extract_statuses_field(response, expected_len=len(self.plans))
        if new_statuses is None:
            logger.warning(
                "plan_critique response was not a valid statuses list of length %d; "
                "keeping previous statuses. Raw response: %r",
                len(self.plans),
                response,
            )
            return list(self.plan_statuses)
        return new_statuses

    def generate_subtask(self, task: str, images: list[np.ndarray]) -> str:
        """Produce the next high-level subtask given the current observation.

        On the first call of an episode this also bootstraps the plan.
        Mirrors the control flow of
        ``openpi-comet/src/openpi/shared/client.py:294`` but avoids double-
        pushing the current image into the history deque and uses structural
        status comparison (not prose equality) to decide when to reset
        ``subtask_history``.

        Prompt adapted from ``openpi-comet/src/openpi/shared/client.py:305``;
        the subtask string is enforced via ``SUBTASK_SCHEMA`` to keep the
        output short enough for the pi0.5 action prompt budget.
        """
        if not images:
            raise ValueError("generate_subtask requires at least one image")

        if self.plans is None:
            # First frame of the episode — bootstrap the plan. generate_plan
            # already pushes initial_image into the history deque.
            self.generate_plan(task, images[0])
            self.subtask_history = []
            for extra in images[1:]:
                self.history_multi_modals.append(extra)
        else:
            for img in images:
                self.history_multi_modals.append(img)
            # Once the plan is fully complete, subsequent critique/subtask
            # calls produce degenerate output ("finished", "none") because
            # there is nothing left to plan. Skip the VLM calls entirely and
            # reuse the last real subtask so the action policy still gets a
            # meaningful prompt.
            if all_steps_done(self.plan_statuses):
                last_subtask = self.subtask_history[-1] if self.subtask_history else task
                self.subtask_history.append(last_subtask)
                return last_subtask
            if self.subtask_history:
                updated_statuses = self.plan_critique(task)
                if updated_statuses != self.plan_statuses:
                    self.subtask_history = []
                self.plan_statuses = updated_statuses

        # Unreachable in practice: generate_plan always sets plans/statuses,
        # and we took the "first frame" branch above when plans was None.
        assert self.plans is not None and self.plan_statuses is not None

        rendered_plan_status = render_plan_status(self.plans, self.plan_statuses)
        last_subtask = self.subtask_history[-1] if self.subtask_history else "None"
        user_prompt = (
            f"You are given the task of '{task}'. The status of the plans are:\n"
            f"    {rendered_plan_status}\n"
            f"    Note that [-] indicates in progress. [o] indicates completed. [x] indicates not started.\n"
            f"    The last high-level objective given to the robot was '{last_subtask}'."
            f"Based on your analysis, what should be the next high-level objective the robot should achieve? "
            'Respond with a JSON object {"subtask": "..."} where the value is '
            "a 3-6 word lowercase imperative phrase."
        )
        response = self._chat(user_prompt, self._sampled_history(), response_schema=SUBTASK_SCHEMA)
        subtask = _extract_subtask_field(response)
        self.subtask_history.append(subtask)
        return subtask


def _extract_subtask_field(response: str) -> str:
    """Pull the ``subtask`` field from a JSON object response.

    Falls back to the cleaned raw text if parsing fails or the field is
    missing so we never lose the data the model actually returned.
    """
    cleaned = strip_think_tags(response).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Subtask response was not valid JSON; using raw text. Got: %r", cleaned)
        return cleaned
    if isinstance(obj, dict) and isinstance(obj.get("subtask"), str):
        return obj["subtask"].strip()
    logger.warning("Subtask JSON missing 'subtask' string field; using raw text. Got: %r", obj)
    return cleaned


def _extract_statuses_field(response: str, expected_len: int) -> list[PlanStepStatus] | None:
    """Pull the ``statuses`` array from a JSON critique response.

    Returns a list of exactly ``expected_len`` validated ``PlanStepStatus``
    values, or ``None`` when parsing fails, the array is missing, lengths
    don't match, or any element isn't a known status. Callers should keep
    the prior plan_statuses on ``None`` rather than reset them.
    """
    cleaned = strip_think_tags(response).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    statuses = obj.get("statuses")
    if not isinstance(statuses, list) or len(statuses) != expected_len:
        return None
    validated: list[PlanStepStatus] = []
    for item in statuses:
        if item not in _STATUS_VALUES:
            return None
        validated.append(item)
    return validated
