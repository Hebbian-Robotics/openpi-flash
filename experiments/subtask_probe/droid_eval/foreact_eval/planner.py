"""ForeAct VLM planner with the paper's Table 5 two-turn prompt.

Ported verbatim from Appendix A.5 of the ForeAct paper (arxiv 2602.12322).
The prompt strings are part of the method — do not reword them.

Per-episode state is minimal: just ``previous_subtask: str | None``. The
"reason-execute-monitor" cycle relies on the VLM re-deriving the plan
latently each turn; there is no explicit plan list or status marker.

We additionally enforce a JSON schema on the response. The paper only says
"concise and deterministic", but our Comet experience showed free-form output
adds 500ms-3s of latency and causes semantic drift. The ``subtask`` field is
what the paper actually uses; the ``previous_finished`` bool is ours, added
purely for observability (so we can log when the planner advances).
"""

from __future__ import annotations

import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from experiments.subtask_probe.droid_eval.comet_style._gemini_utils import (
    call_with_retry,
    encode_png,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts (Table 5, Appendix A.5 of foreact.pdf). Only ``{task}`` substitutes.
# ---------------------------------------------------------------------------

INITIAL_PROMPT_TEMPLATE = """\
You are a robot controller. Please plan to finish the task in several steps. \
And give instruction for each step in a concise way.
The task is to "{task}".

RULES:
\u2022 During the job, I will continuously give you an observation image of the current state.
\u2022 Based on the observation, please judge if the last instruction has been finished.
    - If yes, give me the instruction for the next step.
    - If no, repeat the instruction of the ongoing subtask.
\u2022 You're not required to describe the observation. Only output the instruction for each subtask.

Now, you are only required to output instruction for the first step."""


FOLLOW_UP_PROMPT_TEMPLATE = """\
Pay attention to the latest observation. Firstly, judge if the last instruction \
has been finished. Secondly, if yes, give me the instruction for the next step; \
if no, repeat the instruction of the ongoing subtask.
Your answer should be concise and deterministic.
Remember, your Overall Task is "{task}"."""


SUBTASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subtask": {"type": "string"},
        "previous_finished": {"type": "boolean"},
    },
    "required": ["subtask", "previous_finished"],
    "additionalProperties": False,
}


PromptPhase = Literal["initial", "follow_up"]
ImagePosition = Literal["start", "end"]


# ---------------------------------------------------------------------------
# Base planner with the per-episode state + dispatch logic
# ---------------------------------------------------------------------------


class BasePlanner(ABC):
    """Stateful per-episode ForeAct planner; subclasses implement the VLM call.

    ``use_schema`` controls whether we force ``{"subtask": str,
    "previous_finished": bool}`` JSON output via the backend's response-format
    hook. The paper (Table 5, §3.3) doesn't mention any schema — it only asks
    for "concise and deterministic" free-form text. Our schema is an addition
    that helped the Comet-style reasoner but may be hurting step-level
    decomposition here (see FINDINGS). Set ``False`` to reproduce the paper
    more literally.
    """

    def __init__(self, use_schema: bool = True) -> None:
        self.previous_subtask: str | None = None
        self._use_schema = use_schema

    def reset(self) -> None:
        self.previous_subtask = None

    def generate_subtask(self, task: str, current_image: np.ndarray) -> dict[str, Any]:
        """Run one planner turn and update ``previous_subtask`` in place.

        Returns ``{"subtask": str, "previous_finished": bool, "prompt_phase": str}``.
        On parse failure returns empty subtask_text + previous_finished=False so
        the outer loop can keep marching without crashing the episode.
        """
        if self.previous_subtask is None:
            prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
            # Table 5: "VISUAL INPUT: [Initial Observation Image]" appears at
            # the END of the initial prompt block.
            image_position: ImagePosition = "end"
            prompt_phase: PromptPhase = "initial"
        else:
            prompt = FOLLOW_UP_PROMPT_TEMPLATE.format(task=task)
            # Table 5: "VISUAL INPUT: [Current Observation Image]" appears at
            # the START of the follow-up prompt block.
            image_position = "start"
            prompt_phase = "follow_up"

        raw = self._chat(
            prompt=prompt,
            image=current_image,
            image_position=image_position,
            response_schema=SUBTASK_SCHEMA if self._use_schema else None,
        )
        parsed = (
            _parse_response_json(raw)
            if self._use_schema
            else _parse_response_freeform(raw, self.previous_subtask)
        )
        if parsed["subtask"]:
            self.previous_subtask = parsed["subtask"]
        return {**parsed, "prompt_phase": prompt_phase}

    @abstractmethod
    def _chat(
        self,
        prompt: str,
        image: np.ndarray,
        image_position: ImagePosition,
        response_schema: dict[str, Any] | None,
    ) -> str:
        """Return the raw VLM output — JSON-shape when ``response_schema`` is set,
        free-form text when it is ``None``."""


def _parse_response_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Planner response was not valid JSON: %r", raw[:200])
        return {"subtask": "", "previous_finished": False}
    subtask = str(payload.get("subtask") or "").strip()
    previous_finished = bool(payload.get("previous_finished", False))
    return {"subtask": subtask, "previous_finished": previous_finished}


def _parse_response_freeform(raw: str, previous_subtask: str | None) -> dict[str, Any]:
    """Parse a free-form VLM response.

    The paper's prompt asks for a concise instruction with no other text, so
    the common case is a single short sentence. We strip surrounding quotes /
    whitespace and, as a safety net, take the last non-empty line if the model
    was chatty. ``previous_finished`` is inferred from whether the new subtask
    string differs from ``previous_subtask`` — this matches the paper's
    semantics where "if finished, give the next step; if not, repeat the
    current one".
    """
    cleaned = raw.strip().strip("\"'`")
    lines = [line.strip().strip("\"'`") for line in cleaned.splitlines() if line.strip()]
    subtask = lines[-1] if lines else ""
    previous_finished = bool(
        previous_subtask is not None and subtask and subtask != previous_subtask
    )
    return {"subtask": subtask, "previous_finished": previous_finished}


# ---------------------------------------------------------------------------
# Backend A — OpenAI-compatible (targets local vLLM hosting Qwen3-VL-8B-Instruct,
# which is the exact model the paper uses for both VLM+\u03c0_0 and ForeAct \u00a74.3)
# ---------------------------------------------------------------------------


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class OpenAICompatPlanner(BasePlanner):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str = "none",
        temperature: float = 1.0,
        timeout_s: float = 600.0,
        use_schema: bool = True,
    ) -> None:
        super().__init__(use_schema=use_schema)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._temperature = temperature
        # Conversation history for the reason-execute-monitor cycle. Each
        # turn appends (user_message_text_only, assistant_response). We strip
        # image parts from the stored user message — keeping 57 base64-
        # encoded images in context would blow past Qwen3-VL's 32k window
        # within ~20 turns. The model's plan persistence relies on its own
        # earlier assistant text, not on re-observing previous frames.
        self._conversation: list[ChatCompletionMessageParam] = []

    def reset(self) -> None:
        super().reset()
        self._conversation = []

    def _chat(
        self,
        prompt: str,
        image: np.ndarray,
        image_position: ImagePosition,
        response_schema: dict[str, Any] | None,
    ) -> str:
        png_bytes = encode_png(image)
        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
        image_part: dict[str, Any] = {"type": "image_url", "image_url": {"url": data_url}}
        text_part: dict[str, Any] = {"type": "text", "text": prompt}
        content = [text_part, image_part] if image_position == "end" else [image_part, text_part]
        # Active call sees: full conversation history + the current turn
        # (with the CURRENT image). Previous turns are text-only in history.
        current_user_msg = cast(
            ChatCompletionMessageParam, {"role": "user", "content": content}
        )
        messages: list[ChatCompletionMessageParam] = [*self._conversation, current_user_msg]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "foreact_subtask",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        response = self._client.chat.completions.create(**kwargs)
        assistant_text = (response.choices[0].message.content or "").strip()

        # Persist the turn to history. Strip the image part to keep context
        # cheap; the assistant's plan lives in the text it emits.
        user_text_only = cast(
            ChatCompletionMessageParam, {"role": "user", "content": prompt}
        )
        assistant_msg = cast(
            ChatCompletionMessageParam, {"role": "assistant", "content": assistant_text}
        )
        self._conversation.append(user_text_only)
        self._conversation.append(assistant_msg)

        return assistant_text


# ---------------------------------------------------------------------------
# Backend B — Gemini (optional; the paper doesn't use it, but we have keys set
# up from the Comet work and it's a cheap apples-to-apples vs. our prior runs)
# ---------------------------------------------------------------------------


DEFAULT_GEMINI_MODEL = "gemini-robotics-er-1.6-preview"


class GeminiPlanner(BasePlanner):
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        thinking_budget: int = 0,
        max_retries: int = 10,
        request_timeout_s: float = 120.0,
        use_schema: bool = True,
    ) -> None:
        super().__init__(use_schema=use_schema)
        # google-genai is an optional dependency for this backend only —
        # import lazily so OpenAI-only users don't need it installed.
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            http_options=types.HttpOptions(timeout=int(request_timeout_s * 1000))
        )
        self._model = model
        self._thinking_budget = thinking_budget
        self._max_retries = max_retries

    def _chat(
        self,
        prompt: str,
        image: np.ndarray,
        image_position: ImagePosition,
        response_schema: dict[str, Any] | None,
    ) -> str:
        types = self._types
        image_part = types.Part.from_bytes(data=encode_png(image), mime_type="image/png")
        contents: list[Any] = (
            [prompt, image_part] if image_position == "end" else [image_part, prompt]
        )
        config_kwargs: dict[str, Any] = {
            "temperature": 1.0,
            "thinking_config": types.ThinkingConfig(thinking_budget=self._thinking_budget),
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            return (response.text or "").strip()

        return call_with_retry(_call, max_retries=self._max_retries)
