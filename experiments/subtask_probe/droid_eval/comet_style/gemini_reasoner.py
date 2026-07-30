"""Gemini Robotics-ER backed reasoner for Comet-style hierarchical subtask generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from ._gemini_utils import call_with_retry, encode_png
from .reasoner_base import BaseReasoner

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"


class GeminiReasoner(BaseReasoner):
    """Plan/critique/subtask loop backed by Gemini Robotics-ER 1.6 Preview.

    Accepts any Gemini model via ``model=`` — e.g. ``gemini-3.1-pro-preview``
    if you want a general-purpose reasoning VLM instead of the robotics-tuned
    default.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        thinking_budget: int = 0,
        max_retries: int = 10,
        request_timeout_s: float = 120.0,
        history_maxlen: int = 640,
        sampled_images_max: int = 64,
        history_stride: int = 5,
    ) -> None:
        super().__init__(
            history_maxlen=history_maxlen,
            sampled_images_max=sampled_images_max,
            history_stride=history_stride,
        )
        # timeout is in milliseconds per google-genai's HttpOptions. A 2-minute
        # cap is well above normal ~5-10s latency but short enough that a hung
        # socket surfaces fast instead of stalling the whole run.
        self._client = genai.Client(
            http_options=types.HttpOptions(timeout=int(request_timeout_s * 1000))
        )
        self._model = model
        self._thinking_budget = thinking_budget
        self._max_retries = max_retries

    def _chat(
        self,
        user_prompt: str,
        images: list[np.ndarray],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        contents: list = []
        for image in images:
            contents.append(types.Part.from_bytes(data=encode_png(image), mime_type="image/png"))
        contents.append(user_prompt)

        config_kwargs: dict[str, Any] = {
            "temperature": 1.0,
            "thinking_config": types.ThinkingConfig(thinking_budget=self._thinking_budget),
        }
        if response_schema is not None:
            # The google-genai SDK accepts a dict JSON schema directly and
            # converts it to a types.Schema internally. Pairing with
            # response_mime_type="application/json" enforces the structure.
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
