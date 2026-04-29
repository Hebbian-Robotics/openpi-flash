"""OpenAI-compatible VLM backend for Comet-style hierarchical subtask generation.

Targets a self-hosted vLLM server (or any server that speaks the OpenAI
chat-completions protocol with image inputs). Default model name matches the
only VLM named in the openpi-comet repo, ``Qwen3-VL-30B-A3B-Instruct``
(``openpi-comet/src/openpi/shared/client.py:169``), but any multimodal chat
model the server hosts will work.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, cast

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ._gemini_utils import encode_png
from .reasoner_base import BaseReasoner

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def _encode_data_url(image: np.ndarray) -> str:
    """PNG-encode an image as a ``data:`` URL suitable for OpenAI image inputs."""
    png_bytes = encode_png(image)
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"


class OpenAICompatReasoner(BaseReasoner):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = "none",
        temperature: float = 1.0,
        timeout_s: float = 600.0,
        history_maxlen: int = 640,
        sampled_images_max: int = 64,
        history_stride: int = 5,
    ) -> None:
        super().__init__(
            history_maxlen=history_maxlen,
            sampled_images_max=sampled_images_max,
            history_stride=history_stride,
        )
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model
        self._temperature = temperature

    def _chat(
        self,
        user_prompt: str,
        images: list[np.ndarray],
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_data_url(image)},
                }
            )
        # The chat-completions SDK types are strict TypedDicts that don't
        # accept our general-purpose content list; cast is safer than
        # maintaining parallel TypedDict literals for every image.
        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": content}])

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if response_schema is not None:
            # vLLM exposes OpenAI-spec structured outputs backed by xgrammar.
            # The server fills in decoding constraints from the JSON schema so
            # the response is guaranteed to parse.
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        response = self._client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message.content or ""
        return message.strip()
