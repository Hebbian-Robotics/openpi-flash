"""Runtime override helpers for serving-time PyTorch compile mode."""

from __future__ import annotations

import os
from typing import Literal, get_args

OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR = "OPENPI_PYTORCH_COMPILE_MODE"
DEFAULT_SERVING_PYTORCH_COMPILE_MODE = "default"

# Literal type for active compile modes (passed to torch.compile).
ActiveCompileMode = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]

# All valid env var values, including "none" which maps to None (eager mode).
ServingCompileMode = Literal[
    "none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
SUPPORTED_PYTORCH_COMPILE_MODES: tuple[str, ...] = get_args(ServingCompileMode)


def get_serving_pytorch_compile_mode() -> ActiveCompileMode | None:
    """Return the serving compile mode, honoring an env override when set.

    Returns None when set to "none", which disables torch.compile entirely
    and runs in eager mode.
    """
    configured_compile_mode = os.environ.get(
        OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR,
        DEFAULT_SERVING_PYTORCH_COMPILE_MODE,
    )
    active_modes: dict[str, ActiveCompileMode] = {
        "default": "default",
        "reduce-overhead": "reduce-overhead",
        "max-autotune": "max-autotune",
        "max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
    }
    if configured_compile_mode == "none":
        return None
    if configured_compile_mode in active_modes:
        return active_modes[configured_compile_mode]
    supported_compile_modes = ", ".join(repr(mode) for mode in SUPPORTED_PYTORCH_COMPILE_MODES)
    raise ValueError(
        f"Unsupported {OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR}="
        f"{configured_compile_mode!r}. Expected one of: {supported_compile_modes}."
    )
