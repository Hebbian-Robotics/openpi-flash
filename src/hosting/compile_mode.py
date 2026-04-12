"""Runtime override helpers for serving-time PyTorch compile mode."""

from __future__ import annotations

import os

OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR = "OPENPI_PYTORCH_COMPILE_MODE"
DEFAULT_SERVING_PYTORCH_COMPILE_MODE = "default"
SUPPORTED_PYTORCH_COMPILE_MODES = (
    "none",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def get_serving_pytorch_compile_mode() -> str | None:
    """Return the serving compile mode, honoring an env override when set.

    Returns None when set to "none", which disables torch.compile entirely
    and runs in eager mode.
    """
    configured_compile_mode = os.environ.get(
        OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR,
        DEFAULT_SERVING_PYTORCH_COMPILE_MODE,
    )
    if configured_compile_mode not in SUPPORTED_PYTORCH_COMPILE_MODES:
        supported_compile_modes = ", ".join(repr(mode) for mode in SUPPORTED_PYTORCH_COMPILE_MODES)
        raise ValueError(
            f"Unsupported {OPENPI_PYTORCH_COMPILE_MODE_ENV_VAR}="
            f"{configured_compile_mode!r}. Expected one of: {supported_compile_modes}."
        )
    if configured_compile_mode == "none":
        return None
    return configured_compile_mode
