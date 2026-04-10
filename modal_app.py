"""Modal deployment for openpi inference.

Serves an openpi policy over WebSocket using Modal's GPU infrastructure.
Supports both JAX and PyTorch checkpoints (auto-detected).

Usage:
    modal serve modal_app.py    # development (hot-reload)
    modal deploy modal_app.py   # production
"""

from typing import Any

import modal

from hosting.modal_helpers import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MODEL_CONFIG_NAME,
    GPU_TYPE,
    MODEL_CACHE_MOUNT_PATH,
    REGION,
    create_openpi_image,
    model_weights_volume,
    prepare_openpi_config,
)

app = modal.App("openpi-inference")

openpi_image = create_openpi_image()


@app.cls(
    image=openpi_image,
    gpu=GPU_TYPE,
    region=REGION,
    volumes={MODEL_CACHE_MOUNT_PATH: model_weights_volume},
    scaledown_window=300,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)
class OpenPIInference:
    model_config_name: str = modal.parameter(default=DEFAULT_MODEL_CONFIG_NAME)
    checkpoint_dir: str = modal.parameter(default=DEFAULT_CHECKPOINT_DIR)
    model_version: str = modal.parameter(default="pi05_v1")
    default_prompt: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def prepare_config(self) -> None:
        """CPU-only phase: patches, imports, config. Captured in memory snapshot."""
        self._train_config = prepare_openpi_config(self.model_config_name)

    @modal.enter(snap=False)
    def load_model(self) -> None:
        """GPU phase: load weights and run warmup. Runs after snapshot restore."""
        from hosting.modal_helpers import load_openpi_model

        self._policy, train_config = load_openpi_model(
            self._train_config,
            self.checkpoint_dir,
            self.default_prompt,
        )
        self._metadata = train_config.policy_metadata

        # Persist inductor cache so subsequent cold starts skip compilation.
        model_weights_volume.commit()
        print("Inductor cache committed to volume")

    @modal.asgi_app()
    def serve(self) -> Any:
        from hosting.modal_asgi import create_openpi_asgi_app

        return create_openpi_asgi_app(
            policy=self._policy,
            metadata=self._metadata,
        )
