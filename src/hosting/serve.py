"""Hosted inference server entry point.

Loads an openpi policy and serves it over WebSocket using the built-in
WebsocketPolicyServer. Supports both JAX and PyTorch checkpoints
(auto-detected by create_trained_policy).

Usage:
    INFERENCE_CONFIG_PATH=config.json python -m hosting.serve
"""

import dataclasses
import logging
import time

import numpy as np
from openpi.models import pi0_config as _pi0_config
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config

from hosting.config import load_config

logger = logging.getLogger(__name__)

# Compile mode "default" is proven reliable on Modal (~2.5 min compile, ~76ms
# inference). The config default "max-autotune" optimises for training
# throughput and takes longer to compile with no serving-speed benefit.
PYTORCH_COMPILE_MODE = "default"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    service_config = load_config()
    logger.info(
        "Loading model: config=%s, checkpoint=%s",
        service_config.model_config_name,
        service_config.checkpoint_dir,
    )

    train_config = _config.get_config(service_config.model_config_name)

    # Override compile mode for faster, reliable compilation during serving.
    if isinstance(train_config.model, _pi0_config.Pi0Config):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, pytorch_compile_mode=PYTORCH_COMPILE_MODE
            ),
        )

    load_start = time.monotonic()
    policy = _policy_config.create_trained_policy(
        train_config,
        service_config.checkpoint_dir,
        default_prompt=service_config.default_prompt,
    )
    load_elapsed = time.monotonic() - load_start
    logger.info("Model loaded in %.1fs", load_elapsed)

    # Warmup inference triggers torch.compile and populates the inductor cache
    # so the first real client request doesn't block for minutes.
    warmup_observation = {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "warmup",
    }
    logger.info("Compiling model (warmup inference) ...")
    compile_start = time.monotonic()
    policy.infer(warmup_observation)
    compile_elapsed = time.monotonic() - compile_start
    logger.info("Compilation done in %.1fs", compile_elapsed)

    server = WebsocketPolicyServer(
        policy=policy,
        port=service_config.port,
        metadata=train_config.policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
