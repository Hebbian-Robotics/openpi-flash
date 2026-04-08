"""Hosted inference server entry point.

Loads an openpi policy and serves it over WebSocket using the built-in
WebsocketPolicyServer. Supports both JAX and PyTorch checkpoints
(auto-detected by create_trained_policy).

Usage:
    INFERENCE_CONFIG_PATH=config.json python -m hosting.serve
"""

import logging

from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config

from hosting.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    service_config = load_config()
    logger.info(
        "Loading model: config=%s, checkpoint=%s",
        service_config.model_config_name,
        service_config.checkpoint_dir,
    )

    train_config = _config.get_config(service_config.model_config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        service_config.checkpoint_dir,
        default_prompt=service_config.default_prompt,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        port=service_config.port,
        metadata=train_config.policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
