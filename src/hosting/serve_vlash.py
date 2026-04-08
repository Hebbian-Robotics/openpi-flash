"""VLASH hosted inference server entry point.

Loads a VLASH policy and serves it over WebSocket with async chunk
pre-computation. See vlash_server.py for the protocol details.

Usage:
    INFERENCE_CONFIG_PATH=config.vlash.json python -m hosting.serve_vlash
"""

import logging
import socket

import torch
from vlash.policies.factory import get_policy_class
from vlash.run import warmup_compiled_policy

from hosting.serve import setup_logging
from hosting.vlash_config import load_vlash_config
from hosting.vlash_server import VlashPolicyServer

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    config = load_vlash_config()
    logger.info(
        "Loading VLASH model: type=%s, path=%s",
        config.policy_type,
        config.pretrained_path,
    )

    # Load VLASH policy.
    policy_cls = get_policy_class(config.policy_type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=config.pretrained_path)
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    logger.info("Policy loaded on device=%s", device)

    # Optional: warmup compiled policy to trigger torch.compile.
    if config.compile_model:
        warmup_compiled_policy(policy, config.task)

    # Build metadata sent to client on connection.
    policy_metadata = {
        "policy_type": config.policy_type,
        "model_version": config.model_version,
        "n_action_steps": policy.config.n_action_steps,
    }

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating VLASH hosted server (host=%s, ip=%s)", hostname, local_ip)

    server = VlashPolicyServer(
        policy=policy,
        device=device,
        service_config=config,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
