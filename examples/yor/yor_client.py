"""YOR robot -> OpenPI hosted inference client.

A simple polling loop that reads sensor data from a YOR robot via
``commlink`` ZMQ RPC, sends observations to a remote OpenPI inference server
over QUIC, and writes the returned actions back to the robot's joints.

YOR is a 6-DOF bimanual mobile manipulator from NYU GRAIL.
See https://github.com/YOR-robot/YOR for robot setup.

Usage:
    # ALOHA mapping (recommended — native 6-DOF fit)
    uv run python examples/yor/yor_client.py \\
        --robot-host 10.0.0.10 \\
        --server-host 10.0.0.42 \\
        --prompt "pick up the red cup"

    # DROID mapping
    uv run python examples/yor/yor_client.py \\
        --robot-host 10.0.0.10 \\
        --server-host 10.0.0.42 \\
        --prompt "pick up the red cup" \\
        --mapping droid-dual-arm

    # With MuJoCo simulation (localhost)
    uv run python examples/yor/yor_client.py \\
        --server-host 10.0.0.42 \\
        --prompt "pick up the red cup"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Allow bare imports from examples/ for the shared module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.mappings import AlohaMapping, DroidDualArmMapping, EmbodimentMapping
from shared.types import MappingName

from hosting.flash_transport_policy import FlashTransportPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOR-specific constants
# ---------------------------------------------------------------------------
YOR_JOINTS_PER_ARM = 6
YOR_RPC_PORT = 5557

# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------
# YOR has a ZED stereo camera on the head.  Wrist cameras are optional and
# depend on the specific build.  Adjust these to match your hardware.
#
# The keys are logical camera names used by the mapping; the values are
# OpenCV device indices or paths (e.g. "/dev/video0").

DEFAULT_CAMERA_DEVICES: dict[str, int | str] = {
    "cam_high": 0,  # ZED head camera (left eye)
}

# ---------------------------------------------------------------------------
# Mapping registry
# ---------------------------------------------------------------------------
MAPPINGS: dict[MappingName, EmbodimentMapping] = {
    MappingName.ALOHA: AlohaMapping(
        camera_names_config=("cam_high",),
        use_dummy_cam_low=True,  # servers expecting 4 ALOHA cameras get a black placeholder
    ),
    MappingName.DROID_DUAL_ARM: DroidDualArmMapping(
        joints_per_arm=YOR_JOINTS_PER_ARM,
        camera_observation_keys={
            "cam_high": "observation/exterior_image_1_left",
        },
    ),
}


# ---------------------------------------------------------------------------
# Parsed CLI arguments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClientConfig:
    """Parsed and validated CLI arguments for the YOR client."""

    robot_host: str
    robot_port: int
    server_host: str
    server_port: int
    prompt: str
    mapping_name: MappingName
    inference_rate_hz: float
    camera_devices: dict[str, int | str]

    @property
    def mapping(self) -> EmbodimentMapping:
        return MAPPINGS[self.mapping_name]


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------
class CameraCapture:
    """Manages OpenCV video captures for the robot's cameras."""

    def __init__(self, camera_devices: dict[str, int | str]) -> None:
        self._captures: dict[str, cv2.VideoCapture] = {}
        for name, device in camera_devices.items():
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():
                logger.warning("Camera %s (device %s) failed to open", name, device)
            else:
                logger.info("Camera %s opened on device %s", name, device)
            self._captures[name] = cap

    def read(self) -> dict[str, np.ndarray]:
        """Capture one frame from each camera, returning HWC RGB uint8 arrays."""
        images: dict[str, np.ndarray] = {}
        for name, cap in self._captures.items():
            ret, frame = cap.read()
            if ret:
                images[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.warning("Failed to read frame from camera %s", name)
        return images

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class YOROpenPIClient:
    """Bridges a YOR robot to a hosted OpenPI server.

    Unlike the Galaxea client (which is a ROS 2 node), YOR uses ``commlink``
    ZMQ RPC for robot communication — so this is a simple polling loop.
    """

    def __init__(self, config: ClientConfig) -> None:
        from commlink import RPCClient  # ty: ignore[unresolved-import]

        self._prompt = config.prompt
        self._mapping = config.mapping
        self._inference_rate_hz = config.inference_rate_hz

        # Connect to YOR robot via ZMQ RPC
        logger.info("Connecting to YOR robot at %s:%d ...", config.robot_host, config.robot_port)
        self._yor = RPCClient(host=config.robot_host, port=config.robot_port)
        self._yor.init()
        logger.info("Connected to YOR robot")

        # Connect to OpenPI server via QUIC
        logger.info(
            "Connecting to OpenPI server at %s:%d (QUIC)...",
            config.server_host,
            config.server_port,
        )
        self._policy = FlashTransportPolicy(
            host=config.server_host,
            port=config.server_port,
        )
        server_metadata = self._policy.get_server_metadata()
        logger.info("Connected to OpenPI. Server metadata: %s", server_metadata)

        # Open cameras
        self._cameras = CameraCapture(config.camera_devices)

    def run(self) -> None:
        """Run the inference loop until interrupted."""
        period = 1.0 / self._inference_rate_hz
        logger.info(
            "Starting inference loop at %.1f Hz (mapping=%s)",
            self._inference_rate_hz,
            type(self._mapping).__name__,
        )

        try:
            while True:
                loop_start = time.monotonic()

                # 1. Read joint state from robot
                left_joints = np.asarray(self._yor.get_left_joint_positions(), dtype=np.float64)
                right_joints = np.asarray(self._yor.get_right_joint_positions(), dtype=np.float64)
                left_gripper = float(self._yor.get_left_gripper_pose())
                right_gripper = float(self._yor.get_right_gripper_pose())

                # 2. Capture camera images
                images = self._cameras.read()
                expected_cameras = set(self._mapping.camera_names)
                if not expected_cameras.issubset(images.keys()):
                    missing = expected_cameras - images.keys()
                    logger.warning("Missing camera frames: %s — skipping cycle", missing)
                    time.sleep(period)
                    continue

                # 3. Build observation
                observation = self._mapping.build_observation(
                    left_arm_positions=left_joints,
                    right_arm_positions=right_joints,
                    left_gripper_position=left_gripper,
                    right_gripper_position=right_gripper,
                    images=images,
                    prompt=self._prompt,
                )

                # 4. Run inference
                server_response = self._policy.infer(observation)

                server_timing = server_response.get("server_timing")
                if server_timing:
                    infer_ms = server_timing.get("infer_ms", "?")
                    logger.debug("Server inference: %sms", infer_ms)

                # 5. Send actions back to robot
                action_command = self._mapping.unpack_actions(server_response)
                self._yor.set_left_joint_target(
                    action_command.left_arm_joint_positions,
                    action_command.left_gripper_position,
                    preview_time=0.1,
                )
                self._yor.set_right_joint_target(
                    action_command.right_arm_joint_positions,
                    action_command.right_gripper_position,
                    preview_time=0.1,
                )

                # 6. Rate limit
                elapsed = time.monotonic() - loop_start
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")

    def close(self) -> None:
        self._cameras.close()
        self._policy.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments() -> ClientConfig:
    parser = argparse.ArgumentParser(
        description="YOR robot client for OpenPI hosted inference",
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default="localhost",
        help="YOR robot hostname or IP (default: localhost for MuJoCo sim)",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=YOR_RPC_PORT,
        help=f"YOR robot RPC port (default: {YOR_RPC_PORT})",
    )
    parser.add_argument(
        "--server-host",
        type=str,
        required=True,
        help="OpenPI server hostname or IP (e.g. 10.0.0.42)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5555,
        help="OpenPI server QUIC port (default: 5555)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="do something",
        help="Language instruction for the policy (default: 'do something')",
    )
    parser.add_argument(
        "--mapping",
        type=MappingName,
        default=MappingName.ALOHA,
        choices=list(MappingName),
        help="Embodiment mapping to use (default: aloha)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=15.0,
        help="Inference rate in Hz (default: 15.0)",
    )
    args = parser.parse_args()

    return ClientConfig(
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        server_host=args.server_host,
        server_port=args.server_port,
        prompt=args.prompt,
        mapping_name=args.mapping,
        inference_rate_hz=args.rate,
        camera_devices=DEFAULT_CAMERA_DEVICES,
    )


def main() -> None:
    config = parse_arguments()
    client = YOROpenPIClient(config)

    try:
        client.run()
    finally:
        client.close()


if __name__ == "__main__":
    main()
