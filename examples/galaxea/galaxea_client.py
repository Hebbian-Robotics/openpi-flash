"""Galaxea robot -> OpenPI hosted inference client.

A ROS 2 node that reads sensor data from a Galaxea R1-series robot, sends
observations to a remote OpenPI inference server over QUIC, and publishes
the returned actions back to the robot's joint command topics.

Supports both the R1 Pro (7-DOF per arm) and R1 Lite (6-DOF per arm).

See examples/galaxea/README.md for full setup instructions.

Usage:
    # R1 Pro with DROID mapping (recommended for Pro)
    uv run python examples/galaxea/galaxea_client.py \\
        --robot r1-pro \\
        --host 10.0.0.42 \\
        --prompt "pick up the red cup"

    # R1 Lite with ALOHA mapping (native 6-DOF fit, recommended for Lite)
    uv run python examples/galaxea/galaxea_client.py \\
        --robot r1-lite \\
        --host 10.0.0.42 \\
        --prompt "fold the towel" \\
        --mapping aloha
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy  # ty: ignore[unresolved-import]  # ROS 2 — not in this venv
from _common import (
    JOINTS_PER_ARM,
    GalaxeaTopicConfig,
    MappingName,
    RobotModel,
)
from embodiment_mappings import MAPPINGS_BY_ROBOT, get_topic_config
from rclpy.node import Node  # ty: ignore[unresolved-import]
from rclpy.qos import (  # ty: ignore[unresolved-import]
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, JointState  # ty: ignore[unresolved-import]
from shared.mappings import EmbodimentMapping
from shared.types import ActionCommand
from std_msgs.msg import Float32  # ty: ignore[unresolved-import]

from hosting.flash_transport_policy import FlashTransportPolicy

# Both R1 Pro and R1 Lite use BEST_EFFORT QoS for all sensor topics
GALAXEA_SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# ---------------------------------------------------------------------------
# Parsed CLI arguments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClientConfig:
    """Parsed and validated CLI arguments for the Galaxea client."""

    robot_model: RobotModel
    server_host: str
    server_port: int
    prompt: str
    mapping_name: MappingName
    inference_rate_hz: float

    @property
    def joints_per_arm(self) -> int:
        return JOINTS_PER_ARM[self.robot_model]

    @property
    def mapping(self) -> EmbodimentMapping:
        return MAPPINGS_BY_ROBOT[self.robot_model][self.mapping_name]

    @property
    def topic_config(self) -> GalaxeaTopicConfig:
        return get_topic_config(self.robot_model, self.mapping_name)


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class GalaxeaOpenPIClient(Node):
    """ROS 2 node bridging a Galaxea robot to a hosted OpenPI server."""

    def __init__(self, config: ClientConfig) -> None:
        node_name = f"galaxea_{config.robot_model.name.lower()}_openpi_client"
        super().__init__(node_name)

        self._prompt = config.prompt
        self._mapping = config.mapping
        self._topic_config = config.topic_config
        self._joints_per_arm = config.joints_per_arm

        # -- Locks for thread-safe access to latest sensor data ---------
        self._joint_lock = threading.Lock()
        self._image_lock = threading.Lock()

        # Latest sensor readings (None until first message arrives)
        self._left_arm_positions: np.ndarray | None = None
        self._right_arm_positions: np.ndarray | None = None
        self._left_gripper_position: float = 0.0
        self._right_gripper_position: float = 0.0
        self._latest_images: dict[str, np.ndarray] = {}

        # -- Connect to OpenPI server via QUIC --------------------------
        self.get_logger().info(
            f"Connecting to OpenPI server at {config.server_host}:{config.server_port} (QUIC)..."
        )
        self._policy = FlashTransportPolicy(
            host=config.server_host,
            port=config.server_port,
        )
        server_metadata = self._policy.get_server_metadata()
        self.get_logger().info(f"Connected. Server metadata: {server_metadata}")

        # -- Subscribe to joint state topics ----------------------------
        self._subscribe_to_joint_topics(self._topic_config)

        # -- Subscribe to camera topics --------------------------------
        for camera_name, ros_topic in self._topic_config.camera_topics.items():
            self.get_logger().info(f"Subscribing to camera: {camera_name} -> {ros_topic}")
            self.create_subscription(
                CompressedImage,
                ros_topic,
                lambda msg, name=camera_name: self._on_camera_image(name, msg),
                GALAXEA_SENSOR_QOS,
            )

        # -- Create publishers for action commands ---------------------
        self._left_arm_command_publisher = self.create_publisher(
            JointState,
            self._topic_config.left_arm_command_topic,
            10,
        )
        self._right_arm_command_publisher = self.create_publisher(
            JointState,
            self._topic_config.right_arm_command_topic,
            10,
        )
        self._left_gripper_command_publisher = self.create_publisher(
            Float32,
            self._topic_config.left_gripper_command_topic,
            10,
        )
        self._right_gripper_command_publisher = self.create_publisher(
            Float32,
            self._topic_config.right_gripper_command_topic,
            10,
        )

        # -- Inference timer -------------------------------------------
        inference_period_seconds = 1.0 / config.inference_rate_hz
        self._inference_timer = self.create_timer(
            inference_period_seconds,
            self._inference_loop,
        )
        self.get_logger().info(
            f"Inference loop started at {config.inference_rate_hz} Hz "
            f"(robot={config.robot_model}, mapping={type(self._mapping).__name__}, "
            f"joints_per_arm={self._joints_per_arm})"
        )

    def _subscribe_to_joint_topics(self, topic_config: GalaxeaTopicConfig) -> None:
        self.create_subscription(
            JointState,
            topic_config.left_arm_feedback_topic,
            self._on_left_arm_feedback,
            GALAXEA_SENSOR_QOS,
        )
        self.create_subscription(
            JointState,
            topic_config.right_arm_feedback_topic,
            self._on_right_arm_feedback,
            GALAXEA_SENSOR_QOS,
        )
        self.create_subscription(
            JointState,
            topic_config.left_gripper_feedback_topic,
            self._on_left_gripper_feedback,
            GALAXEA_SENSOR_QOS,
        )
        self.create_subscription(
            JointState,
            topic_config.right_gripper_feedback_topic,
            self._on_right_gripper_feedback,
            GALAXEA_SENSOR_QOS,
        )

    # -------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------
    def _on_left_arm_feedback(self, msg: JointState) -> None:
        with self._joint_lock:
            self._left_arm_positions = np.array(
                msg.position[: self._joints_per_arm], dtype=np.float64
            )

    def _on_right_arm_feedback(self, msg: JointState) -> None:
        with self._joint_lock:
            self._right_arm_positions = np.array(
                msg.position[: self._joints_per_arm], dtype=np.float64
            )

    def _on_left_gripper_feedback(self, msg: JointState) -> None:
        with self._joint_lock:
            self._left_gripper_position = float(msg.position[0]) if msg.position else 0.0

    def _on_right_gripper_feedback(self, msg: JointState) -> None:
        with self._joint_lock:
            self._right_gripper_position = float(msg.position[0]) if msg.position else 0.0

    def _on_camera_image(self, camera_name: str, msg: CompressedImage) -> None:
        image_array = np.frombuffer(msg.data, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded_image is None:
            self.get_logger().warn(f"Failed to decode image from {camera_name}")
            return
        decoded_image_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
        with self._image_lock:
            self._latest_images[camera_name] = decoded_image_rgb

    # -------------------------------------------------------------------
    # Inference loop
    # -------------------------------------------------------------------
    def _has_all_sensor_data(self) -> bool:
        if self._left_arm_positions is None or self._right_arm_positions is None:
            return False
        required_cameras = set(self._topic_config.camera_topics.keys())
        return required_cameras.issubset(self._latest_images.keys())

    def _inference_loop(self) -> None:
        if not self._has_all_sensor_data():
            self.get_logger().info(
                "Waiting for sensor data...",
                throttle_duration_sec=2.0,
            )
            return

        with self._joint_lock:
            left_arm = self._left_arm_positions.copy()
            right_arm = self._right_arm_positions.copy()
            left_gripper = self._left_gripper_position
            right_gripper = self._right_gripper_position

        with self._image_lock:
            images_snapshot = {name: img.copy() for name, img in self._latest_images.items()}

        observation = self._mapping.build_observation(
            left_arm_positions=left_arm,
            right_arm_positions=right_arm,
            left_gripper_position=left_gripper,
            right_gripper_position=right_gripper,
            images=images_snapshot,
            prompt=self._prompt,
        )

        server_response = self._policy.infer(observation)

        server_timing = server_response.get("server_timing")
        if server_timing:
            infer_ms = server_timing.get("infer_ms", "?")
            self.get_logger().debug(f"Server inference: {infer_ms}ms")

        action_command = self._mapping.unpack_actions(server_response)
        self._publish_action_command(action_command)

    def _publish_action_command(self, command: ActionCommand) -> None:
        left_arm_msg = JointState()
        left_arm_msg.position = command.left_arm_joint_positions.tolist()
        self._left_arm_command_publisher.publish(left_arm_msg)

        right_arm_msg = JointState()
        right_arm_msg.position = command.right_arm_joint_positions.tolist()
        self._right_arm_command_publisher.publish(right_arm_msg)

        left_gripper_msg = Float32()
        left_gripper_msg.data = command.left_gripper_position
        self._left_gripper_command_publisher.publish(left_gripper_msg)

        right_gripper_msg = Float32()
        right_gripper_msg.data = command.right_gripper_position
        self._right_gripper_command_publisher.publish(right_gripper_msg)

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    def destroy_node(self) -> None:
        self.get_logger().info("Shutting down...")
        self._policy.close()
        super().destroy_node()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments() -> ClientConfig:
    parser = argparse.ArgumentParser(
        description="Galaxea robot client for OpenPI hosted inference",
    )
    parser.add_argument(
        "--robot",
        type=RobotModel,
        required=True,
        choices=list(RobotModel),
        help="Robot model (e.g. r1-pro, r1-lite)",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="OpenPI server hostname or IP (e.g. 10.0.0.42)",
    )
    parser.add_argument(
        "--port",
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
        default=MappingName.DROID_DUAL_ARM,
        choices=list(MappingName),
        help="Embodiment mapping to use (default: droid-dual-arm)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=15.0,
        help="Inference rate in Hz (default: 15.0)",
    )
    args = parser.parse_args()

    return ClientConfig(
        robot_model=args.robot,
        server_host=args.host,
        server_port=args.port,
        prompt=args.prompt,
        mapping_name=args.mapping,
        inference_rate_hz=args.rate,
    )


def main() -> None:
    config = parse_arguments()

    rclpy.init()
    node = GalaxeaOpenPIClient(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
