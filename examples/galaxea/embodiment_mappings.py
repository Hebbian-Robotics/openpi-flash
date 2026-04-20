"""Galaxea R1-series robot -> OpenPI mapping configurations.

This module instantiates the shared ``AlohaMapping`` and ``DroidDualArmMapping``
classes with Galaxea-specific camera and topic configurations, and provides
the ``MAPPINGS_BY_ROBOT`` registry used by the Galaxea client.

Robot-specific differences (camera layout, ROS topic names, joints per arm)
are captured here — the mapping logic itself lives in ``shared.mappings``.
"""

from __future__ import annotations

from _common import (
    JOINTS_PER_ARM,
    GalaxeaTopicConfig,
    MappingName,
    RobotModel,
)
from shared.mappings import (
    AlohaMapping,
    DroidDualArmMapping,
    EmbodimentMapping,
)

# ---------------------------------------------------------------------------
# Camera ROS 2 topic configurations per robot model and mapping
# ---------------------------------------------------------------------------

# Maps (RobotModel, MappingName) -> {camera_name: ros_topic_string}
CAMERA_ROS_TOPICS: dict[tuple[RobotModel, MappingName], dict[str, str]] = {
    (RobotModel.R1_PRO, MappingName.DROID_DUAL_ARM): {
        "cam_head": "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
        "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    },
    (RobotModel.R1_PRO, MappingName.ALOHA): {
        "cam_high": "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "cam_low": "/hdas/camera_chassis_front_left/rgb/compressed",
        "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
        "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    },
    (RobotModel.R1_LITE, MappingName.DROID_DUAL_ARM): {
        "cam_head_left": "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "cam_head_right": "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
        "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    },
    (RobotModel.R1_LITE, MappingName.ALOHA): {
        "cam_high": "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "cam_left_wrist": "/hdas/camera_wrist_left/color/image_raw/compressed",
        "cam_right_wrist": "/hdas/camera_wrist_right/color/image_raw/compressed",
    },
}


def get_topic_config(robot_model: RobotModel, mapping_name: MappingName) -> GalaxeaTopicConfig:
    """Build the full ROS 2 topic configuration for a specific robot + mapping combo."""
    return GalaxeaTopicConfig(
        camera_topics=CAMERA_ROS_TOPICS[(robot_model, mapping_name)],
    )


# ---------------------------------------------------------------------------
# Registry — per-robot mapping instances
# ---------------------------------------------------------------------------
MAPPINGS_BY_ROBOT: dict[RobotModel, dict[MappingName, EmbodimentMapping]] = {
    RobotModel.R1_PRO: {
        MappingName.DROID_DUAL_ARM: DroidDualArmMapping(
            joints_per_arm=JOINTS_PER_ARM[RobotModel.R1_PRO],
            camera_observation_keys={
                "cam_head": "observation/exterior_image_1_left",
                "cam_left_wrist": "observation/wrist_image_left",
                "cam_right_wrist": "observation/wrist_image_right",
            },
        ),
        MappingName.ALOHA: AlohaMapping(
            camera_names_config=(
                "cam_high",
                "cam_low",
                "cam_left_wrist",
                "cam_right_wrist",
            ),
            dropped_joint_index=0,
        ),
    },
    RobotModel.R1_LITE: {
        MappingName.DROID_DUAL_ARM: DroidDualArmMapping(
            joints_per_arm=JOINTS_PER_ARM[RobotModel.R1_LITE],
            camera_observation_keys={
                "cam_head_left": "observation/exterior_image_1_left",
                "cam_head_right": "observation/exterior_image_2_left",
                "cam_left_wrist": "observation/wrist_image_left",
                "cam_right_wrist": "observation/wrist_image_right",
            },
        ),
        MappingName.ALOHA: AlohaMapping(
            camera_names_config=(
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ),
        ),
    },
}
