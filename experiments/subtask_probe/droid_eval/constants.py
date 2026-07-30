"""Shared constants for the DROID subtask evaluation pipeline."""

from typing import Literal

# Inference mode — finite set of valid values for the "mode" field in observation dicts.
# "subtask_only": generate subtask text only (no action generation)
# "action_only": generate actions only (skip subtask generation)
InferenceMode = Literal["subtask_only", "action_only"]

# DROID action dimensions
ACTION_HORIZON = 15  # Number of future action steps predicted per frame
DROID_ACTION_DIM = 8  # 7 joints + 1 gripper
MODEL_ACTION_DIM = 32  # pi0.5 internal latent action dimension

# Evaluation conditions
CONDITION_NAMES = ["baseline", "subtask"]

# Joint names for per-dimension metrics and visualization
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]

# Gripper state threshold (values above = closed, below = open)
GRIPPER_THRESHOLD = 0.5

# Visualization colors per condition
CONDITION_COLORS = {
    "baseline": "#4a90d9",
    "subtask": "#d94a4a",
    "ground_truth": "#888888",
}

# Default server ports
DEFAULT_QUIC_PORT = 5555
