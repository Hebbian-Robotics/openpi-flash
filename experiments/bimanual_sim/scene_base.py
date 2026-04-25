"""Scene-module conventions.

Every file in `scenes/` is expected to expose, at module level:

    NAME: str                       # display name used in the Viser GUI
    ARM_PREFIXES: tuple[str, ...]   # prefix strings identifying arms in the model
                                    #   (empty tuple for scenes with no articulated arm)
    N_CUBES: int                    # number of grippable objects (0 if no grasp)

    def build_spec() -> mujoco.MjSpec:
        '''Construct the uncompiled MJCF spec.'''

    def apply_initial_state(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[ArmSide, ArmHandles],
        cube_body_ids: list[int],
    ) -> None:
        '''Set qpos/ctrl/eq_active to their demo-start values.'''

One of these two, depending on whether the scene is scripted or free-play:

    def make_task_plan(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[ArmSide, ArmHandles],
        cube_body_ids: list[int],
    ) -> dict[ArmSide, list[Step]]:
        '''Return a per-arm list of waypoint steps.'''

    def step_free_play(
        t: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        '''Called each frame; set data.ctrl however you like.'''

The runner introspects the module, so either (or both) can be absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

import numpy as np
from jaxtyping import Float

# ---------------------------------------------------------------------------
# Array shape aliases (jaxtyping).
# These are purely annotation aliases — jaxtyping carries no runtime cost
# unless combined with a checker. They exist so IK, welds, and scene helpers
# can encode the expected shape (`(3,)`, `(4,)`, `(6,)`) in the signature
# instead of passing bare `np.ndarray`.
# ---------------------------------------------------------------------------
Position3 = Float[np.ndarray, "3"]
QuatWxyz = Float[np.ndarray, "4"]
JointConfig = Float[np.ndarray, "6"]  # Piper 6-DoF arm qpos

# The set of gripper command states is finite and project-wide; using a Literal
# lets ty/mypy catch a typo at the call site instead of blowing up at runtime.
GripperState = Literal["open", "closed"]

# Bounded index into a scene's grippable-object list. Construct via
# `make_cube_id` so the bound check lives in one place; downstream code
# (runner, Step) accepts it without re-checking.
CubeID = NewType("CubeID", int)


def make_cube_id(index: int, n_cubes: int) -> CubeID:
    """Build a bounds-checked CubeID; raises if the index is out of range."""
    if not 0 <= index < n_cubes:
        raise IndexError(f"cube index {index} out of range for scene with n_cubes={n_cubes}")
    return CubeID(index)


@dataclass
class Step:
    """A single waypoint in a per-arm timeline."""

    label: str
    arm_q: JointConfig  # shape (6,) — target joint configuration for the arm
    gripper: GripperState
    duration: float  # seconds at seep 1.0

    # Grasp welds (arm.link6 ↔ cube). Addressed by a CubeID (bounds-checked int
    # index into cube_body_ids). `weld_activate` teleports the cube to the TCP;
    # used for the "free cube in sink" case. `weld_deactivate` just clears the
    # flag.
    weld_activate: CubeID | None = None
    weld_deactivate: CubeID | None = None

    # Generic attachment welds (body ↔ body), addressed by the weld's MJCF
    # name. Used when the two bodies are already in their desired relative
    # pose (cable connector seated in a port, server slotted in a rack, arm
    # closing on an already-positioned connector). `attach_activate` freezes
    # the current relative pose; `attach_deactivate` clears the flag.
    #
    # Scenes with a closed set of attachment welds should define a scene-local
    # `StrEnum` (e.g. `class AttachmentWeldName(StrEnum): ...`) and pass its
    # members here — `StrEnum` values are `str` subclasses, so the tuple type
    # below accepts them while downstream code still benefits from enum-level
    # typo protection at the call site.
    attach_activate: tuple[str, ...] = ()
    attach_deactivate: tuple[str, ...] = ()

    # Same as `attach_activate` but each entry pairs a weld name with an
    # explicit target world pose `((x, y, z), (qw, qx, qy, qz))`. The
    # weld activates such that body_b lands at exactly that pose,
    # *regardless of where it currently is* — used for deterministic
    # placements (server-on-shelf, new-server-in-rack) where capturing
    # the runtime position would propagate arm-tracking offsets into
    # the final object pose.
    attach_activate_at: (
        tuple[
            tuple[
                str,
                tuple[float, float, float],
                tuple[float, float, float, float],
            ],
            ...,
        ]
        | None
    ) = None

    # Scene-owned actuator targets (e.g. a lift prismatic joint). Keys are
    # actuator names the scene declares via AUX_ACTUATOR_NAMES; values are
    # target ctrl values. Runner interpolates to these alongside arm joints.
    # Same StrEnum-as-str convention as attach_activate: scenes may pass
    # enum members and still satisfy the `str` key type.
    aux_ctrl: dict[str, float] | None = None

    def __post_init__(self) -> None:
        # arm_q is the one field the type system can't fully express (shape
        # invariant at runtime), so we keep this check. `gripper` is handled
        # by Literal at type-check time; `weld_*` is bounds-checked by
        # `make_cube_id`.
        self.arm_q = np.asarray(self.arm_q, dtype=float)
        if self.arm_q.shape != (6,):
            raise ValueError(f"arm_q must be length 6, got shape {self.arm_q.shape}")
