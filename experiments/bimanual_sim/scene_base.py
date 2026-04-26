"""Scene-module conventions.

Every file in `scenes/` is expected to expose, at module level:

    NAME: str                       # display name used in the Viser GUI
    ARM_PREFIXES: tuple[str, ...]   # prefix strings identifying arms in the model
                                    #   (empty tuple for scenes with no articulated arm)
    N_CUBES: int                    # number of grippable objects (0 if no grasp)

    def build_spec() -> tuple[mujoco.MjModel, mujoco.MjData]:
        '''Construct + compile the scene; return (model, data).'''

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

Scripted scenes may also expose:

    PHASE_CONTRACTS: tuple[PhaseContract, ...]
        Declarative start/end expectations for named TaskPhase boundaries.
        Debug tooling uses these to write snapshots and fail fast when a
        phase leaves attachments or base joints in the wrong state.

The runner introspects the module, so either (or both) can be absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NewType

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


class TaskPhase(StrEnum):
    """High-level phase for a scripted task-plan step.

    These values describe user-visible task progress, not implementation
    mechanics. They let tools group timeline rows without parsing labels.
    """

    UNPHASED = "unphased"
    SETUP = "setup"
    DISCONNECT_CABLES = "disconnect_cables"
    REMOVE_OLD_SERVER = "remove_old_server"
    STOW_OLD_SERVER = "stow_old_server"
    RETRIEVE_NEW_SERVER = "retrieve_new_server"
    INSTALL_NEW_SERVER = "install_new_server"
    RECONNECT_CABLES = "reconnect_cables"
    RESET = "reset"


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
    phase: TaskPhase = TaskPhase.UNPHASED

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
    aux_ctrl: Mapping[Any, float] | None = None

    def __post_init__(self) -> None:
        # arm_q is the one field the type system can't fully express (shape
        # invariant at runtime), so we keep this check. `gripper` is handled
        # by Literal at type-check time; `weld_*` is bounds-checked by
        # `make_cube_id`.
        self.arm_q = np.asarray(self.arm_q, dtype=float)
        if self.arm_q.shape != (6,):
            raise ValueError(f"arm_q must be length 6, got shape {self.arm_q.shape}")


@dataclass(frozen=True)
class GrippablePoseExpectation:
    """Expected world pose of a grippable body at a phase boundary.

    Catches IK-drift / wrong-pose-weld bugs that pure attachment-flag
    contracts miss: a server can be welded into the rack while sitting
    8 cm off-centre because the upstream IK approach landed there.
    `position` + `tolerance_m` is enough for translational drift;
    `quat` is optional because most grippables in this project are
    pose-locked by their welds and we mainly care about position.
    """

    name: str
    position: tuple[float, float, float]
    tolerance_m: float = 0.02


@dataclass(frozen=True)
class PhaseState:
    """Concrete checkpoint state at a phase boundary.

    This is intentionally declarative: it documents what should be true
    before/after a phase, and tools can print or later assert it without
    parsing free-form comments.
    """

    description: str
    active_attachments: tuple[str, ...] = ()
    inactive_attachments: tuple[str, ...] = ()
    base_aux: tuple[tuple[str, float], ...] = ()
    expected_grippable_poses: tuple[GrippablePoseExpectation, ...] = ()


@dataclass(frozen=True)
class QaccSentinel:
    """Phase invariant: MuJoCo's QACC warning counter must not grow.

    `mjData.warning[mjWARN_BADQACC].number` is cumulative across the
    sim run; the check captures it at phase start and asserts the
    delta is `<= max_increase` at phase end. Default 0 means "no new
    QACC warnings during this phase" — the canonical stability guard
    for tendon-equality / contact-rich phases (cable replug, server
    insertion).
    """

    max_increase: int = 0


@dataclass(frozen=True)
class WeldHoldInvariant:
    """Phase invariant: an attachment weld must stay in a fixed state.

    Boundary checks already verify weld state at phase start/end.
    This invariant verifies the weld doesn't *flicker* mid-phase —
    e.g. a "stow old server" phase where the cart-bottom weld must
    stay active throughout, even though IK perturbations can briefly
    knock it off.
    """

    name: str
    must_be_active: bool


PhaseInvariant = QaccSentinel | WeldHoldInvariant


@dataclass(frozen=True)
class PhaseContract:
    """Start/end contract for a task phase, plus mid-phase invariants.

    `starts` / `ends` are evaluated at phase boundaries (cheap).
    `invariants` are sampled every tick the phase is active — keep them
    cheap to evaluate. The runner enforces both when run with --strict.
    """

    phase: TaskPhase
    starts: PhaseState
    ends: PhaseState
    invariants: tuple[PhaseInvariant, ...] = ()
