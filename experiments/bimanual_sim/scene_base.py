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

# jaxtyping aliases carry no runtime cost without a checker; they encode the
# expected shape in signatures instead of passing bare `np.ndarray`.
Position3 = Float[np.ndarray, "3"]
QuatWxyz = Float[np.ndarray, "4"]
JointConfig = Float[np.ndarray, "6"]

GripperState = Literal["open", "closed"]


class TaskPhase(StrEnum):
    """High-level phase for a scripted task-plan step.

    These values describe user-visible task progress, not implementation
    mechanics. They let tools group timeline rows without parsing labels.
    """

    UNPHASED = "unphased"
    SETUP = "setup"
    # Server-swap (mobile_aloha_ur10e_server_swap) phases:
    REMOVE_OLD_SERVER = "remove_old_server"
    BACKUP_FROM_RACK = "backup_from_rack"
    TRAVERSE_TO_CART = "traverse_to_cart"
    STOW_OLD_SERVER = "stow_old_server"
    RETRIEVE_NEW_SERVER = "retrieve_new_server"
    TRAVERSE_TO_RACK = "traverse_to_rack"
    ADVANCE_INTO_RACK = "advance_into_rack"
    INSTALL_NEW_SERVER = "install_new_server"
    RESET = "reset"
    # Indicator-check (mobile_aloha_piper_indicator_check) phases:
    TRAVERSE_INTO_AISLE = "traverse_into_aisle"
    ALIGN_TO_TARGET = "align_to_target"
    REACH_TO_SERVER = "reach_to_server"
    WAIT_AT_SERVER = "wait_at_server"
    RETRACT = "retract"


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
    arm_q: JointConfig
    gripper: GripperState
    duration: float  # seconds at speed 1.0
    phase: TaskPhase = TaskPhase.UNPHASED

    # Grasp welds (arm.link6 ↔ cube), addressed by `CubeID` index into
    # `cube_body_ids`. `weld_activate` teleports the cube to the TCP — the
    # "free cube in sink" case. `weld_deactivate` just clears the flag.
    weld_activate: CubeID | None = None
    weld_deactivate: CubeID | None = None

    # Generic body↔body attachment welds, addressed by MJCF name. Used when
    # the two bodies are already in their desired relative pose (server in
    # rack, connector seated in port). `attach_activate` freezes the current
    # relative pose.
    #
    # Scenes with a closed weld set should define a scene-local `StrEnum`
    # (`class AttachmentWeldName(StrEnum): ...`) and pass its members here.
    # StrEnum values are `str` subclasses, so the tuple type accepts them
    # while keeping enum-level typo protection at the call site.
    attach_activate: tuple[str, ...] = ()
    attach_deactivate: tuple[str, ...] = ()

    # Like `attach_activate` but each entry pairs a weld name with an explicit
    # target world pose `((x, y, z), (qw, qx, qy, qz))`. body_b lands exactly
    # at that pose regardless of its current position — used for deterministic
    # placements (server-on-shelf, new-server-in-rack) where capturing the
    # runtime pose would propagate arm-tracking offsets into the final object.
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

    # Scene-owned actuator targets (e.g. a lift prismatic). Keys are actuator
    # names declared via `AUX_ACTUATOR_NAMES`; runner interpolates to these
    # alongside arm joints. StrEnum-as-str: scenes may pass enum members.
    aux_ctrl: Mapping[Any, float] | None = None

    # On entry to this step, set each named geom's RGBA. Pairs are
    # `(geom_name, (r, g, b, a))`. The runner writes both `model.geom_rgba`
    # (so any subsequent native MuJoCo render sees the new colour) and the
    # corresponding viser MeshHandle (remove + re-add). Used for visual-only
    # state changes that don't fit the weld/grasp model — e.g. an indicator
    # light flipping from red to green when a service action completes.
    set_geom_rgba: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()

    def __post_init__(self) -> None:
        # arm_q's shape invariant can't be expressed in the type system, so
        # check it at construction. Other fields are guarded by Literal /
        # `make_cube_id` already.
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


@dataclass(frozen=True)
class JointSetStatic:
    """Phase invariant: a named set of joints must not move within the phase.

    Per tick the checker compares qpos against the previous tick; if any
    joint in `joint_names` moved by more than `qpos_epsilon`, the invariant
    fails. `label` shows up in failure messages — pass `"arms"` /
    `"base"` so reports are legible when multiple `JointSetStatic`
    invariants run in the same phase.
    """

    joint_names: tuple[str, ...]
    label: str = "joint_set"
    qpos_epsilon: float = 1e-4


@dataclass(frozen=True)
class GripperStateHold:
    """Phase invariant: gripper actuator(s) must stay at a fixed ctrl value.

    Catches authoring bugs that toggle the gripper mid-carry — in a real
    robot, opening the gripper while still welded to the load would drop
    the held object. `actuator_names` lets one invariant cover both arms.
    """

    actuator_names: tuple[str, ...]
    expected_ctrl_value: float
    label: str = "gripper"
    tolerance: float = 1.0


@dataclass(frozen=True)
class BimanualHandleSeparation:
    """Phase invariant: TCP-to-TCP world distance must stay near `target_distance_m`.

    When both arm grasp welds onto the same rigid body are active, the two
    arms over-constrain the body's pose — if their TCPs drift apart in
    world frame, MuJoCo's solver thrashes (typically blowing up QACC).
    `requires_active_welds` gates the check: if any listed weld is
    inactive, the invariant trivially passes. Pass the per-side grasp
    weld names so the invariant only fires while the bimanual hold is
    actually engaged.
    """

    left_tcp_site: str
    right_tcp_site: str
    target_distance_m: float
    requires_active_welds: tuple[str, ...] = ()
    tolerance_m: float = 0.02


@dataclass(frozen=True)
class HeldObjectLevelness:
    """Phase invariant: a body's pitch/roll must stay within tolerance.

    Encodes the "server stays horizontal during the carry" requirement
    from `NEW_LAYOUT.md`. `requires_active_welds` gates the check the
    same way as `BimanualHandleSeparation` — if the body isn't being
    carried right now (no grasp welds active), the levelness check
    doesn't apply.
    """

    body_name: str
    max_pitch_rad: float
    max_roll_rad: float
    requires_active_welds: tuple[str, ...] = ()


PhaseInvariant = (
    QaccSentinel
    | WeldHoldInvariant
    | JointSetStatic
    | GripperStateHold
    | BimanualHandleSeparation
    | HeldObjectLevelness
)


@dataclass(frozen=True)
class PhaseContract:
    """Start/end contract for a task phase, plus mid-phase invariants.

    `starts` / `ends` are evaluated at phase boundaries (cheap).
    `invariants` are sampled every tick the phase is active — keep them
    cheap to evaluate. The runner enforces both when run with --strict.

    `legal_predecessors` declares which phases may transition INTO this one;
    empty tuple = no constraint (unrestricted). Catches choreography reorder
    bugs at the boundary instead of as opaque physics divergences later.
    Legacy scenes that don't populate the field keep their old behaviour.
    """

    phase: TaskPhase
    starts: PhaseState
    ends: PhaseState
    invariants: tuple[PhaseInvariant, ...] = ()
    legal_predecessors: tuple[TaskPhase, ...] = ()
