"""TCP-drag teleop for keyframe-based demo authoring.

Replaces the scripted `make_task_plan` with a live IK loop driven by
viser drag handles. The user grabs each arm's TCP handle in the
browser, flies the arm to the desired pose, presses **Capture**, and
the controller appends a `Step` to that arm's buffer tagged with the
currently-selected `TaskPhase`. **Save** writes the buffered steps to
JSON; the existing runner can replay that JSON via
`load_recording`.

What teleop captures into each Step:

* `arm_q` — solved joint config from the per-tick IK.
* `gripper` — `open` | `closed` from the per-arm toggle.
* `aux_ctrl` — base x/y/yaw from the base handle's pose, written so
  replay drives the chassis through the existing aux-actuator path.
* `attach_activate` / `attach_deactivate` — diffs against the desired
  weld state declared by per-weld checkboxes; weld toggles take
  effect immediately so the user sees the welded body update in
  viser, and the *transition* is what gets recorded on the Step.
* `weld_activate` / `weld_deactivate` — per-arm grasp welds from
  per-arm grasp-target dropdowns. Same diff-since-last-capture model.

Phase boundary buttons run `check_phase_state` on the currently-
selected phase's contract immediately, so the user sees pass/fail
the moment they mark a phase end. This is the principal payoff of
authoring teleop *after* the contracts went in.

Why teleop and not more scripted IK: the rack/cart approach poses,
gripper-aiming angles, and arm-retract timing are empirical — the
scripted IK gets the math right but produces motion that looks
robotic rather than purposeful. Teleop lets you author the *shape*
of each phase directly in the viewer; the contract guard catches
keyframes that don't actually finish their phase.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import mujoco
import numpy as np
import viser

from arm_handles import ArmHandles, ArmSide, arm_joint_suffixes
from ik import PositionOnly, solve_ik
from scene_base import (
    CubeID,
    GripperState,
    PhaseContract,
    Step,
    TaskPhase,
    make_cube_id,
)
from scene_check import AttachmentConstraint
from tools.observability import check_phase_state
from welds import (
    activate_attachment_weld,
    activate_grasp_weld,
    deactivate_grasp_weld,
    deactivate_weld,
)

# ---------------------------------------------------------------------------
# Per-side runtime state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MarkerSeed:
    """Construction-time bundle for one arm's marker + initial target.

    Created by `_build_marker_seeds` and consumed by `_build_gui` so
    `_ArmTeleopState` can be constructed exactly once with both the
    marker handle and the real slider handles in scope. Pre-existed
    as a placeholder-mutation pattern that violated the "make invalid
    states hard to represent" guideline.
    """

    target_body_xyz: np.ndarray
    target_marker: viser.IcosphereHandle


@dataclass
class _ArmTeleopState:
    """Per-arm teleop runtime state.

    Two parallel slider sets:

    * `joint_sliders` — one per arm DoF (6 for UR10e). These are the
      **authoritative source of truth** for the arm's qpos: `tick()`
      reads from them and writes straight to data.qpos. Lets the user
      pose the arm joint-by-joint when IK can't disambiguate (e.g.
      elbow-up vs elbow-down for the same TCP target).
    * `target_sliders` — body-frame TCP x/y/z. Their `on_update`
      callback solves IK using the current arm pose as seed and
      pushes the solved joint config into `joint_sliders`. The
      target sliders themselves are convenience inputs only — they
      don't drive the arm directly.

    `target_body_xyz` is the IK target expressed in the **base body
    frame** so the marker rides along with the chassis when the user
    moves the base. The world-frame target the IK actually solves to
    is recomputed as `base_pose * target_body_xyz`.

    The ghost-sphere `target_marker` lives in world coordinates so the
    user sees where the gripper is being commanded to.
    """

    side: ArmSide
    target_body_xyz: np.ndarray
    target_marker: viser.IcosphereHandle
    target_sliders: tuple[viser.GuiSliderHandle, viser.GuiSliderHandle, viser.GuiSliderHandle]
    joint_sliders: tuple[viser.GuiSliderHandle, ...]
    gripper: GripperState = "open"
    grasp_target: str | None = None
    last_captured_grasp: str | None = None


@dataclass
class _SceneSnapshot:
    """In-memory snapshot of the dynamic scene state for "reset to here".

    Carries the minimum needed to rewind: qpos / qvel / ctrl /
    eq_active drive the simulator forward, and per-arm
    `target_body_xyz` lets us put the IK sliders back where they were
    so the user doesn't see a sudden visual jump after the reset.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    eq_active: np.ndarray
    target_body_xyz: dict[ArmSide, np.ndarray]
    sim_time: float


@dataclass
class _CapturedKeyframe:
    """One snapshot of an arm's state ready for serialisation.

    Keeps the same shape as `Step` so `load_recording` can build a
    `Step` directly. `aux_ctrl` is shared between left and right arm
    captures (both arms see the same base pose).
    """

    label: str
    phase: TaskPhase
    duration_s: float
    arm_q: np.ndarray
    gripper: GripperState
    aux_ctrl: dict[str, float] = field(default_factory=dict)
    attach_activate: tuple[str, ...] = ()
    attach_deactivate: tuple[str, ...] = ()
    # `CubeID` is the bounds-checked grippable index. Storing it
    # here (rather than raw int) keeps a single source of truth for
    # "valid grasp target index" between capture and Step replay.
    weld_activate_idx: CubeID | None = None
    weld_deactivate_idx: CubeID | None = None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class TeleopController:
    """Wires viser handles + GUI + IK loop together.

    Constructed once via `attach()`. The runner calls `tick(dt)` every
    render frame to advance IK, refresh GUI status, and push the base
    handle's pose into the chassis qpos.
    """

    server: viser.ViserServer
    model: mujoco.MjModel
    data: mujoco.MjData
    arms: dict[ArmSide, ArmHandles]
    locked_joint_names: tuple[str, ...]
    attachments: tuple[AttachmentConstraint, ...]
    phase_contracts: tuple[PhaseContract, ...]
    grippable_names: tuple[str, ...]
    cube_body_ids: tuple[int, ...]
    base_qposadr: dict[str, int]
    base_aux_names: tuple[str, str, str]  # (x, y, yaw)
    _scene_name: str = "unknown"
    _save_root: Path = field(default_factory=lambda: Path("/tmp/teleop_recordings"))

    # --- runtime state ---
    _arm_states: dict[ArmSide, _ArmTeleopState] = field(default_factory=dict)
    _captured: dict[ArmSide, list[_CapturedKeyframe]] = field(default_factory=dict)
    _base_x_slider: viser.GuiSliderHandle | None = None
    _base_y_slider: viser.GuiSliderHandle | None = None
    _base_yaw_slider: viser.GuiSliderHandle | None = None
    # Desired-state map for attachment welds: name -> active?
    _weld_desired: dict[str, bool] = field(default_factory=dict)
    # The weld desired-state at the moment of the LAST capture, used
    # to compute deltas at the next capture.
    _weld_at_last_capture: dict[str, bool] = field(default_factory=dict)
    # GUI handles kept around so methods can read/write them.
    #
    # Justified type exception: the dropdown handles use `Any` rather
    # than `viser.GuiDropdownHandle[str]`. viser narrows the handle's
    # TypeVar to the exact `Literal[...]` of the options tuple at
    # construction time, so a wider `str` annotation rejects what
    # `add_dropdown` returns. Reading `.value` on the dropdown still
    # gives back the literal type at the call site (where it gets
    # converted via `TaskPhase(...)`), so the looseness is contained
    # to the field declaration.
    _phase_dropdown: Any = None
    _label_input: viser.GuiInputHandle[str] | None = None
    _duration_slider: viser.GuiSliderHandle | None = None
    _ik_pos_err_text: viser.GuiTextHandle | None = None
    _capture_count_text: viser.GuiTextHandle | None = None
    _last_save_text: viser.GuiTextHandle | None = None
    _phase_status_text: viser.GuiTextHandle | None = None
    _weld_checkboxes: dict[str, viser.GuiCheckboxHandle] = field(default_factory=dict)
    _grasp_dropdowns: dict[ArmSide, Any] = field(default_factory=dict)  # see _phase_dropdown note
    # Diagnostic counters — used for both the GUI status text and
    # the bounded boot-time stderr logging that helps catch
    # "client never pushes drag updates" failure modes.
    _handle_update_count: int = 0
    _tick_count: int = 0
    # Reset-to-here snapshots. `_scene_start_snapshot` is captured
    # once at construction; `_phase_start_snapshots[phase]` is
    # captured each time the user clicks "Check phase START
    # contract" so the user can rewind to that moment.
    _scene_start_snapshot: _SceneSnapshot | None = None
    _phase_start_snapshots: dict[TaskPhase, _SceneSnapshot] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def attach(
        cls,
        server: viser.ViserServer,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[ArmSide, ArmHandles],
        locked_joint_names: tuple[str, ...],
        attachments: tuple[AttachmentConstraint, ...],
        phase_contracts: tuple[PhaseContract, ...],
        grippable_names: tuple[str, ...],
        cube_body_ids: tuple[int, ...],
        base_aux_names: tuple[str, str, str],
        scene_name: str,
        save_root: Path | None = None,
    ) -> TeleopController:
        """Build a controller and add its handles + GUI to `server`.

        `base_aux_names` is the (x, y, yaw) tuple matching the scene's
        `DataCenterAux` enum; the controller resolves their joint
        indices from `model` and writes qpos directly to drive the
        base from the dragged handle.
        """
        # Resolve base joint qpos addresses (so we can puppet the
        # chassis from the handle).
        base_qposadr: dict[str, int] = {}
        for joint_name in base_aux_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                # Maybe the name is an actuator — resolve via its trnid.
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if aid < 0:
                    raise ValueError(
                        f"teleop: base joint/actuator {joint_name!r} not in compiled model"
                    )
                jid = int(model.actuator_trnid[aid][0])
            base_qposadr[joint_name] = int(model.jnt_qposadr[jid])

        controller = cls(
            server=server,
            model=model,
            data=data,
            arms=arms,
            locked_joint_names=locked_joint_names,
            attachments=attachments,
            phase_contracts=phase_contracts,
            grippable_names=grippable_names,
            cube_body_ids=cube_body_ids,
            base_qposadr=base_qposadr,
            base_aux_names=base_aux_names,
            _scene_name=scene_name,
            _save_root=save_root or Path("/tmp/teleop_recordings"),
        )
        controller._captured = {side: [] for side in arms}
        controller._weld_desired = {att.name: bool(att.initially_active) for att in attachments}
        controller._weld_at_last_capture = dict(controller._weld_desired)
        # Build markers + sliders, then construct each per-arm state
        # exactly once with both already in hand. `_arm_states` is
        # populated by `_build_gui` only — no placeholder values, no
        # mid-construction mutation.
        marker_seeds = controller._build_marker_seeds()
        controller._build_gui(marker_seeds)
        controller._scene_start_snapshot = controller._capture_snapshot()
        return controller

    def _build_marker_seeds(self) -> dict[ArmSide, _MarkerSeed]:
        """Spawn each arm's ghost-sphere marker and compute its initial
        body-frame TCP. Returns a per-arm seed `_MarkerSeed` so
        `_build_gui` can assemble the full `_ArmTeleopState` once the
        slider widgets exist.
        """
        # mj_forward so site_xpos reflects whatever apply_initial_state
        # wrote into qpos — without it the markers would spawn at the
        # default site_xpos = (0, 0, 0) and the user would see them
        # buried in the floor.
        mujoco.mj_forward(self.model, self.data)
        bx, by, byaw = self._read_base_pose()
        seeds: dict[ArmSide, _MarkerSeed] = {}
        for side, arm in self.arms.items():
            tcp_world = np.asarray(self.data.site_xpos[arm.tcp_site_id], dtype=float).copy()
            tcp_body = self._world_to_body(tcp_world, bx, by, byaw)
            colour = (1.0, 0.4, 0.0) if "left" in side else (0.0, 0.6, 1.0)
            marker = self.server.scene.add_icosphere(
                f"teleop/{side.rstrip('/')}/target",
                radius=0.03,
                color=colour,
                position=(float(tcp_world[0]), float(tcp_world[1]), float(tcp_world[2])),
                opacity=0.6,
            )
            seeds[side] = _MarkerSeed(target_body_xyz=tcp_body, target_marker=marker)
        return seeds

    # ------------------------------------------------------------------
    # Base-frame helpers
    # ------------------------------------------------------------------

    def _read_base_pose(self) -> tuple[float, float, float]:
        """Current (base_x, base_y, base_yaw) read from data.qpos."""
        return (
            float(self.data.qpos[self.base_qposadr[self.base_aux_names[0]]]),
            float(self.data.qpos[self.base_qposadr[self.base_aux_names[1]]]),
            float(self.data.qpos[self.base_qposadr[self.base_aux_names[2]]]),
        )

    @staticmethod
    def _snap_to_step(value: float, step: float) -> float:
        """Round `value` to the slider's step grid so viser's GUI
        doesn't display the underlying float64 precision (which
        otherwise shows up as e.g. "1.570796326794896" instead of
        "1.55"). User-driven slider drags already snap to step on
        the client side; this helper handles programmatic writes
        from IK / FK / snapshot restore.
        """
        step_str = f"{step:.10f}".rstrip("0").rstrip(".")
        decimals = len(step_str.split(".")[-1]) if "." in step_str else 0
        return round(round(value / step) * step, decimals)

    @staticmethod
    def _body_to_world(
        body_xyz: np.ndarray, base_x: float, base_y: float, base_yaw: float
    ) -> np.ndarray:
        """Transform a body-frame point to world frame using the chassis
        2D pose (planar translation + yaw about z)."""
        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        wx = base_x + body_xyz[0] * cos_y - body_xyz[1] * sin_y
        wy = base_y + body_xyz[0] * sin_y + body_xyz[1] * cos_y
        wz = body_xyz[2]
        return np.array([wx, wy, wz], dtype=float)

    @staticmethod
    def _world_to_body(
        world_xyz: np.ndarray, base_x: float, base_y: float, base_yaw: float
    ) -> np.ndarray:
        """Inverse of `_body_to_world`."""
        dx = world_xyz[0] - base_x
        dy = world_xyz[1] - base_y
        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)
        bx = dx * cos_y + dy * sin_y
        by = -dx * sin_y + dy * cos_y
        bz = world_xyz[2]
        return np.array([bx, by, bz], dtype=float)

    def _build_gui(self, marker_seeds: dict[ArmSide, _MarkerSeed]) -> None:
        with self.server.gui.add_folder("Teleop"):
            self._label_input = self.server.gui.add_text(
                "label",
                initial_value="approach",
                hint="freeform label saved with the next captured step",
            )
            phase_options = tuple(p.value for p in TaskPhase)
            self._phase_dropdown = self.server.gui.add_dropdown(
                "phase",
                options=phase_options,
                initial_value=TaskPhase.UNPHASED.value,
                hint="phase tag for the next captured step + the contract checked by Mark phase buttons",
            )
            self._duration_slider = self.server.gui.add_slider(
                "duration (s)",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=0.8,
                hint="seek-time for the captured step at speed=1",
            )

            # Per-arm: target-position sliders + gripper + grasp target.
            # Slider ranges are in BODY frame so they describe the
            # arm's reach envelope relative to the chassis. UR10e has
            # ~1.3 m max reach; the ranges cover that with margin.
            slider_ranges = {
                "x": (-0.5, 1.4, 0.01),
                "y": (-0.8, 0.8, 0.01),
                "z": (0.0, 1.8, 0.01),
            }
            # Joint labels for the per-joint sliders. Read from the
            # arm_handles single-source-of-truth — same suffixes the
            # IK / runtime code uses to resolve joint IDs, so a robot
            # swap can't drift labels out of sync with the actual
            # joints they're driving.
            arm_joint_labels = arm_joint_suffixes(next(iter(self.arms.values())).robot_kind)

            for side in self.arms:
                arm_label = side.rstrip("/") or "arm"
                arm = self.arms[side]
                seed = marker_seeds[side]
                with self.server.gui.add_folder(f"{arm_label} arm"):
                    # --- TCP body-frame target sliders ----------------
                    target_sliders_list: list[viser.GuiSliderHandle] = []
                    for axis_idx, (axis_name, (lo, hi, step)) in enumerate(slider_ranges.items()):
                        s = self.server.gui.add_slider(
                            f"{arm_label} target_{axis_name}",
                            min=lo,
                            max=hi,
                            step=step,
                            initial_value=float(seed.target_body_xyz[axis_idx]),
                            hint=f"{arm_label} arm IK target — body-frame {axis_name} (m)",
                        )

                        def _on_target_change(
                            event: viser.GuiEvent[viser.GuiSliderHandle],
                            *,
                            target_side: ArmSide = side,
                            target_axis: int = axis_idx,
                        ) -> None:
                            st = self._arm_states[target_side]
                            new = st.target_body_xyz.copy()
                            new[target_axis] = float(event.target.value)
                            st.target_body_xyz = new
                            self._handle_update_count += 1

                        s.on_update(_on_target_change)
                        target_sliders_list.append(s)
                    target_sliders: tuple[
                        viser.GuiSliderHandle,
                        viser.GuiSliderHandle,
                        viser.GuiSliderHandle,
                    ] = (
                        target_sliders_list[0],
                        target_sliders_list[1],
                        target_sliders_list[2],
                    )

                    solve_ik_btn = self.server.gui.add_button(f"{arm_label}: Solve IK → joints")
                    solve_ik_btn.on_click(
                        lambda _e, target_side=side: self._solve_ik_to_target(target_side)
                    )

                    # --- per-joint sliders (authoritative for qpos) ---
                    joint_sliders_list: list[viser.GuiSliderHandle] = []
                    current_q = [float(self.data.qpos[i]) for i in arm.arm_qpos_idx]
                    for joint_idx, joint_label in enumerate(arm_joint_labels[: len(current_q)]):
                        js = self.server.gui.add_slider(
                            f"{arm_label} {joint_label} (rad)",
                            min=-math.pi,
                            max=math.pi,
                            step=0.05,
                            initial_value=current_q[joint_idx],
                            hint=(
                                f"{arm_label} arm joint {joint_label} angle "
                                "(authoritative — directly drives qpos every tick)"
                            ),
                        )
                        joint_sliders_list.append(js)

                    # Construct the full per-arm state ONCE with both
                    # marker + sliders in scope. No mutation, no
                    # placeholders.
                    self._arm_states[side] = _ArmTeleopState(
                        side=side,
                        target_body_xyz=seed.target_body_xyz.copy(),
                        target_marker=seed.target_marker,
                        target_sliders=target_sliders,
                        joint_sliders=tuple(joint_sliders_list),
                    )

                    # Wire each joint slider to re-centre the IK
                    # target on the resulting TCP. Registered after
                    # `_arm_states[side]` exists so the callback's
                    # state lookup is safe. Fires on user drags AND
                    # programmatic value sets (e.g. from Solve IK →
                    # joints), so the marker always tracks the actual
                    # gripper.
                    for js in joint_sliders_list:
                        js.on_update(
                            lambda _e, target_side=side: self._sync_target_to_tcp(target_side)
                        )

                    reset_arm_btn = self.server.gui.add_button(f"{arm_label}: reset joints to home")
                    reset_arm_btn.on_click(
                        lambda _e, target_side=side: self._reset_arm_to_home(target_side)
                    )

                    open_btn = self.server.gui.add_button(f"{arm_label}: gripper OPEN")
                    close_btn = self.server.gui.add_button(f"{arm_label}: gripper CLOSE")

                    def _set_open(_event: Any, *, target_side: ArmSide = side) -> None:
                        self._arm_states[target_side].gripper = "open"

                    def _set_close(_event: Any, *, target_side: ArmSide = side) -> None:
                        self._arm_states[target_side].gripper = "closed"

                    open_btn.on_click(_set_open)
                    close_btn.on_click(_set_close)

                    grasp_options = ("(none)", *self.grippable_names)
                    grasp_dropdown = self.server.gui.add_dropdown(
                        f"{arm_label} grasp",
                        options=grasp_options,
                        initial_value="(none)",
                        hint="grippable to weld to this arm at next capture",
                    )
                    self._grasp_dropdowns[side] = grasp_dropdown

                    def _on_grasp_change(
                        event: viser.GuiEvent[viser.GuiDropdownHandle[str]],
                        *,
                        target_side: ArmSide = side,
                    ) -> None:
                        target = event.target.value
                        self._arm_states[target_side].grasp_target = (
                            None if target == "(none)" else target
                        )

                    grasp_dropdown.on_update(_on_grasp_change)

            # Base controls — three sliders for x, y, yaw. Same
            # rationale as arm sliders: viser TransformControls don't
            # round-trip drag updates in this version.
            with self.server.gui.add_folder("Base"):
                base_init_x = float(self.data.qpos[self.base_qposadr[self.base_aux_names[0]]])
                base_init_y = float(self.data.qpos[self.base_qposadr[self.base_aux_names[1]]])
                base_init_yaw = float(self.data.qpos[self.base_qposadr[self.base_aux_names[2]]])
                self._base_x_slider = self.server.gui.add_slider(
                    "base x", min=-2.0, max=2.0, step=0.01, initial_value=base_init_x
                )
                self._base_y_slider = self.server.gui.add_slider(
                    "base y", min=-2.0, max=2.0, step=0.01, initial_value=base_init_y
                )
                self._base_yaw_slider = self.server.gui.add_slider(
                    "base yaw (rad)",
                    min=-math.pi,
                    max=math.pi,
                    step=0.05,
                    initial_value=base_init_yaw,
                )

            # Attachment-weld checkboxes (active state at end of next capture).
            if self.attachments:
                with self.server.gui.add_folder("Welds"):
                    for att in self.attachments:
                        cb = self.server.gui.add_checkbox(
                            att.name,
                            initial_value=bool(att.initially_active),
                            hint=f"{type(att).__name__}: {att.body_a} ↔ {att.body_b}",
                        )

                        def _on_weld_toggle(
                            event: viser.GuiEvent[viser.GuiCheckboxHandle],
                            *,
                            target_name: str = att.name,
                        ) -> None:
                            self._weld_desired[target_name] = bool(event.target.value)
                            self._apply_weld_state(target_name)

                        cb.on_update(_on_weld_toggle)
                        self._weld_checkboxes[att.name] = cb

            # Capture / save / phase-boundary buttons.
            capture_btn = self.server.gui.add_button("Capture (both arms)")
            capture_btn.on_click(lambda _e: self.capture_step())

            mark_start_btn = self.server.gui.add_button("Check phase START contract")
            mark_start_btn.on_click(lambda _e: self.assert_phase_boundary("starts"))

            mark_end_btn = self.server.gui.add_button("Check phase END contract")
            mark_end_btn.on_click(lambda _e: self.assert_phase_boundary("ends"))

            save_btn = self.server.gui.add_button("Save trajectory → JSON")
            save_btn.on_click(lambda _e: self.save_to_json())

            clear_btn = self.server.gui.add_button("Clear captured buffer")
            clear_btn.on_click(lambda _e: self.clear_buffer())

            reset_phase_btn = self.server.gui.add_button("Reset to phase START")
            reset_phase_btn.on_click(lambda _e: self.reset_to_phase_start())

            reset_scene_btn = self.server.gui.add_button("Reset to scene START")
            reset_scene_btn.on_click(lambda _e: self.reset_to_scene_start())

            print_home_btn = self.server.gui.add_button("Print home_q for layout")
            print_home_btn.on_click(lambda _e: self.print_home_q())

            self._ik_pos_err_text = self.server.gui.add_text(
                "ik err (mm)", initial_value="—", disabled=True
            )
            self._capture_count_text = self.server.gui.add_text(
                "captured", initial_value=" + ".join("0" for _ in self.arms), disabled=True
            )
            self._phase_status_text = self.server.gui.add_text(
                "last contract check", initial_value="—", disabled=True
            )
            self._last_save_text = self.server.gui.add_text(
                "last saved", initial_value="—", disabled=True
            )

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Run IK against per-arm slider targets + drive base from sliders."""
        del dt  # IK runs at the render rate the runner already controls
        self._tick_count += 1

        # 1. Drive base from its three sliders.
        if (
            self._base_x_slider is not None
            and self._base_y_slider is not None
            and self._base_yaw_slider is not None
        ):
            self.data.qpos[self.base_qposadr[self.base_aux_names[0]]] = float(
                self._base_x_slider.value
            )
            self.data.qpos[self.base_qposadr[self.base_aux_names[1]]] = float(
                self._base_y_slider.value
            )
            self.data.qpos[self.base_qposadr[self.base_aux_names[2]]] = float(
                self._base_yaw_slider.value
            )
            for name in self.base_aux_names:
                # Zero the base joints' velocity each tick so the
                # integrator doesn't carry over residual motion from
                # the previous frame's qpos jump.
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    self.data.qvel[int(self.model.jnt_dofadr[jid])] = 0.0

        # 2. Drive each arm directly from its joint sliders. The
        #    target sliders only fire IK on click; in normal teleop
        #    the user moves either the joint sliders (FK mode) or
        #    drives the IK button after dragging the target sliders.
        bx, by, byaw = self._read_base_pose()
        for side, state in self._arm_states.items():
            arm = self.arms[side]
            qpos = np.array([float(s.value) for s in state.joint_sliders], dtype=float)
            self.data.qpos[arm.arm_qpos_idx] = qpos
            self.data.qvel[arm.arm_dof_idx] = 0.0
            self.data.ctrl[arm.act_arm_ids] = qpos

            # Update the world-frame marker showing where we'd LIKE
            # the gripper to be (body target * current base pose).
            target = self._body_to_world(state.target_body_xyz, bx, by, byaw)
            state.target_marker.position = (
                float(target[0]),
                float(target[1]),
                float(target[2]),
            )

            target_g = arm.gripper_open if state.gripper == "open" else arm.gripper_closed
            self.data.ctrl[arm.act_gripper_id] = target_g
            if arm.robot_kind == "piper":
                self.data.qpos[arm.qpos_idx[6]] = target_g
                self.data.qpos[arm.qpos_idx[7]] = -target_g
                self.data.qvel[arm.dof_idx[6]] = 0.0
                self.data.qvel[arm.dof_idx[7]] = 0.0

        if self._ik_pos_err_text is not None:
            self._ik_pos_err_text.value = (
                f"updates={self._handle_update_count}  (joints drive qpos directly)"
            )

        # Bounded stderr breadcrumb so a "drag did nothing" report
        # has actionable evidence: tick count, handle-update count,
        # and the current per-arm target. Suppressed after 60 ticks
        # to avoid flooding logs once the user has confirmed teleop
        # is alive.
        if self._tick_count <= 60 and self._tick_count % 10 == 1:
            for side, state in self._arm_states.items():
                arm = self.arms[side]
                tcp = self.data.site_xpos[arm.tcp_site_id]
                bx, by, byaw = self._read_base_pose()
                world_target = self._body_to_world(state.target_body_xyz, bx, by, byaw)
                print(
                    f"[teleop tick={self._tick_count}] {side} "
                    f"body_target=({state.target_body_xyz[0]:+.3f},"
                    f"{state.target_body_xyz[1]:+.3f},{state.target_body_xyz[2]:+.3f}) "
                    f"world_target=({world_target[0]:+.3f},"
                    f"{world_target[1]:+.3f},{world_target[2]:+.3f}) "
                    f"tcp=({tcp[0]:+.3f},{tcp[1]:+.3f},{tcp[2]:+.3f}) "
                    f"updates={self._handle_update_count}"
                )

    # ------------------------------------------------------------------
    # Weld + grasp application (immediate, so the user sees the change)
    # ------------------------------------------------------------------

    def _apply_weld_state(self, weld_name: str) -> None:
        """Apply the desired_active state of an attachment weld to data.eq_active.

        Triggered the moment the user toggles a checkbox — the
        corresponding bodies snap together (or release) right away
        so teleop feedback is immediate. The captured Step records
        the transition that produced the change.
        """
        eq_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
        if eq_id < 0:
            return
        if self._weld_desired[weld_name]:
            if int(self.model.eq_type[eq_id]) == int(mujoco.mjtEq.mjEQ_CONNECT):
                self.data.eq_active[eq_id] = 1
            else:
                body_a = int(self.model.eq_obj1id[eq_id])
                body_b = int(self.model.eq_obj2id[eq_id])
                activate_attachment_weld(self.model, self.data, eq_id, body_a, body_b)
        else:
            deactivate_weld(self.data, eq_id)

    def _apply_grasp_change(self, side: ArmSide, new_target: str | None) -> None:
        """Apply a per-arm grasp transition (None ↔ grippable name)."""
        arm = self.arms[side]
        # Drop existing grasp first — only one grasp weld per arm.
        prev = self._arm_states[side].last_captured_grasp
        if prev is not None:
            try:
                prev_idx = self.grippable_names.index(prev)
                deactivate_grasp_weld(self.data, int(arm.weld_ids[prev_idx]))
            except ValueError:
                pass
        if new_target is not None:
            new_idx = self.grippable_names.index(new_target)
            activate_grasp_weld(
                self.model,
                self.data,
                int(arm.weld_ids[new_idx]),
                arm.link6_id,
                self.cube_body_ids[new_idx],
                arm.tcp_site_id,
            )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_step(self) -> None:
        """Snapshot both arms' current state as a parallel keyframe pair."""
        if self._label_input is None or self._phase_dropdown is None:
            return
        if self._duration_slider is None:
            return
        label = self._label_input.value or "(unlabelled)"
        phase = TaskPhase(self._phase_dropdown.value)
        duration = float(self._duration_slider.value)

        # Compute weld deltas since the previous capture.
        attach_activate = tuple(
            name
            for name, want in self._weld_desired.items()
            if want and not self._weld_at_last_capture.get(name, False)
        )
        attach_deactivate = tuple(
            name
            for name, want in self._weld_desired.items()
            if not want and self._weld_at_last_capture.get(name, False)
        )

        # Snapshot base aux for replay.
        aux_ctrl = {
            self.base_aux_names[0]: float(
                self.data.qpos[self.base_qposadr[self.base_aux_names[0]]]
            ),
            self.base_aux_names[1]: float(
                self.data.qpos[self.base_qposadr[self.base_aux_names[1]]]
            ),
            self.base_aux_names[2]: float(
                self.data.qpos[self.base_qposadr[self.base_aux_names[2]]]
            ),
        }

        for side, state in self._arm_states.items():
            arm = self.arms[side]
            qpos_now = np.array([self.data.qpos[i] for i in arm.arm_qpos_idx], dtype=float)

            # Per-arm grasp transition since last capture. The
            # grippable name is index-resolved here and immediately
            # wrapped in a `CubeID` so the bounds check fires at
            # capture time, not at replay (when an off-by-one would
            # crash the runner deep inside advance_arm).
            weld_activate_idx: CubeID | None = None
            weld_deactivate_idx: CubeID | None = None
            new_grasp = state.grasp_target
            old_grasp = state.last_captured_grasp
            if new_grasp != old_grasp:
                if new_grasp is not None:
                    weld_activate_idx = make_cube_id(
                        self.grippable_names.index(new_grasp), len(self.grippable_names)
                    )
                elif old_grasp is not None:
                    weld_deactivate_idx = make_cube_id(
                        self.grippable_names.index(old_grasp), len(self.grippable_names)
                    )
                self._apply_grasp_change(side, new_grasp)
                state.last_captured_grasp = new_grasp

            self._captured[side].append(
                _CapturedKeyframe(
                    label=label,
                    phase=phase,
                    duration_s=duration,
                    arm_q=qpos_now,
                    gripper=state.gripper,
                    aux_ctrl=dict(aux_ctrl),
                    attach_activate=attach_activate,
                    attach_deactivate=attach_deactivate,
                    weld_activate_idx=weld_activate_idx,
                    weld_deactivate_idx=weld_deactivate_idx,
                )
            )

        self._weld_at_last_capture = dict(self._weld_desired)

        if self._capture_count_text is not None:
            counts = " + ".join(str(len(self._captured[s])) for s in self.arms)
            self._capture_count_text.value = counts

    def clear_buffer(self) -> None:
        for side in self.arms:
            self._captured[side].clear()
        if self._capture_count_text is not None:
            self._capture_count_text.value = " + ".join("0" for _ in self.arms)

    # ------------------------------------------------------------------
    # Phase boundary contract assertions
    # ------------------------------------------------------------------

    def assert_phase_boundary(self, boundary: Literal["starts", "ends"]) -> None:
        """Run the currently-selected phase's contract against current state.

        Reports pass/fail in the GUI text widget; doesn't raise. Used
        as a "did I actually finish this phase correctly?" check while
        teleoping. As a side-effect when checking the START boundary
        we also snapshot the current scene under that phase — gives
        the user a "rewind to here" point reachable via the "Reset to
        phase START" button.
        """
        if self._phase_dropdown is None or self._phase_status_text is None:
            return
        phase = TaskPhase(self._phase_dropdown.value)
        contract = next((c for c in self.phase_contracts if c.phase == phase), None)
        if contract is None:
            self._phase_status_text.value = f"no contract declared for {phase.value}"
            return
        target_state = contract.starts if boundary == "starts" else contract.ends
        report = check_phase_state(self.model, self.data, target_state)
        if report.ok:
            self._phase_status_text.value = f"{phase.value} {boundary}: OK"
            if boundary == "starts":
                self._phase_start_snapshots[phase] = self._capture_snapshot()
        else:
            first = report.failures[0]
            extra = f" (+{len(report.failures) - 1} more)" if len(report.failures) > 1 else ""
            self._phase_status_text.value = f"{phase.value} {boundary} FAIL: {first.message}{extra}"

    # ------------------------------------------------------------------
    # Snapshot / restore (scene-start + phase-start rewinds)
    # ------------------------------------------------------------------

    def _capture_snapshot(self) -> _SceneSnapshot:
        """Snapshot the current scene state for later restore."""
        return _SceneSnapshot(
            qpos=np.asarray(self.data.qpos, dtype=float).copy(),
            qvel=np.asarray(self.data.qvel, dtype=float).copy(),
            ctrl=np.asarray(self.data.ctrl, dtype=float).copy(),
            eq_active=np.asarray(self.data.eq_active, dtype=np.uint8).copy(),
            target_body_xyz={
                side: state.target_body_xyz.copy() for side, state in self._arm_states.items()
            },
            sim_time=float(self.data.time),
        )

    def _restore_snapshot(self, snapshot: _SceneSnapshot) -> None:
        """Write a snapshot back to the live data + sync the GUI sliders."""
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.ctrl[:] = snapshot.ctrl
        self.data.eq_active[:] = snapshot.eq_active
        self.data.time = snapshot.sim_time
        for side, body_xyz in snapshot.target_body_xyz.items():
            state = self._arm_states[side]
            state.target_body_xyz = body_xyz.copy()
            # Push slider values back so the user sees the rewind.
            for axis_idx, slider in enumerate(state.target_sliders):
                slider.value = self._snap_to_step(float(body_xyz[axis_idx]), slider.step)
            arm = self.arms[side]
            for joint_idx, slider in enumerate(state.joint_sliders):
                slider.value = self._snap_to_step(
                    float(snapshot.qpos[arm.arm_qpos_idx[joint_idx]]), slider.step
                )
        # Refresh the base sliders too — chassis qpos was just rewritten.
        if (
            self._base_x_slider is not None
            and self._base_y_slider is not None
            and self._base_yaw_slider is not None
        ):
            self._base_x_slider.value = self._snap_to_step(
                float(self.data.qpos[self.base_qposadr[self.base_aux_names[0]]]),
                self._base_x_slider.step,
            )
            self._base_y_slider.value = self._snap_to_step(
                float(self.data.qpos[self.base_qposadr[self.base_aux_names[1]]]),
                self._base_y_slider.step,
            )
            self._base_yaw_slider.value = self._snap_to_step(
                float(self.data.qpos[self.base_qposadr[self.base_aux_names[2]]]),
                self._base_yaw_slider.step,
            )
        mujoco.mj_forward(self.model, self.data)

    def reset_to_scene_start(self) -> None:
        """Restore the snapshot taken at controller construction."""
        if self._scene_start_snapshot is None:
            return
        self._restore_snapshot(self._scene_start_snapshot)
        if self._phase_status_text is not None:
            self._phase_status_text.value = "reset → scene start"

    def reset_to_phase_start(self) -> None:
        """Restore the snapshot taken when the user last clicked
        'Check phase START contract' for the currently-selected phase.

        Falls back to scene-start if the user hasn't snapshotted that
        phase yet — better to rewind too far than silently no-op.
        """
        if self._phase_dropdown is None:
            return
        phase = TaskPhase(self._phase_dropdown.value)
        snapshot = self._phase_start_snapshots.get(phase)
        if snapshot is None:
            self.reset_to_scene_start()
            if self._phase_status_text is not None:
                self._phase_status_text.value = (
                    f"no '{phase.value}' START snapshot yet — reset to scene start"
                )
            return
        self._restore_snapshot(snapshot)
        if self._phase_status_text is not None:
            self._phase_status_text.value = f"reset → {phase.value} START"

    # ------------------------------------------------------------------
    # IK + per-joint helpers
    # ------------------------------------------------------------------

    def _sync_target_to_tcp(self, side: ArmSide) -> None:
        """Re-centre the IK target (sliders + marker) on the arm's
        current TCP after a joint-slider change.

        Without this, dragging joint sliders moves the arm but leaves
        the target marker stuck at its old IK target — so the user
        can't tell where the gripper actually is, and a subsequent
        "Solve IK → joints" snaps the arm back to the stale target.
        Sync runs on every joint-slider update (user-driven or
        programmatic from `_solve_ik_to_target` / `_reset_arm_to_home`
        / snapshot restore).
        """
        state = self._arm_states[side]
        arm = self.arms[side]
        # Apply the current joint-slider values to qpos and refresh
        # kinematics before reading site_xpos. tick() will rewrite
        # the same qpos next frame, so this isn't a state escape.
        qpos = np.array([float(s.value) for s in state.joint_sliders], dtype=float)
        self.data.qpos[arm.arm_qpos_idx] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        tcp_world = np.asarray(self.data.site_xpos[arm.tcp_site_id], dtype=float).copy()
        bx, by, byaw = self._read_base_pose()
        tcp_body = self._world_to_body(tcp_world, bx, by, byaw)
        state.target_body_xyz = tcp_body
        for axis_idx, slider in enumerate(state.target_sliders):
            slider.value = self._snap_to_step(float(tcp_body[axis_idx]), slider.step)
        state.target_marker.position = (
            float(tcp_world[0]),
            float(tcp_world[1]),
            float(tcp_world[2]),
        )

    def _solve_ik_to_target(self, side: ArmSide) -> None:
        """Solve IK for the arm to reach its current TCP target slider
        position, then push the solved joint config into the per-joint
        sliders (which are authoritative for tick's qpos write).

        Triggered by the per-arm `Solve IK → joints` button — keeps
        the IK solve as an explicit action the user invokes rather
        than a continuous behavior, which avoids surprise joint flips
        when the user is fine-tuning with joint sliders.
        """
        state = self._arm_states[side]
        arm = self.arms[side]
        bx, by, byaw = self._read_base_pose()
        target_world = self._body_to_world(state.target_body_xyz, bx, by, byaw)
        seed = np.array([float(s.value) for s in state.joint_sliders], dtype=float)
        try:
            solved_q, err_m = solve_ik(
                self.model,
                self.data,
                arm,
                target_world,
                orientation=PositionOnly(),
                seed_q=seed,
                locked_joint_names=self.locked_joint_names,
                solver="daqp",
                max_iters=200,
            )
        except RuntimeError:
            if self._phase_status_text is not None:
                self._phase_status_text.value = f"{side} IK: solver error"
            return
        for slider, value in zip(state.joint_sliders, solved_q, strict=True):
            slider.value = self._snap_to_step(float(value), slider.step)
        if self._phase_status_text is not None:
            self._phase_status_text.value = f"{side} IK: err={err_m * 1000:.1f} mm"

    def _reset_arm_to_home(self, side: ArmSide) -> None:
        """Snap one arm's joint sliders back to the scene's per-side home pose.

        Reads from `HOME_ARM_Q_BY_SIDE[side]` — the layout's per-arm
        rest configuration — so left and right arms restore to their
        own mirrored stow poses rather than a shared vector.
        """
        # Lazy import to avoid a circular dependency at module load
        # (teleop is imported by runner; runner imports scenes).
        from scenes.data_center_layout import HOME_ARM_Q_BY_SIDE

        home_q = HOME_ARM_Q_BY_SIDE[side]
        state = self._arm_states[side]
        if len(home_q) != len(state.joint_sliders):
            return
        for slider, value in zip(state.joint_sliders, home_q, strict=True):
            slider.value = self._snap_to_step(float(value), slider.step)
        if self._phase_status_text is not None:
            self._phase_status_text.value = f"{side} → home_q"

    def print_home_q(self) -> None:
        """Print the current per-arm joint configurations in a copy-
        pasteable form so the user can drop them into `_Arm.home_q`'s
        default factory in `scenes/data_center_layout.py`.
        """
        print("# --- paste into _Arm.home_q default_factory ---")
        print("{")
        for side, state in self._arm_states.items():
            joints = [round(float(s.value), 4) for s in state.joint_sliders]
            side_name = "LEFT" if side is ArmSide.LEFT else "RIGHT"
            print(f"    ArmSide.{side_name}: np.array({joints!r}),")
        print("}")
        if self._phase_status_text is not None:
            self._phase_status_text.value = "home_q values printed to runner stdout"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_to_json(self) -> Path:
        """Write captured keyframes to `<save_root>/<scene>/teleop_<UTC>.json`."""
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_dir = self._save_root / self._scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"teleop_{timestamp}.json"
        payload = {
            "scene": self._scene_name,
            "captured_at_utc": timestamp,
            "grippable_names": list(self.grippable_names),
            "arms": {
                str(side): [
                    {
                        "label": kf.label,
                        "phase": kf.phase.value,
                        "duration_s": kf.duration_s,
                        "arm_q": kf.arm_q.tolist(),
                        "gripper": kf.gripper,
                        "aux_ctrl": kf.aux_ctrl,
                        "attach_activate": list(kf.attach_activate),
                        "attach_deactivate": list(kf.attach_deactivate),
                        "weld_activate_idx": kf.weld_activate_idx,
                        "weld_deactivate_idx": kf.weld_deactivate_idx,
                    }
                    for kf in self._captured[side]
                ]
                for side in self.arms
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._last_save_text is not None:
            self._last_save_text.value = str(out_path)
        return out_path


# ---------------------------------------------------------------------------
# Replay loader
# ---------------------------------------------------------------------------


def load_recording(path: Path) -> Mapping[ArmSide, list[Step]]:
    """Read a teleop JSON recording and return a per-arm `Step` list.

    Reconstructs every captured field — `aux_ctrl` for base motion,
    `attach_activate` / `attach_deactivate` for weld toggles, and
    `weld_activate` / `weld_deactivate` for per-arm grasps. The
    runner replays these via the same code paths the scripted scene
    uses, so a teleoped trajectory replays identically to a scripted
    one.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arms_payload = payload["arms"]
    grippable_names: list[str] = list(payload.get("grippable_names") or [])
    n_cubes = len(grippable_names)

    def _maybe_cube_id(value: object) -> CubeID | None:
        # JSON-loaded indices arrive as `object` from json.load. We
        # parse-not-validate at the boundary: any non-None value is
        # coerced to int, then `make_cube_id` does the in-range check
        # and stamps the bounded `CubeID` brand. Downstream Step
        # construction can trust the result without re-checking.
        if value is None:
            return None
        return make_cube_id(int(str(value)), n_cubes)

    plan: dict[ArmSide, list[Step]] = {}
    for side_raw, keyframes in arms_payload.items():
        side = ArmSide(side_raw)
        plan[side] = [
            Step(
                label=str(kf["label"]),
                arm_q=np.asarray(kf["arm_q"], dtype=float),
                gripper=cast(GripperState, kf["gripper"]),
                duration=float(kf["duration_s"]),
                phase=TaskPhase(kf["phase"]),
                aux_ctrl=dict(kf.get("aux_ctrl") or {}) or None,
                attach_activate=tuple(kf.get("attach_activate") or ()),
                attach_deactivate=tuple(kf.get("attach_deactivate") or ()),
                weld_activate=_maybe_cube_id(kf.get("weld_activate_idx")),
                weld_deactivate=_maybe_cube_id(kf.get("weld_deactivate_idx")),
            )
            for kf in keyframes
        ]
    return plan
