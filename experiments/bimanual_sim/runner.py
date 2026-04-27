"""Generic scene runner.

Imports a scene module by name, compiles its spec, and plays it through Viser.
Two modes, picked by what the scene module exposes:

  * Task-planned (`make_task_plan`): per-arm state machines advance in
    parallel, each interpolating ctrl through a list of Steps. Weld
    activate/deactivate transitions fire on entry to the relevant step.
  * Free-play (`step_free_play`): the scene's callback is invoked every
    render tick to set ctrl directly.

`--render-hz` caps the viser+physics update rate; the browser doesn't see
60 Hz vs 125 Hz, but the websocket CPU cost scales linearly. `--max-rate`
drops the realtime throttle so the sim runs as fast as MuJoCo can step.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
import viser

# Make sibling modules importable regardless of the invoking cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_handles import ArmHandles, ArmSide, arm_joint_suffixes, get_arm_handles
from cameras import CameraRole, add_frustum_widgets, update_frustum_widgets
from phase_monitor import PhaseContractViolation, PhaseRuntimeMonitor
from rerun_stream import RerunStreamer
from scene_base import PhaseContract, Step, TaskPhase
from scene_check import AttachmentConstraint, CameraInvariant, check_scene, print_schematic
from teleop import TeleopController
from viser_render import build_viser_scene, update_viser
from welds import (
    activate_attachment_weld,
    activate_grasp_weld,
    deactivate_grasp_weld,
    deactivate_weld,
)


@dataclass
class ArmTimelineState:
    """Per-arm progress through its Step list. Each arm advances independently."""

    start_q: np.ndarray
    start_g: float
    # Most-recently committed target per scene-owned aux actuator, so
    # interpolation continues smoothly across steps. Missing key ⇒ use current
    # data.ctrl at tick entry.
    start_aux: dict[str, float] = field(default_factory=dict)
    step: int = 0
    t: float = 0.0


def _load_scene(name: str) -> ModuleType:
    return importlib.import_module(f"scenes.{name}")


def _collect_cube_body_ids(model: mujoco.MjModel, n_cubes: int) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube{i}") for i in range(n_cubes)]


def _extract_attachment_constraints(scene: Any) -> tuple[AttachmentConstraint, ...]:
    return tuple(getattr(scene, "ATTACHMENTS", ()))


def _apply_initial_state(
    scene: ModuleType,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
    *,
    start_phase: TaskPhase | None,
) -> None:
    """Forward to `scene.apply_initial_state`, passing `start_phase` only when
    the scene declares the kwarg. Pre-feature scenes have a 4-arg signature;
    passing an unknown kwarg would crash them."""
    sig = inspect.signature(scene.apply_initial_state)
    if "start_phase" in sig.parameters:
        scene.apply_initial_state(model, data, arms, cube_body_ids, start_phase=start_phase)
    else:
        if start_phase is not None:
            print(
                f"[runner] scene {scene.NAME!r} does not support --start-phase; "
                f"booting at scene home."
            )
        scene.apply_initial_state(model, data, arms, cube_body_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Scene module name under scenes/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--render-hz",
        type=float,
        default=45.0,
        help=(
            "viser + physics update rate; physics timestep is still the scene's"
            " mj timestep. Default 45 Hz trades 60→45 of server-side updates for"
            " ~25%% less websocket / message-buffer CPU; the browser renders"
            " between updates via WebGL so the visual result stays smooth."
        ),
    )
    parser.add_argument(
        "--max-rate",
        action="store_true",
        help="skip the realtime sleep throttle; run physics as fast as CPU allows",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "build + compile the scene, run scene_check, print a body/geom "
            "schematic to stdout, and exit before physics/viser. Pair with a "
            "layout edit to confirm the scene still passes invariants."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "if the scene exposes PHASE_CONTRACTS, raise the moment any "
            "contract (boundary expectation or in-phase invariant) fails. "
            "Without --strict, contract failures are collected and printed "
            "at exit but the demo keeps running."
        ),
    )
    parser.add_argument(
        "--rerun-port",
        type=int,
        default=None,
        help=(
            "open a rerun gRPC server on localhost:PORT and stream qpos / "
            "body transforms / phase events live. For EC2→laptop streaming, "
            "SSH-tunnel with `-L PORT:localhost:PORT` then connect a local "
            "rerun viewer to `rerun+http://localhost:PORT`."
        ),
    )
    parser.add_argument(
        "--rerun-connect",
        type=str,
        default=None,
        help=(
            "push events to an already-running rerun viewer at URL "
            "(e.g. `rerun+http://laptop.local:9876`). Mutually exclusive "
            "with --rerun-port."
        ),
    )
    parser.add_argument(
        "--rerun-rrd",
        type=Path,
        default=None,
        help="write the same rerun stream to a `.rrd` file for offline replay.",
    )
    parser.add_argument(
        "--rerun-camera-every",
        type=int,
        default=0,
        help=(
            "if >0, log named-camera frames every N render ticks (e.g. 5 → "
            "9 Hz at the default 45 Hz render rate). 0 disables camera "
            "logging — recommended over slow SSH tunnels."
        ),
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help=(
            "skip the scripted task plan; instead expose viser drag handles "
            "for each arm's TCP and run live IK. Use the in-browser GUI to "
            "capture keyframes per phase and save them to JSON for later "
            "replay via --play-recording."
        ),
    )
    parser.add_argument(
        "--play-recording",
        type=Path,
        default=None,
        help=(
            "load a teleop JSON recording (saved by --teleop) and replay "
            "it as the task plan. Mutually exclusive with --teleop."
        ),
    )
    parser.add_argument(
        "--start-phase",
        type=str,
        default=None,
        help=(
            "boot the scene at a hand-authored phase START state instead of "
            "scene home. Pass a TaskPhase value (case-insensitive, e.g. "
            "'remove_old_server' or 'REMOVE_OLD_SERVER'). The pose comes "
            "from the scene layout's PHASE_HOMES map (populated via teleop's "
            "'Print phase homes for layout' button). Compatible with both "
            "--teleop and --play-recording — boot mid-demo and either "
            "author further from there or replay onward."
        ),
    )
    args = parser.parse_args()
    if args.teleop and args.play_recording is not None:
        parser.error("--teleop and --play-recording are mutually exclusive")
    start_phase: TaskPhase | None = None
    if args.start_phase is not None:
        # Case-insensitive — accept 'REMOVE_OLD_SERVER' or 'remove_old_server'.
        try:
            start_phase = TaskPhase(args.start_phase.lower())
        except ValueError:
            valid = ", ".join(p.value for p in TaskPhase)
            parser.error(f"--start-phase {args.start_phase!r} not in TaskPhase. Valid: {valid}")
    if args.rerun_port is not None and args.rerun_connect is not None:
        parser.error("--rerun-port and --rerun-connect are mutually exclusive")

    scene = _load_scene(args.scene)
    print(f"Building scene: {getattr(scene, 'NAME', args.scene)}")

    model, data = scene.build_spec()
    print(
        f"compiled: nbody={model.nbody} njnt={model.njnt} nu={model.nu} "
        f"neq={model.neq} ngeom={model.ngeom}"
    )

    arm_sides: tuple[ArmSide, ...] = getattr(scene, "ARM_PREFIXES", ())
    n_cubes: int = getattr(scene, "N_CUBES", 0)
    # Default piper for backward compatibility with scenes pre-dating dispatch.
    robot_kind = getattr(scene, "ROBOT_KIND", "piper")
    cube_body_ids = _collect_cube_body_ids(model, n_cubes)
    arms: dict[ArmSide, ArmHandles] = {
        side: get_arm_handles(model, side, n_cubes, robot_kind) for side in arm_sides
    }

    # Scene-owned actuators (e.g. a lift prismatic). qpos/qvel addresses are
    # resolved alongside ids so puppet-mode can do direct writes.
    aux_name_to_id: dict[str, int] = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(scene, "AUX_ACTUATOR_NAMES", ())
    }
    for name, aid in aux_name_to_id.items():
        if aid < 0:
            raise ValueError(
                f"scene declared aux actuator {name!r} but no such actuator in compiled model"
            )
    aux_qposadr: dict[str, int] = {}
    aux_dofadr: dict[str, int] = {}
    for name, aid in aux_name_to_id.items():
        # Aux actuators must be JOINT-transmission position actuators.
        jnt_id = int(model.actuator_trnid[aid][0])
        aux_qposadr[name] = int(model.jnt_qposadr[jnt_id])
        aux_dofadr[name] = int(model.jnt_dofadr[jnt_id])

    _apply_initial_state(scene, model, data, arms, cube_body_ids, start_phase=start_phase)

    # check_scene runs before make_task_plan so geometry bugs surface as
    # scene errors instead of IK-residual panics.
    grippable_names: tuple[str, ...] = getattr(scene, "GRIPPABLES", ())
    allowed_overlaps: tuple[tuple[str, str], ...] = getattr(scene, "ALLOWED_STATIC_OVERLAPS", ())
    attachment_constraints = _extract_attachment_constraints(scene)
    camera_invariants: tuple[CameraInvariant, ...] = getattr(scene, "CAMERA_INVARIANTS", ())

    if args.inspect:
        # Print first so the user sees the body/geom tree even if check_scene
        # is about to raise.
        print_schematic(
            model,
            data,
            arms=arms,
            grippable_names=grippable_names,
            attachment_constraints=attachment_constraints,
        )
        check_scene(
            model,
            data,
            arms=arms,
            grippable_names=grippable_names,
            allowed_static_overlaps=allowed_overlaps,
            attachment_constraints=attachment_constraints,
            camera_invariants=camera_invariants,
        )
        print("\ncheck_scene: OK")
        return

    check_scene(
        model,
        data,
        arms=arms,
        grippable_names=grippable_names,
        allowed_static_overlaps=allowed_overlaps,
        attachment_constraints=attachment_constraints,
        camera_invariants=camera_invariants,
    )

    task_plan: dict[ArmSide, list[Step]] | None = None
    if args.teleop:
        print("Teleop mode: dragging TCP handles drives live IK.")
    elif args.play_recording is not None:
        from teleop import load_recording

        loaded = load_recording(args.play_recording)
        task_plan = {side: list(steps) for side, steps in loaded.items()}
        _apply_initial_state(scene, model, data, arms, cube_body_ids, start_phase=start_phase)
        print(f"Replaying recording: {args.play_recording}")
        for side in arm_sides:
            print(f"  [{side}] {len(task_plan[side])} steps loaded")
    elif hasattr(scene, "make_task_plan"):
        print("Solving IK waypoints...")
        task_plan = scene.make_task_plan(model, data, arms, cube_body_ids)
        _apply_initial_state(scene, model, data, arms, cube_body_ids, start_phase=start_phase)

    phase_contracts: tuple[PhaseContract, ...] = getattr(scene, "PHASE_CONTRACTS", ())
    phase_monitor = PhaseRuntimeMonitor(phase_contracts, strict=args.strict)
    if phase_monitor.enabled:
        print(
            f"PhaseRuntimeMonitor active ({len(phase_contracts)} contracts, strict={args.strict})"
        )

    # One source feeds at most one rerun sink (gRPC serve, gRPC connect, .rrd).
    rerun_streamer: RerunStreamer | None = None
    if args.rerun_port is not None:
        rerun_streamer = RerunStreamer.serve_grpc(scene_name=args.scene, grpc_port=args.rerun_port)
    elif args.rerun_connect is not None:
        rerun_streamer = RerunStreamer.connect_grpc(scene_name=args.scene, url=args.rerun_connect)
    elif args.rerun_rrd is not None:
        rerun_streamer = RerunStreamer.save_rrd(scene_name=args.scene, rrd_path=args.rerun_rrd)

    # Cache body ids + joint suffixes once so per-tick rerun logging skips
    # name resolution.
    rerun_body_ids: dict[str, int] = {}
    rerun_joint_names: dict[ArmSide, tuple[str, ...]] = {}
    if rerun_streamer is not None:
        for grippable_name in grippable_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grippable_name)
            if bid >= 0:
                rerun_body_ids[grippable_name] = bid
        for side, arm in arms.items():
            wrist_id = arm.link6_id
            if wrist_id >= 0:
                rerun_body_ids[f"{side.rstrip('/')}/wrist"] = wrist_id
        suffixes = arm_joint_suffixes(robot_kind)
        for side in arm_sides:
            rerun_joint_names[side] = suffixes

    rerun_tick_counter = {"n": 0}

    has_free_play = hasattr(scene, "step_free_play")
    if task_plan is None and not has_free_play and not args.teleop:
        print(
            "warning: scene provides neither make_task_plan nor step_free_play; "
            "arms will hold their initial pose"
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    # `build_viser_scene` reads `data` to bake initial static-geom poses, so
    # `apply_initial_state` must already have run.
    handles = build_viser_scene(server, model, data)

    cameras: tuple[tuple[str, CameraRole], ...] = getattr(scene, "CAMERAS", ())
    frustum_handles = add_frustum_widgets(server, model, data, cameras) if cameras else []

    gui_state = server.gui.add_text("state", initial_value="ready", disabled=True)
    gui_speed = server.gui.add_slider(
        "speed", min=0.1, max=3.0, step=0.1, initial_value=float(args.speed)
    )
    gui_play = server.gui.add_button("▶ play / ⏸ pause")
    gui_reset = server.gui.add_button("↺ reset")

    per_arm_gui: dict[ArmSide, viser.GuiTextHandle] = {}
    for side in arm_sides:
        per_arm_gui[side] = server.gui.add_text(
            side.rstrip("_") or "arm", initial_value="-", disabled=True
        )

    teleop_controller: TeleopController | None = None
    if args.teleop:
        # Teleop drives the chassis from a base handle, so it requires the
        # scene to expose a (base_x, base_y, base_yaw) triple.
        teleop_locked = tuple(getattr(scene, "IK_LOCKED_JOINT_NAMES", ()))
        if len(teleop_locked) != 3:
            raise SystemExit(
                f"--teleop requires the scene to expose IK_LOCKED_JOINT_NAMES as a "
                f"3-tuple (base_x, base_y, base_yaw); got {teleop_locked!r}"
            )
        a, b, c = teleop_locked
        teleop_base_aux: tuple[str, str, str] = (str(a), str(b), str(c))
        teleop_controller = TeleopController.attach(
            server,
            model=model,
            data=data,
            arms=arms,
            locked_joint_names=teleop_locked,
            attachments=attachment_constraints,
            phase_contracts=phase_contracts,
            grippable_names=grippable_names,
            cube_body_ids=tuple(cube_body_ids),
            base_aux_names=teleop_base_aux,
            scene_name=args.scene,
        )

    control = {"playing": True, "reset_requested": False}

    @gui_play.on_click
    def _on_play(_event: Any) -> None:
        control["playing"] = not control["playing"]

    @gui_reset.on_click
    def _on_reset(_event: Any) -> None:
        control["reset_requested"] = True

    sim_dt = float(model.opt.timestep)
    # Decouple render rate from physics timestep: each frame we step physics
    # `phys_steps_per_frame` times so wall-clock advance = render_dt.
    render_dt = 1.0 / max(args.render_hz, 1e-3)
    phys_steps_per_frame = max(1, round(render_dt / sim_dt))
    # Re-derive render_dt from the rounded step count so the physics clock and
    # the sleep throttle agree exactly.
    render_dt = sim_dt * phys_steps_per_frame
    print(
        f"render: {1.0 / render_dt:.1f} Hz "
        f"({phys_steps_per_frame} physics steps x {sim_dt * 1000:.1f} ms)"
        + (" [max-rate: throttle off]" if args.max_rate else "")
    )

    def fresh_state() -> dict[ArmSide, ArmTimelineState]:
        return {
            side: ArmTimelineState(
                start_q=np.array([data.qpos[i] for i in arms[side].arm_qpos_idx]),
                start_g=float(data.ctrl[arms[side].act_gripper_id]),
            )
            for side in arm_sides
        }

    per_arm: dict[ArmSide, ArmTimelineState] = fresh_state()

    def restart() -> None:
        nonlocal per_arm
        _apply_initial_state(scene, model, data, arms, cube_body_ids, start_phase=start_phase)
        per_arm = fresh_state()
        phase_monitor.reset()

    def advance_arm(side: ArmSide, dt: float) -> str:
        assert task_plan is not None
        script = task_plan[side]
        st = per_arm[side]
        arm = arms[side]

        if st.step >= len(script):
            return "done"

        step = script[st.step]
        duration = step.duration / max(gui_speed.value, 0.05)

        first_tick = st.t == 0.0
        st.t += dt

        if first_tick:
            # Phase guard fires before weld toggles so phase-end checks reflect
            # the previous step's leftover state, and phase-start baselines
            # capture before the incoming step mutates anything.
            phase_monitor.on_step_started(step, model, data)
            if step.weld_activate is not None:
                activate_grasp_weld(
                    model,
                    data,
                    int(arm.weld_ids[step.weld_activate]),
                    arm.link6_id,
                    cube_body_ids[step.weld_activate],
                    arm.tcp_site_id,
                )
            if step.weld_deactivate is not None:
                deactivate_grasp_weld(data, int(arm.weld_ids[step.weld_deactivate]))
            # WELD freezes the current relpose (no teleport); CONNECT is a
            # simple flag flip — the anchor is already in eq_data.
            for weld_name in step.attach_activate:
                eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
                if eq_id < 0:
                    raise ValueError(f"step '{step.label}': unknown attach weld {weld_name!r}")
                if int(model.eq_type[eq_id]) == int(mujoco.mjtEq.mjEQ_CONNECT):
                    data.eq_active[eq_id] = 1
                else:
                    body_a = int(model.eq_obj1id[eq_id])
                    body_b = int(model.eq_obj2id[eq_id])
                    activate_attachment_weld(model, data, eq_id, body_a, body_b)
            for weld_name, target_xyz, target_quat in step.attach_activate_at or ():
                eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
                if eq_id < 0:
                    raise ValueError(f"step '{step.label}': unknown attach_at weld {weld_name!r}")
                if int(model.eq_type[eq_id]) == int(mujoco.mjtEq.mjEQ_CONNECT):
                    raise ValueError(
                        f"step '{step.label}': attach_activate_at requires a WELD "
                        f"equality, got CONNECT for {weld_name!r}"
                    )
                body_a = int(model.eq_obj1id[eq_id])
                body_b = int(model.eq_obj2id[eq_id])
                activate_attachment_weld(
                    model,
                    data,
                    eq_id,
                    body_a,
                    body_b,
                    target_world_pose=(target_xyz, target_quat),
                )
            for weld_name in step.attach_deactivate:
                eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
                if eq_id < 0:
                    raise ValueError(f"step '{step.label}': unknown attach weld {weld_name!r}")
                deactivate_weld(data, eq_id)

        alpha = min(1.0, st.t / max(duration, 1e-3))
        alpha_s = 0.5 - 0.5 * math.cos(math.pi * alpha)

        # Puppet mode: direct qpos write. Zero qvel so mj_step doesn't drift
        # the integrator off the puppet target; mirror ctrl so position
        # actuators don't fight back with stale PD force.
        curr_q = (1.0 - alpha_s) * st.start_q + alpha_s * step.arm_q
        data.qpos[arm.arm_qpos_idx] = curr_q
        data.qvel[arm.arm_dof_idx] = 0.0
        data.ctrl[arm.act_arm_ids] = curr_q

        tgt_g = arm.gripper_open if step.gripper == "open" else arm.gripper_closed
        curr_g = (1.0 - alpha_s) * st.start_g + alpha_s * tgt_g
        if arm.robot_kind == "piper":
            # Piper joint7/8 are tendon-coupled finger slides. Puppet-write
            # both so the tendon equality has nothing to enforce.
            data.qpos[arm.qpos_idx[6]] = curr_g
            data.qpos[arm.qpos_idx[7]] = -curr_g
            data.qvel[arm.dof_idx[6]] = 0.0
            data.qvel[arm.dof_idx[7]] = 0.0
        # UR10e + 2F-85: ctrl drives the tendon equality; the 4-bar linkage
        # joints settle on their own.
        data.ctrl[arm.act_gripper_id] = curr_g

        # Multiple arms may write the same aux on overlapping steps; last
        # write wins — scenes are expected to keep their targets consistent.
        if step.aux_ctrl:
            for aux_name, aux_target in step.aux_ctrl.items():
                aux_key = str(aux_name)
                aid = aux_name_to_id[aux_key]
                start = st.start_aux.get(aux_key, float(data.qpos[aux_qposadr[aux_key]]))
                curr_aux = (1.0 - alpha_s) * start + alpha_s * aux_target
                data.qpos[aux_qposadr[aux_key]] = curr_aux
                data.qvel[aux_dofadr[aux_key]] = 0.0
                data.ctrl[aid] = curr_aux

        label = step.label
        if alpha >= 1.0:
            st.start_q = step.arm_q.copy()
            st.start_g = tgt_g
            if step.aux_ctrl:
                for aux_name, aux_target in step.aux_ctrl.items():
                    st.start_aux[str(aux_name)] = aux_target
            st.step += 1
            st.t = 0.0
            return f"{st.step}/{len(script)} {label} ✓"
        return f"{st.step + 1}/{len(script)} {label}"

    def all_done() -> bool:
        if task_plan is None:
            return False
        return all(per_arm[side].step >= len(task_plan[side]) for side in arm_sides)

    print(f"Viser on {args.host}:{args.port} ({len(handles)} geoms)")
    print(
        f"If remote: `ssh -L {args.port}:localhost:{args.port} user@host` "
        f"then open http://localhost:{args.port}"
    )
    if task_plan is not None:
        parts = ", ".join(f"{side}={len(task_plan[side])}" for side in arm_sides)
        print(f"Timeline: {parts} steps (run in parallel)")

    next_tick = time.perf_counter()
    sim_t = 0.0

    plan_finished_announced = False

    try:
        while True:
            if control["reset_requested"]:
                restart()
                sim_t = 0.0
                plan_finished_announced = False
                control["reset_requested"] = False

            # Teleop tick is unconditional: dragging a handle must move the
            # arm even when the sim is paused, since teleop is for authoring.
            if teleop_controller is not None:
                teleop_controller.tick(render_dt)
                gui_state.value = "teleop"
            elif control["playing"]:
                if task_plan is not None:
                    for side in arm_sides:
                        per_arm_gui[side].value = advance_arm(side, render_dt)
                    if all_done():
                        if not plan_finished_announced:
                            phase_monitor.on_plan_finished(model, data)
                            plan_finished_announced = True
                        gui_state.value = "done — press reset"
                        control["playing"] = False
                    else:
                        gui_state.value = "running"
                elif has_free_play:
                    scene.step_free_play(sim_t, model, data)
                    gui_state.value = "running"
                else:
                    gui_state.value = "idle"
            else:
                gui_state.value = "paused" if not all_done() else "done — press reset"

            for _ in range(phys_steps_per_frame):
                mujoco.mj_step(model, data)
                phase_monitor.on_tick(model, data)
            sim_t += render_dt

            update_viser(server, model, data, handles)
            if frustum_handles:
                update_frustum_widgets(server, data, frustum_handles)

            # Log once per render tick: rerun is for user-visible debugging,
            # so per-physics-tick is wasteful.
            if rerun_streamer is not None:
                rerun_streamer.set_sim_time(sim_t)
                for side in arm_sides:
                    arm = arms[side]
                    qpos = np.asarray([data.qpos[i] for i in arm.arm_qpos_idx], dtype=float)
                    rerun_streamer.log_joint_scalars(
                        side_prefix=str(side),
                        joint_names=rerun_joint_names[side],
                        qpos=qpos,
                    )
                for body_name, body_id in rerun_body_ids.items():
                    rerun_streamer.log_body_transform(
                        name=body_name,
                        xpos=np.asarray(data.xpos[body_id], dtype=float),
                        xquat=np.asarray(data.xquat[body_id], dtype=float),
                    )
                rerun_tick_counter["n"] += 1

            if not args.max_rate:
                next_tick += render_dt
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("stopped")
    except PhaseContractViolation as exc:
        print(f"\n[strict] phase contract violation:\n  {exc}")
        sys.exit(2)
    finally:
        if phase_monitor.enabled and phase_monitor.failures:
            print(f"\nPhaseRuntimeMonitor: {len(phase_monitor.failures)} contract failure(s):")
            for failure in phase_monitor.failures:
                print(f"  [{failure.kind}] {failure.name}: {failure.message}")


if __name__ == "__main__":
    main()
