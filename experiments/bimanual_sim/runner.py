"""Generic scene runner.

Imports a scene module by name, compiles its spec, and plays it through
Viser. Two modes, picked by what the scene module exposes:

  * Task-planned scenes (`make_task_plan`): per-arm state machines advance in
    parallel, each interpolating its ctrl through a list of Steps. Weld
    activate/deactivate transitions fire on entry to the relevant step.
  * Free-play scenes (`step_free_play`): the scene's callback is invoked
    every render tick to set ctrl directly.

CLI:
    python runner.py --scene sink_bimanual [--host 127.0.0.1] [--port 8080]
                                           [--speed 1.0] [--render-hz 60]
                                           [--max-rate]

`--render-hz` caps how often the Viser scene and physics are advanced — the
browser can't tell the difference between 60 Hz and 125 Hz updates, but the
viser/websocket CPU cost scales linearly with the rate. `--max-rate` drops
the realtime throttle entirely so the sim runs as fast as MuJoCo can step
(useful for batch trajectory generation or video capture).

Connecting from a laptop when the runner is on a remote host: SSH-tunnel the
port, e.g. `ssh -L 8080:localhost:8080 user@host`, then open localhost:8080.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import viser

# Make sibling modules importable regardless of the invoking cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_handles import ArmHandles, ArmSide, get_arm_handles
from cameras import CameraRole, add_frustum_widgets, update_frustum_widgets
from scene_base import Step
from scene_check import AttachmentConstraint, CameraInvariant, check_scene, print_schematic
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
    # Remembers the most-recently committed target for each scene-owned aux
    # actuator, so interpolation can continue smoothly across steps. Keys are
    # actuator names (same as Step.aux_ctrl keys). Missing key ⇒ use current
    # data.ctrl at tick entry.
    start_aux: dict[str, float] = field(default_factory=dict)
    step: int = 0
    t: float = 0.0


def _load_scene(name: str):
    return importlib.import_module(f"scenes.{name}")


def _collect_cube_body_ids(model: mujoco.MjModel, n_cubes: int) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube{i}") for i in range(n_cubes)]


def _extract_attachment_constraints(scene) -> tuple[AttachmentConstraint, ...]:
    """Read a scene's `ATTACHMENTS` tuple — already a tuple of the
    `WeldAttachment | ConnectAttachment` union from `scene_check`. Scenes
    without the registry get an empty tuple, and `check_scene` silently
    skips the attachment validations."""
    return tuple(getattr(scene, "ATTACHMENTS", ()))


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
    args = parser.parse_args()

    scene = _load_scene(args.scene)
    print(f"Building scene: {getattr(scene, 'NAME', args.scene)}")

    model, data = scene.build_spec()
    print(
        f"compiled: nbody={model.nbody} njnt={model.njnt} nu={model.nu} "
        f"neq={model.neq} ngeom={model.ngeom}"
    )

    arm_sides: tuple[ArmSide, ...] = getattr(scene, "ARM_PREFIXES", ())
    n_cubes: int = getattr(scene, "N_CUBES", 0)
    cube_body_ids = _collect_cube_body_ids(model, n_cubes)
    arms: dict[ArmSide, ArmHandles] = {
        side: get_arm_handles(model, side, n_cubes) for side in arm_sides
    }

    # Scene-owned actuators (e.g. a lift prismatic). Resolved once at startup;
    # Step.aux_ctrl entries refer to these by name. We also resolve the
    # underlying joint qpos/qvel addresses for puppet-mode direct writes.
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
        # Aux actuators must be JOINT-transmission position actuators
        # (e.g. torso_lift_joint). actuator_trnid[aid][0] is the joint id.
        jnt_id = int(model.actuator_trnid[aid][0])
        aux_qposadr[name] = int(model.jnt_qposadr[jnt_id])
        aux_dofadr[name] = int(model.jnt_dofadr[jnt_id])

    scene.apply_initial_state(model, data, arms, cube_body_ids)

    # ---- Compile-time sanity checks (always-on; see scene_check) ------------
    # Runs before `make_task_plan` so IK-unfriendly geometry shows up as a
    # scene error rather than an IK-residual-too-large panic. Scenes opt in to
    # additional descriptors (allow-listed overlaps, the attachment-constraint
    # registry, grippable-name list) via module-level attributes that we read
    # here with `getattr` so scenes that don't declare them still work.
    grippable_names: tuple[str, ...] = getattr(scene, "GRIPPABLES", ())
    allowed_overlaps: tuple[tuple[str, str], ...] = getattr(scene, "ALLOWED_STATIC_OVERLAPS", ())
    attachment_constraints = _extract_attachment_constraints(scene)
    camera_invariants: tuple[CameraInvariant, ...] = getattr(scene, "CAMERA_INVARIANTS", ())

    if args.inspect:
        # --inspect: print the schematic, run checks (which may raise), exit
        # before physics/viser. Print happens first so the user always sees
        # the body/geom tree, even if check_scene is about to raise.
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
    if hasattr(scene, "make_task_plan"):
        print("Solving IK waypoints...")
        task_plan = scene.make_task_plan(model, data, arms, cube_body_ids)
        scene.apply_initial_state(model, data, arms, cube_body_ids)

    has_free_play = hasattr(scene, "step_free_play")
    if task_plan is None and not has_free_play:
        print(
            "warning: scene provides neither make_task_plan nor step_free_play; "
            "arms will hold their initial pose"
        )

    # Viser
    server = viser.ViserServer(host=args.host, port=args.port)
    # `build_viser_scene` now needs `data` so it can bake the initial world
    # pose of static geoms into `add_mesh_simple` and drop them from the
    # per-frame update list. Callers must have finished `apply_initial_state`
    # by this point (we have, above).
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

    control = {"playing": True, "reset_requested": False}

    @gui_play.on_click
    def _on_play(_event):
        control["playing"] = not control["playing"]

    @gui_reset.on_click
    def _on_reset(_event):
        control["reset_requested"] = True

    sim_dt = float(model.opt.timestep)
    # Decouple render rate from the physics timestep. Every frame we step
    # physics `phys_steps_per_frame` times so wall-clock advance = render_dt.
    # At the default 60 Hz and a 2 ms mj timestep that's 8 steps/frame — half
    # the per-second viser work of the previous 125 Hz loop for the same
    # simulated-time budget.
    render_dt = 1.0 / max(args.render_hz, 1e-3)
    phys_steps_per_frame = max(1, int(round(render_dt / sim_dt)))
    # Re-derive render_dt from the rounded step count so the physics clock and
    # the sleep throttle agree exactly.
    render_dt = sim_dt * phys_steps_per_frame
    print(
        f"render: {1.0 / render_dt:.1f} Hz "
        f"({phys_steps_per_frame} physics steps × {sim_dt * 1000:.1f} ms)"
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
        scene.apply_initial_state(model, data, arms, cube_body_ids)
        per_arm = fresh_state()

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
            # Attachment equalities. Two kinds, branched by eq_type:
            #   WELD — freeze current relative pose (no teleport) so the
            #     body stays put when the constraint flips on.
            #   CONNECT — anchor is already baked into eq_data at build
            #     time; activation is a simple flag flip. The solver will
            #     pull body_b's origin to the anchor over the next few
            #     steps (small snap if the bodies were close).
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

        # Puppet mode: write joint qpos directly (no PD tracking, no
        # gravity droop, no overshoot). Zero qvel so mj_step's integrator
        # doesn't advance qpos away from where we put it. Mirror ctrl so
        # the position actuators don't fight the qpos write with residual
        # PD force from a stale ctrl value.
        curr_q = (1.0 - alpha_s) * st.start_q + alpha_s * step.arm_q
        data.qpos[arm.arm_qpos_idx] = curr_q
        data.qvel[arm.arm_dof_idx] = 0.0
        data.ctrl[arm.act_arm_ids] = curr_q

        tgt_g = arm.gripper_open if step.gripper == "open" else arm.gripper_closed
        curr_g = (1.0 - alpha_s) * st.start_g + alpha_s * tgt_g
        # Gripper joint7 (index 6) and joint8 (index 7) are tendon-coupled
        # mirrors; write both so mj_step's tendon constraint has nothing
        # to enforce.
        data.qpos[arm.qpos_idx[6]] = curr_g
        data.qpos[arm.qpos_idx[7]] = -curr_g
        data.qvel[arm.dof_idx[6]] = 0.0
        data.qvel[arm.dof_idx[7]] = 0.0
        data.ctrl[arm.act_gripper_id] = curr_g

        # Scene-owned auxiliary actuators (e.g. lift). Multiple arms may
        # write the same aux on overlapping steps; last write wins per tick
        # — scenes are expected to keep their targets consistent.
        if step.aux_ctrl:
            for aux_name, aux_target in step.aux_ctrl.items():
                aid = aux_name_to_id[aux_name]
                start = st.start_aux.get(aux_name, float(data.qpos[aux_qposadr[aux_name]]))
                curr_aux = (1.0 - alpha_s) * start + alpha_s * aux_target
                data.qpos[aux_qposadr[aux_name]] = curr_aux
                data.qvel[aux_dofadr[aux_name]] = 0.0
                data.ctrl[aid] = curr_aux

        label = step.label
        if alpha >= 1.0:
            st.start_q = step.arm_q.copy()
            st.start_g = tgt_g
            if step.aux_ctrl:
                for aux_name, aux_target in step.aux_ctrl.items():
                    st.start_aux[aux_name] = aux_target
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

    try:
        while True:
            if control["reset_requested"]:
                restart()
                sim_t = 0.0
                control["reset_requested"] = False

            if control["playing"]:
                if task_plan is not None:
                    for side in arm_sides:
                        per_arm_gui[side].value = advance_arm(side, render_dt)
                    if all_done():
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
            sim_t += render_dt

            update_viser(server, model, data, handles)
            if frustum_handles:
                update_frustum_widgets(server, data, frustum_handles)

            if not args.max_rate:
                next_tick += render_dt
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
