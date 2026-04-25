"""Shared helpers for the `tools/` debug CLIs.

Every debug tool needs some subset of: "import a scene module by name",
"build + compile the spec", "advance the runner's task-plan timeline to
time t", "render from a camera or free-cam". This module owns that
plumbing so each tool stays a thin CLI wrapper focused on its own job
(render one frame, stitch video, grid-compose cameras, diff two PNGs,
sweep IK feasibility, print the task-plan timeline).

Importing from `tools._runtime` inside other tools keeps a single source
of truth for step interpretation: whatever the real runner does at a
step boundary (weld activate, attach deactivate, ctrl interpolation) is
replicated here once.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, NewType

# Set EGL before mujoco import so headless Linux rendering works without
# an X server. Tool invocations set this via env or we default it here.
os.environ.setdefault("MUJOCO_GL", "egl")

# Add the scene root (parent of tools/) to sys.path so tool scripts can
# import `arm_handles`, `scenes`, etc. regardless of cwd.
_SCENE_ROOT = Path(__file__).resolve().parent.parent
if str(_SCENE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCENE_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from arm_handles import ArmHandles, ArmSide, get_arm_handles  # noqa: E402
from scene_base import Step  # noqa: E402
from welds import (  # noqa: E402
    activate_attachment_weld,
    activate_grasp_weld,
    deactivate_grasp_weld,
)

SceneName = NewType("SceneName", str)
Seconds = NewType("Seconds", float)
AzimuthDeg = NewType("AzimuthDeg", float)
ElevationDeg = NewType("ElevationDeg", float)
Metres = NewType("Metres", float)
WorldPoint = tuple[float, float, float]


def load_scene(name: SceneName | str) -> ModuleType:
    """Import `scenes.<name>` — matches runner.py's convention so a
    --scene argument behaves identically between runner and tools."""
    return importlib.import_module(f"scenes.{name}")


def parse_world_point(raw: str, *, field_name: str) -> WorldPoint:
    """Parse `'x,y,z'` into a 3-tuple; raise with a useful message on
    anything else. CLI boundary parsing — downstream code gets a refined
    tuple and never re-validates."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"--{field_name} must be 'x,y,z' (three comma-separated numbers); got {raw!r}"
        )
    try:
        vals = [float(p) for p in parts]
    except ValueError as err:
        raise ValueError(f"--{field_name}: every component must be numeric; got {raw!r}") from err
    return (vals[0], vals[1], vals[2])


@dataclass(frozen=True)
class FreeCameraPose:
    """Orbit-camera pose in the convention the viser viewer uses."""

    azimuth_deg: AzimuthDeg
    elevation_deg: ElevationDeg
    distance_m: Metres
    lookat: WorldPoint


def build_free_cam(pose: FreeCameraPose) -> mujoco.MjvCamera:
    """Materialise a `FreeCameraPose` into MuJoCo's `MjvCamera` type."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(pose.azimuth_deg)
    cam.elevation = float(pose.elevation_deg)
    cam.distance = float(pose.distance_m)
    cam.lookat[:] = pose.lookat
    return cam


CameraSpec = mujoco.MjvCamera | str | None
"""Union the tools pass to `render_frame`:
 - MjvCamera: a free-cam pose
 - str: a named scene camera (e.g. 'top_d435i_cam')
 - None: MuJoCo's default free camera
"""


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera: CameraSpec = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Render a single frame. Returns an (H, W, 3) uint8 array.

    We deliberately don't call `renderer.close()` — MuJoCo 3.7's
    Renderer.__del__ calls close() at GC, and an explicit close()
    followed by __del__'s call raises `_mjr_context` AttributeError on
    the second pass. Letting GC handle cleanup avoids the noise without
    leaking resources (Python drops the reference when the function
    returns).
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    if isinstance(camera, str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
                for i in range(model.ncam)
            ]
            raise ValueError(f"unknown camera {camera!r}; available: {available}")
        renderer.update_scene(data, camera=cam_id)
    elif isinstance(camera, mujoco.MjvCamera):
        renderer.update_scene(data, camera=camera)
    else:
        renderer.update_scene(data)
    return renderer.render()


@dataclass
class _ArmTimelineState:
    step: int
    t: float
    start_q: np.ndarray
    start_g: float
    start_aux: dict[str, float]


def _advance_one_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
    script: list[Step],
    st: _ArmTimelineState,
    sim_dt: float,
    aux_name_to_id: dict[str, int],
) -> None:
    """Mirror of `runner.advance_arm` but stripped to headless
    replay — no speed slider, no GUI updates, just step ctrl and fire
    the weld/attach side effects at step boundaries."""
    if st.step >= len(script):
        return
    step = script[st.step]
    first_tick = st.t == 0.0
    st.t += sim_dt

    if first_tick:
        if step.weld_activate is not None:
            activate_grasp_weld(
                model,
                data,
                int(arm.weld_ids[step.weld_activate]),
                arm.link6_id,
                arm.link6_id,
                arm.tcp_site_id,
            )
        if step.weld_deactivate is not None:
            deactivate_grasp_weld(data, int(arm.weld_ids[step.weld_deactivate]))
        for weld_name in step.attach_activate:
            eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
            if int(model.eq_type[eq_id]) == int(mujoco.mjtEq.mjEQ_CONNECT):
                data.eq_active[eq_id] = 1
            else:
                activate_attachment_weld(
                    model,
                    data,
                    eq_id,
                    int(model.eq_obj1id[eq_id]),
                    int(model.eq_obj2id[eq_id]),
                )
        for weld_name, target_xyz, target_quat in step.attach_activate_at or ():
            eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
            activate_attachment_weld(
                model,
                data,
                eq_id,
                int(model.eq_obj1id[eq_id]),
                int(model.eq_obj2id[eq_id]),
                target_world_pose=(target_xyz, target_quat),
            )
        for weld_name in step.attach_deactivate:
            eq_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name))
            data.eq_active[eq_id] = 0

    alpha = min(1.0, st.t / max(step.duration, 1e-3))
    alpha_s = 0.5 - 0.5 * math.cos(math.pi * alpha)

    # Puppet mode: write qpos directly (mirrors runner.advance_arm).
    curr_q = (1.0 - alpha_s) * st.start_q + alpha_s * step.arm_q
    data.qpos[arm.arm_qpos_idx] = curr_q
    data.qvel[arm.arm_dof_idx] = 0.0
    data.ctrl[arm.act_arm_ids] = curr_q

    target_gripper = arm.gripper_open if step.gripper == "open" else arm.gripper_closed
    curr_g = (1.0 - alpha_s) * st.start_g + alpha_s * target_gripper
    data.qpos[arm.qpos_idx[6]] = curr_g
    data.qpos[arm.qpos_idx[7]] = -curr_g
    data.qvel[arm.dof_idx[6]] = 0.0
    data.qvel[arm.dof_idx[7]] = 0.0
    data.ctrl[arm.act_gripper_id] = curr_g

    if step.aux_ctrl:
        for aux_name, aux_target in step.aux_ctrl.items():
            aid = aux_name_to_id[aux_name]
            jnt_id = int(model.actuator_trnid[aid][0])
            qadr = int(model.jnt_qposadr[jnt_id])
            dadr = int(model.jnt_dofadr[jnt_id])
            start = st.start_aux.get(aux_name, float(data.qpos[qadr]))
            curr_aux = (1.0 - alpha_s) * start + alpha_s * aux_target
            data.qpos[qadr] = curr_aux
            data.qvel[dadr] = 0.0
            data.ctrl[aid] = curr_aux

    if alpha >= 1.0:
        st.start_q = step.arm_q.copy()
        st.start_g = target_gripper
        if step.aux_ctrl:
            for aux_name, aux_target in step.aux_ctrl.items():
                st.start_aux[aux_name] = aux_target
        st.step += 1
        st.t = 0.0


def advance_timeline(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    task_plan: dict[ArmSide, list[Step]],
    aux_name_to_id: dict[str, int],
    sim_dt: float,
    until_s: Seconds,
) -> None:
    """Replay the task plan's per-arm step loop until sim time reaches
    `until_s`. Every `sim_dt` runs one interpolation pass per arm then
    one `mj_step`. Mutates `data` in place."""
    state = {
        side: _ArmTimelineState(
            step=0,
            t=0.0,
            start_q=np.array([data.qpos[i] for i in arms[side].arm_qpos_idx]),
            start_g=float(data.ctrl[arms[side].act_gripper_id]),
            start_aux={},
        )
        for side in arms
    }
    n_steps = int(float(until_s) / sim_dt)
    for _ in range(n_steps):
        for side, arm in arms.items():
            _advance_one_arm(model, data, arm, task_plan[side], state[side], sim_dt, aux_name_to_id)
        mujoco.mj_step(model, data)


@dataclass
class SceneContext:
    """Handle returned by `build_scene_and_advance` so every tool
    reaches the same (model, data, arms, task_plan) quadruple without
    copy-pasting scene-initialisation code across six CLIs."""

    model: mujoco.MjModel
    data: mujoco.MjData
    arms: dict[ArmSide, ArmHandles]
    task_plan: dict[ArmSide, list[Step]] | None
    scene_module: ModuleType


def build_scene_and_advance(scene_name: SceneName | str, t: Seconds | float = 0.0) -> SceneContext:
    """Load + compile the scene, apply initial state, advance task plan
    to `t`. Every tool calls this instead of reimplementing the six-step
    init dance. Passing `t=0` skips task-plan construction entirely, so
    tools that only care about the home-pose spec (e.g. `tree.py`) don't
    pay the IK cost."""
    scene = load_scene(scene_name)
    spec = scene.build_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    arm_sides: tuple[ArmSide, ...] = getattr(scene, "ARM_PREFIXES", ())
    n_cubes: int = getattr(scene, "N_CUBES", 0)
    cube_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in getattr(scene, "GRIPPABLES", ())
    ]
    arms: dict[ArmSide, ArmHandles] = {
        side: get_arm_handles(model, side, n_cubes) for side in arm_sides
    }
    aux_name_to_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(scene, "AUX_ACTUATOR_NAMES", ())
    }
    scene.apply_initial_state(model, data, arms, cube_body_ids)

    task_plan: dict[ArmSide, list[Step]] | None = None
    if hasattr(scene, "make_task_plan"):
        # Build the plan unconditionally (the IK solve is cheap relative
        # to waiting for callers to ask for it and then rebuilding from
        # scratch). Re-apply initial state after the plan's `snap` calls
        # so t=0 renders show the home pose, not the IK seeding state.
        task_plan = scene.make_task_plan(model, data, arms, cube_body_ids)
        scene.apply_initial_state(model, data, arms, cube_body_ids)
    if task_plan is not None and float(t) > 0:
        sim_dt = float(model.opt.timestep)
        advance_timeline(model, data, arms, task_plan, aux_name_to_id, sim_dt, Seconds(float(t)))
    elif float(t) > 0:
        raise RuntimeError(f"scene {scene_name!r} has no make_task_plan, can't advance past t=0")

    mujoco.mj_forward(model, data)
    return SceneContext(model=model, data=data, arms=arms, task_plan=task_plan, scene_module=scene)


# Video encoder discrimination — the one place we translate a CLI
# filename suffix into the parsed output format. Downstream video-
# writing code takes `VideoFormat` and doesn't re-inspect the path.
VideoFormat = Literal["mp4", "gif"]


def parse_video_format(out_path: Path) -> VideoFormat:
    suffix = out_path.suffix.lower().lstrip(".")
    if suffix == "mp4":
        return "mp4"
    if suffix == "gif":
        return "gif"
    raise ValueError(f"unsupported video suffix {out_path.suffix!r}; use .mp4 or .gif")
