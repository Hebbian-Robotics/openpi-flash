"""Render the indicator-check demo from each named camera + a directorial cut.

Produces four mp4s:
- `forward.mp4`   — chassis-mounted top camera, yaws with the base
- `left_wrist.mp4`  — D405 on the left Piper's wrist
- `right_wrist.mp4` — D405 on the right Piper's wrist
- `directorial.mp4` — two static shots cut on phase boundaries:
    establishing aisle (drive-in) → behind-the-shoulder (yaw, reach, wait, retract)

Headless: uses `tools._runtime` helpers (which set `MUJOCO_GL=egl` on Linux)
so this runs over SSH without a display server. No viser involvement.

Usage:
    cd ~/sim
    uv run python tools/render_pov_videos.py --out-dir /tmp/pov

Output mp4s play at real-time speed (1 sim-second = 1 wall-second at the
chosen fps) so the four files can be ffmpeg-stitched without rate matching.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

# Pick a headless GL backend BEFORE the first mujoco import — EC2 has no
# display server, so the GLFW default crashes with `gladLoadGL error`.
# `tools._runtime` does this too, but we duplicate it here because Python
# evaluates the `import mujoco` below before the `from tools._runtime`
# line runs (transitive imports of imageio / numpy may pull mujoco in).
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

# Bootstrap project root so `from tools._runtime import …` resolves when
# running this file as a script (Python puts only `tools/` on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v3 as iio
import mujoco
import numpy as np
import typer

from tools._runtime import (
    Seconds,
    advance_timeline_with_state,
    build_scene_and_advance,
    make_timeline_state,
)

SCENE = "mobile_aloha_piper_indicator_check"


class FfmpegPreset(StrEnum):
    """libx264 preset values accepted by ffmpeg."""

    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"


@dataclass(frozen=True)
class FixedCameraPass:
    """Render pass for one compiled scene camera."""

    filename: str
    camera_id: int


@dataclass(frozen=True)
class DirectorialCameraPass:
    """Render pass whose camera is derived from directorial cuts per frame."""

    filename: str


CameraPass = FixedCameraPass | DirectorialCameraPass


# Directorial cuts: (start_t, cam_position, look_at), all in world frame.
# Pose tuples come from the runner's "📸 log cam pose" button — paste
# them here as (pos, lookat) pairs and the renderer cuts to each one
# on its `start_t`. Same coord system viser exposes, no conversion.
#
# Phase boundaries (sim seconds), from make_task_plan:
#   SETUP 0.0 → 1.5
#   TRAVERSE 1.5 → 11.5
#   ALIGN 11.5 → 17.5
#   REACH 17.5 → 20.5
#   WAIT 20.5 → 25.0
#   RETRACT 25.0 → 27.0
_DIRECTORIAL_CUTS: tuple[
    tuple[float, tuple[float, float, float], tuple[float, float, float]], ...
] = (
    # 0 - 11.5 s: establishing wide of the aisle / robot driving in.
    # Zoomed out to 1.6x the user's captured distance: same lookat, same
    # azimuth/elevation direction, just pulled further back so the scene
    # reads as a wide aisle establishing rather than a medium shot.
    (0.0, (-1.312, -1.617, 2.945), (2.322, -0.839, 0.614)),
    # 11.5 s - end: robot stopped before the turn; over-the-shoulder of
    # the chassis as it yaws, steps, reaches, waits, and retracts. Held
    # for the rest of the demo per user request - no third cut.
    (11.5, (4.174, -0.837, 2.294), (5.258, -0.069, 0.853)),
)


def _camera_from_pose(
    position: tuple[float, float, float],
    look_at: tuple[float, float, float],
) -> mujoco.MjvCamera:
    """Convert a (position, lookat) world-frame pose into an MjvCamera.

    MuJoCo's free camera doesn't accept a position directly — it uses
    (azimuth, elevation, distance, lookat) where `position` is computed
    as `lookat - distance * (cos(el)cos(az), cos(el)sin(az), sin(el))`.
    Note the leading minus: MuJoCo treats az/el as the direction the
    camera is *looking* (toward lookat), not the direction *from* lookat
    to the camera. Verified empirically — setting (az=0, el=0, dist=5,
    lookat=origin) puts the camera at world (-5, 0, 0), not (+5, 0, 0).
    Inverting that mapping is one normalize + two trig calls.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    pos = np.asarray(position, dtype=float)
    look = np.asarray(look_at, dtype=float)
    offset = look - pos
    distance = float(np.linalg.norm(offset))
    if distance < 1e-6:
        raise ValueError(f"camera position equals lookat: {position!r}")
    unit = offset / distance
    cam.azimuth = float(np.degrees(np.arctan2(unit[1], unit[0])))
    cam.elevation = float(np.degrees(np.arcsin(unit[2])))
    cam.distance = distance
    cam.lookat[:] = look
    return cam


def _build_directorial_camera(target_t: float) -> mujoco.MjvCamera:
    """Pick a fixed cinematic pose for the current sim time.

    Cuts (not interpolations) — switching shots on phase boundaries gives
    the viewer a stable frame per beat instead of a moving cam that's hard
    to track. The active cut is the one whose `start_t` is the latest
    that's still ≤ target_t.
    """
    active_pose = _DIRECTORIAL_CUTS[0]
    for cut in _DIRECTORIAL_CUTS:
        if cut[0] <= target_t:
            active_pose = cut
        else:
            break
    _, pos, lookat = active_pose
    return _camera_from_pose(pos, lookat)


def _fixed_camera_pass(
    model: mujoco.MjModel, *, filename: str, camera_name: str
) -> FixedCameraPass:
    """Parse a scene camera name into a render pass with a compiled camera id."""
    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
    if camera_id < 0:
        raise typer.BadParameter(f"camera {camera_name!r} not found in compiled model")
    return FixedCameraPass(filename=filename, camera_id=camera_id)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Annotated[Path, typer.Option(help="directory for the four .mp4 files")] = Path(
        "/tmp/pov"
    ),
    fps: Annotated[int, typer.Option(min=1, help="playback fps")] = 30,
    # Default dimensions: full HD. macro_block_size=1 in the imwrite
    # call below disables libx264's 16-block alignment, so 1080 doesn't
    # get auto-padded to 1088 with a thin black bar.
    width: Annotated[int, typer.Option(min=1, help="render width px")] = 1920,
    height: Annotated[int, typer.Option(min=1, help="render height px")] = 1080,
    duration_s: Annotated[
        float,
        typer.Option(min=0.0, help="sim seconds to render; 0 → full task plan length"),
    ] = 0.0,
    crf: Annotated[
        int,
        typer.Option(min=0, max=51, help="libx264 CRF (lower = better; 18 ≈ visually lossless)"),
    ] = 18,
    preset: Annotated[FfmpegPreset, typer.Option(help="libx264 preset")] = FfmpegPreset.SLOWER,
) -> None:
    """Render the demo from each scene camera + a directorial cut.

    Memory: each cam is rendered + written to disk in its own pass over
    the timeline, so peak frame-buffer occupancy is one camera's worth of
    raw frames (≈ duration_s * fps * width * height * 3 bytes). At the
    1920x1080 / 30 fps default that's ≈ 5 GB per camera over a 27 s
    demo — fits comfortably in EC2 RAM. Concatenating all four into a
    single shared buffer would 4x that.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_scene_and_advance(SCENE, Seconds(0.0))
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {SCENE!r} has no make_task_plan")

    if duration_s <= 0.0:
        duration_s = max(
            sum(step.duration for step in ctx.task_plan[side]) for side in ctx.task_plan
        )
    typer.echo(
        f"rendering {duration_s:.2f}s of {SCENE!r} at {fps} fps, "
        f"{width}x{height}, crf={crf}, preset={preset.value}"
    )

    sim_dt = float(ctx.model.opt.timestep)
    aux_name_to_id = {
        n: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in getattr(ctx.scene_module, "AUX_ACTUATOR_NAMES", ())
    }

    fixed_cam_specs = (
        ("forward.mp4", "forward_cam"),
        ("left_wrist.mp4", "left/wrist_d405_cam"),
        ("right_wrist.mp4", "right/wrist_d405_cam"),
    )
    cam_passes: list[CameraPass] = [
        _fixed_camera_pass(ctx.model, filename=filename, camera_name=camera_name)
        for filename, camera_name in fixed_cam_specs
    ]
    cam_passes.append(DirectorialCameraPass(filename="directorial.mp4"))

    # Single Renderer instance reused across every frame x every camera.
    # `tools._runtime.render_frame` allocates a fresh Renderer per call,
    # which leaks EGL contexts on EC2. One instance + per-call
    # `update_scene` is also significantly faster (no MjrContext re-init).
    renderer = mujoco.Renderer(ctx.model, height=height, width=width)
    # Hide debug overlays we don't want in the rendered video: site
    # markers (the "balls" at TCP / arm-mount points), camera frustums,
    # and contact widgets. Frustums are off because the recorded view is
    # itself a camera, so a frustum widget at the cam origin would render
    # over the frame.
    scene_option = mujoco.MjvOption()
    # Sites (TCP markers, mount sites) are controlled by `sitegroup`, not
    # an mjVIS flag — there is no `mjVIS_SITE`. Zeroing every group hides
    # them all. Geom group 0 is the default-visible bucket; leave it on
    # so actual scene geometry still renders.
    scene_option.sitegroup[:] = 0
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # x264 params are passed straight through imageio_ffmpeg → ffmpeg.
    # `-pix_fmt yuv420p` keeps the file playable in browsers / QuickTime.
    # `-g`/`-keyint_min` force one keyframe per second (default GOP is
    # 250 frames ≈ 8 s at 30 fps, which made software decoders choke
    # mid-task during high-motion sections — the player couldn't recover
    # until the next I-frame). 1 s GOP costs a few % filesize for
    # significantly smoother seeking + playback resilience.
    ffmpeg_params = [
        "-crf",
        str(crf),
        "-preset",
        preset.value,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
    ]

    # Frame timing uses integer step accounting to avoid the drift that
    # bites a naive `dt = target_t - prev_t; advance_timeline(...dt...)`
    # loop: with sim_dt=0.002 and frame interval 1/30=0.0333..., each call
    # to `advance_timeline_with_state` runs `int(0.0333/0.002)=16` mini-
    # steps = 0.032 s, losing ~1 ms per frame. Over a 27 s render that
    # accumulates to ~0.8 s of "missing" task time at the end (arms only
    # half-finish the retract). Tracking absolute step counts keeps the
    # video's last frame at exactly task_duration_s of sim time.
    # Snapshot initial geom_rgba so we can restore it between passes.
    # `Step.set_geom_rgba` mutates `model.geom_rgba` in place during a
    # pass (e.g. the alert flip from red to green at t=23.5), and
    # `apply_initial_state` only resets `data` (qpos/qvel/ctrl) — it
    # doesn't touch the model. Without this snapshot+restore, pass 2
    # onward starts with a green alert (carryover from pass 1's flip)
    # and never shows the red→green transition. Mirrors the
    # `initial_geom_rgba` / `restart()` logic in runner.py.
    initial_geom_rgba = ctx.model.geom_rgba.copy()

    n_frames = int(duration_s * fps) + 1
    for camera_pass in cam_passes:
        # Re-create timeline state at the start of each pass — the four
        # passes run independently, so each begins at sim t=0 with the
        # same initial qpos / ctrl AND the same initial geom colours.
        ctx.model.geom_rgba[:] = initial_geom_rgba
        ctx.scene_module.apply_initial_state(
            ctx.model, ctx.data, ctx.arms, getattr(ctx, "cube_body_ids", []) or []
        )
        state = make_timeline_state(ctx.data, ctx.arms)
        frames: list[np.ndarray] = []
        prev_steps = 0
        for f in range(n_frames):
            target_t = f / fps
            target_steps = round(target_t / sim_dt)
            delta_steps = target_steps - prev_steps
            if delta_steps > 0:
                advance_timeline_with_state(
                    ctx.model,
                    ctx.data,
                    ctx.arms,
                    ctx.task_plan,
                    state,
                    aux_name_to_id,
                    sim_dt,
                    Seconds(delta_steps * sim_dt),
                )
                mujoco.mj_forward(ctx.model, ctx.data)
                prev_steps = target_steps
            match camera_pass:
                case DirectorialCameraPass():
                    camera = _build_directorial_camera(target_t)
                    renderer.update_scene(ctx.data, camera=camera, scene_option=scene_option)
                case FixedCameraPass(camera_id=camera_id):
                    renderer.update_scene(ctx.data, camera=camera_id, scene_option=scene_option)
            frames.append(renderer.render().copy())
        out_path = out_dir / camera_pass.filename
        iio.imwrite(
            out_path,
            frames,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=ffmpeg_params,
        )
        typer.echo(f"  → {out_path} ({len(frames)} frames)")
        del frames


if __name__ == "__main__":
    app()
