"""Unified debug CLI for the bimanual sim.

One entrypoint for rendering, timeline inspection, phase checks, and review
artifacts. Replaces the per-tool scripts:

    uv run python tools/mj.py snapshot --out /tmp/home.png
    uv run python tools/mj.py snapshot --t 22 --camera top_d435i_cam --out /tmp/cable.png
    uv run python tools/mj.py snapshot --t 30 --every 0.5 --out-prefix /tmp/run_
    uv run python tools/mj.py video    --prefix /tmp/run_ --fps 20 --out /tmp/run.mp4
    uv run python tools/mj.py grid     --t 22 --out /tmp/grid.png
    uv run python tools/mj.py plan
    uv run python tools/mj.py contracts --out-root results/runs
    uv run python tools/mj.py phase remove_old_server --out-root results/runs
    uv run python tools/mj.py diff     --a /tmp/a.png --b /tmp/b.png --out /tmp/d.png
    uv run python tools/mj.py ik

Run `... tools/mj.py --help` or `... tools/mj.py <subcommand> --help` for
the full option list.

All subcommands default to `--scene mobile_aloha_ur10e_server_swap`. Rendering goes through
MuJoCo's native `mujoco.Renderer` (EGL backend on headless Linux; the
GL driver already offloads to the GPU on hosts that expose one, no
explicit switch needed).
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

# Pick a safe MuJoCo GL backend before importing mujoco. Linux EC2 needs
# EGL for headless rendering; macOS rejects `egl`, so use GLFW there.
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

# Bootstrap the project root onto sys.path before importing the sibling
# `tools._runtime` module — running `python tools/mj.py` only puts
# `tools/` on the path, so the project-root import we need for
# scene modules must be added by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v3 as iio
import mujoco
import numpy as np
import typer

from rerun_stream import RerunStreamer
from scene_base import PhaseContract, PhaseState, Step, TaskPhase
from tools._runtime import (
    AzimuthDeg,
    CameraSpec,
    ElevationDeg,
    FreeCameraPose,
    Metres,
    SceneContext,
    Seconds,
    TimelineState,
    advance_timeline_with_state,
    build_free_cam,
    build_scene_and_advance,
    load_scene,
    make_timeline_state,
    parse_video_format,
    parse_world_point,
    render_frame,
)
from tools.observability import (
    ContractCheckReport,
    PhaseBoundary,
    RunArtifactWriter,
    check_phase_state,
    make_run_directory,
    phase_contract_to_json_dict,
)

app = typer.Typer(
    help=__doc__,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ---- shared option aliases ---------------------------------------------------


SceneOpt = Annotated[str, typer.Option(help="scene module under scenes/")]
TOpt = Annotated[float, typer.Option("--t", help="sim time to advance to (seconds)")]
WidthOpt = Annotated[int, typer.Option(help="render width in px")]
HeightOpt = Annotated[int, typer.Option(help="render height in px")]
AzOpt = Annotated[float, typer.Option("--az", help="free-cam azimuth (deg)")]
ElOpt = Annotated[float, typer.Option("--el", help="free-cam elevation (deg)")]
DistOpt = Annotated[float, typer.Option("--dist", help="free-cam distance (m)")]
LookatOpt = Annotated[str, typer.Option("--lookat", help="free-cam lookat 'x,y,z'")]


def _resolve_camera(
    camera: str | None,
    az: float,
    el: float,
    dist: float,
    lookat: str,
) -> CameraSpec:
    """`--camera` takes precedence; otherwise construct a free-cam pose
    from the orbit knobs. Mutually exclusive by construction."""
    if camera is not None:
        return camera
    pose = FreeCameraPose(
        azimuth_deg=AzimuthDeg(az),
        elevation_deg=ElevationDeg(el),
        distance_m=Metres(dist),
        lookat=parse_world_point(lookat, field_name="lookat"),
    )
    return build_free_cam(pose)


@dataclass(frozen=True)
class PhaseWindow:
    """Timeline interval covered by one high-level task phase."""

    phase: TaskPhase
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _aux_name_to_id(ctx: SceneContext) -> dict[str, int]:
    return {
        name: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(ctx.scene_module, "AUX_ACTUATOR_NAMES", ())
    }


def _parse_task_phase(raw_phase: str) -> TaskPhase:
    normalized = raw_phase.strip().lower()
    for phase in TaskPhase:
        if normalized in {phase.value, phase.name.lower()}:
            return phase
    available = ", ".join(phase.value for phase in TaskPhase if phase is not TaskPhase.UNPHASED)
    raise typer.BadParameter(f"unknown phase {raw_phase!r}; available: {available}")


def _phase_contracts_by_phase(
    phase_contracts: tuple[PhaseContract, ...],
) -> dict[TaskPhase, PhaseContract]:
    return {contract.phase: contract for contract in phase_contracts}


def _phase_windows(task_plan: Mapping[Any, list[Step]]) -> dict[TaskPhase, PhaseWindow]:
    windows: dict[TaskPhase, tuple[float, float]] = {}
    for script in task_plan.values():
        elapsed = 0.0
        for step in script:
            step_start = elapsed
            step_end = elapsed + step.duration
            elapsed = step_end
            if step.phase is TaskPhase.UNPHASED:
                continue
            if step.phase not in windows:
                windows[step.phase] = (step_start, step_end)
                continue
            current_start, current_end = windows[step.phase]
            windows[step.phase] = (min(current_start, step_start), max(current_end, step_end))
    return {
        phase: PhaseWindow(phase=phase, start_s=start_s, end_s=end_s)
        for phase, (start_s, end_s) in windows.items()
    }


def _sorted_phase_windows(task_plan: Mapping[Any, list[Step]]) -> list[PhaseWindow]:
    return sorted(_phase_windows(task_plan).values(), key=lambda window: window.start_s)


def _render_phase_boundary(
    ctx: SceneContext,
    writer: RunArtifactWriter,
    *,
    phase: TaskPhase,
    boundary: PhaseBoundary,
    camera_spec: CameraSpec,
    width: int,
    height: int,
) -> Path:
    frame = render_frame(ctx.model, ctx.data, camera=camera_spec, width=width, height=height)
    path = writer.run_dir / "renders" / f"{phase.value}_{boundary}.png"
    iio.imwrite(path, frame)
    return path


def _write_optional_rerun_phase_log(
    rrd_path: Path | None,
    *,
    scene_name: str,
    phase_events: list[dict[str, Any]],
) -> None:
    """Write phase-boundary text events to an `.rrd` file for offline replay.

    Thin wrapper around `rerun_stream.RerunStreamer.save_rrd` so the
    contract / phase subcommands share the same lifecycle code as the
    live runner. The streamer raises a clear error if `rerun-sdk` is
    missing.
    """
    if rrd_path is None:
        return
    try:
        streamer = RerunStreamer.save_rrd(scene_name=scene_name, rrd_path=rrd_path)
    except RuntimeError as err:
        raise typer.BadParameter(str(err)) from err

    for event in phase_events:
        streamer.set_sim_time(float(event["time_s"]))
        streamer.log_phase_event(
            phase=event["phase"],
            boundary=event["boundary"],
            contract_ok=bool(event["contract_ok"]),
            message="ok" if event["contract_ok"] else "failed",
        )


def _advance_context(
    ctx: SceneContext,
    duration_s: float,
    timeline_state: TimelineState | None = None,
) -> TimelineState:
    actual_timeline_state = timeline_state or make_timeline_state(ctx.data, ctx.arms)
    if duration_s <= 0:
        mujoco.mj_forward(ctx.model, ctx.data)
        return actual_timeline_state
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {ctx.scene_module.NAME!r} has no make_task_plan")
    advance_timeline_with_state(
        ctx.model,
        ctx.data,
        ctx.arms,
        ctx.task_plan,
        actual_timeline_state,
        _aux_name_to_id(ctx),
        float(ctx.model.opt.timestep),
        Seconds(duration_s),
    )
    mujoco.mj_forward(ctx.model, ctx.data)
    return actual_timeline_state


def _record_phase_boundary(
    ctx: SceneContext,
    writer: RunArtifactWriter,
    *,
    phase: TaskPhase,
    boundary: PhaseBoundary,
    phase_state: PhaseState,
    camera_spec: CameraSpec | None,
    width: int,
    height: int,
    base_tolerance: float,
) -> tuple[ContractCheckReport, dict[str, Any]]:
    report = check_phase_state(
        ctx.model,
        ctx.data,
        phase_state,
        base_tolerance=base_tolerance,
    )
    snapshot_path = writer.write_snapshot(
        ctx.model,
        ctx.data,
        phase=phase,
        boundary=boundary,
        phase_state=phase_state,
        contract_report=report,
    )
    render_path = None
    if camera_spec is not None:
        render_path = _render_phase_boundary(
            ctx,
            writer,
            phase=phase,
            boundary=boundary,
            camera_spec=camera_spec,
            width=width,
            height=height,
        )
    event = {
        "phase": phase.value,
        "boundary": boundary,
        "time_s": float(ctx.data.time),
        "contract_ok": report.ok,
        "snapshot": str(snapshot_path),
        "render": str(render_path) if render_path is not None else None,
        "failures": [failure.message for failure in report.failures],
    }
    return report, event


# ---- snapshot ----------------------------------------------------------------


@app.command()
def snapshot(
    out: Annotated[
        Path | None,
        typer.Option(help="single-frame output path (.png); omit with --out-prefix"),
    ] = None,
    out_prefix: Annotated[
        str | None,
        typer.Option(
            "--out-prefix",
            help="sequence-mode prefix; frames named {prefix}{sec:06.2f}.png",
        ),
    ] = None,
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
    t: TOpt = 0.0,
    every: Annotated[
        float,
        typer.Option(help="sequence mode: emit one frame per N seconds up to --t"),
    ] = 0.0,
    camera: Annotated[
        str | None,
        typer.Option(help="named scene camera (unset = free-cam with --az/--el/…)"),
    ] = None,
    width: WidthOpt = 640,
    height: HeightOpt = 480,
    az: AzOpt = 135.0,
    el: ElOpt = -20.0,
    dist: DistOpt = 2.5,
    lookat: LookatOpt = "0.30,0.0,0.9",
) -> None:
    """Render a scene to PNG — single frame or time-lapse sequence."""
    camera_spec = _resolve_camera(camera, az, el, dist, lookat)
    sequence_mode = every > 0.0

    if sequence_mode and out_prefix is None:
        raise typer.BadParameter("--every requires --out-prefix")
    if sequence_mode and t <= 0:
        raise typer.BadParameter("--every requires --t > 0")
    if not sequence_mode and out is None:
        raise typer.BadParameter("single-frame mode requires --out PATH")

    if not sequence_mode:
        ctx = build_scene_and_advance(scene, Seconds(t))
        frame = render_frame(ctx.model, ctx.data, camera=camera_spec, width=width, height=height)
        assert out is not None  # narrowed by the typer.BadParameter check above
        iio.imwrite(out, frame)
        typer.echo(f"rendered → {out}")
        return

    ctx = build_scene_and_advance(scene, Seconds(0.0))
    if ctx.task_plan is None:
        raise typer.BadParameter(
            f"scene {scene!r} has no make_task_plan — sequence mode needs a timeline"
        )
    aux_name_to_id = {
        name: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(ctx.scene_module, "AUX_ACTUATOR_NAMES", ())
    }
    sim_dt = float(ctx.model.opt.timestep)
    prev_t = 0.0
    timeline_state = make_timeline_state(ctx.data, ctx.arms)
    assert out_prefix is not None
    for target_t in np.arange(0.0, t + 1e-9, every):
        dt = float(target_t) - prev_t
        if dt > 0:
            advance_timeline_with_state(
                ctx.model,
                ctx.data,
                ctx.arms,
                ctx.task_plan,
                timeline_state,
                aux_name_to_id,
                sim_dt,
                Seconds(dt),
            )
            mujoco.mj_forward(ctx.model, ctx.data)
        frame = render_frame(ctx.model, ctx.data, camera=camera_spec, width=width, height=height)
        path = Path(f"{out_prefix}{float(target_t):06.2f}.png")
        iio.imwrite(path, frame)
        typer.echo(f"{float(target_t):5.2f}s → {path}")
        prev_t = float(target_t)


# ---- video -------------------------------------------------------------------


@app.command()
def video(
    prefix: Annotated[str, typer.Option(help="path prefix; stitches {prefix}*.png")],
    out: Annotated[Path, typer.Option(help="output file (.mp4 or .gif)")],
    fps: Annotated[float, typer.Option(help="playback frames per second")] = 20.0,
) -> None:
    """Stitch a sequence of PNGs into an mp4 or gif."""
    prefix_path = Path(prefix)
    search_dir = prefix_path.parent if str(prefix_path.parent) else Path()
    stem = prefix_path.name
    frames = sorted(search_dir.glob(f"{stem}*.png"))
    if not frames:
        raise typer.Exit(code=1)
    fmt = parse_video_format(out)
    typer.echo(f"stitching {len(frames)} frames → {out} ({fmt})")
    images = [iio.imread(frame) for frame in frames]
    if fmt == "gif":
        iio.imwrite(out, images, duration=int(1000 / fps), loop=0)
    else:
        iio.imwrite(out, images, fps=int(fps), codec="libx264")
    typer.echo(f"wrote {out}")


# ---- grid --------------------------------------------------------------------


_GRID_FONT: dict[str, tuple[str, ...]] = {
    "a": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "b": ("#### ", "#   #", "#### ", "#   #", "#   #", "#   #", "#### "),
    "c": (" ####", "#    ", "#    ", "#    ", "#    ", "#    ", " ####"),
    "d": ("#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "),
    "e": ("#####", "#    ", "#### ", "#    ", "#    ", "#    ", "#####"),
    "f": ("#####", "#    ", "#### ", "#    ", "#    ", "#    ", "#    "),
    "g": (" ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ####"),
    "h": ("#   #", "#   #", "#####", "#   #", "#   #", "#   #", "#   #"),
    "i": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"),
    "j": ("    #", "    #", "    #", "    #", "    #", "#   #", " ### "),
    "k": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "l": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "m": ("#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"),
    "n": ("#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #"),
    "o": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "p": ("#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "),
    "q": (" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"),
    "r": ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    "s": (" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "),
    "t": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "u": ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "v": ("#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "),
    "w": ("#   #", "#   #", "#   #", "#   #", "# # #", "# # #", " # # "),
    "x": ("#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"),
    "y": ("#   #", "#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  "),
    "z": ("#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"),
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "# #  ", "  #  ", "  #  ", "  #  ", "#####"),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": ("#### ", "    #", "    #", " ### ", "    #", "    #", "#### "),
    "4": ("#   #", "#   #", "#   #", "#####", "    #", "    #", "    #"),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": (" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "),
    "_": ("     ", "     ", "     ", "     ", "     ", "     ", "#####"),
    "-": ("     ", "     ", "     ", "#####", "     ", "     ", "     "),
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
}


def _draw_label(grid: np.ndarray, top: int, left: int, width: int, height: int, text: str) -> None:
    """Paint `text` into the top strip of one grid tile. Uses an inline
    5x7 pixel font to avoid pulling PIL in just for labels."""
    glyph_w, glyph_h = 5, 7
    scale = max(1, (height - 4) // glyph_h)
    cursor = left + 4
    baseline = top + (height - glyph_h * scale) // 2
    for ch in text.lower():
        glyph = _GRID_FONT.get(ch)
        if glyph is None:
            cursor += (glyph_w + 1) * scale
            continue
        for gy, row in enumerate(glyph):
            for gx, cell in enumerate(row):
                if cell == "#":
                    ry = baseline + gy * scale
                    rx = cursor + gx * scale
                    grid[ry : ry + scale, rx : rx + scale] = (230, 230, 230)
        cursor += (glyph_w + 1) * scale
        if cursor >= left + width - glyph_w * scale:
            break


def _tile_grid(images: list[np.ndarray], labels: list[str], label_height: int = 22) -> np.ndarray:
    """Square-ish auto layout. For fixed columns see `_tile_grid_cols`."""
    cols = math.ceil(math.sqrt(len(images)))
    return _tile_grid_cols(images, labels, cols=cols, label_height=label_height)


def _tile_grid_cols(
    images: list[np.ndarray],
    labels: list[str],
    *,
    cols: int,
    label_height: int = 22,
) -> np.ndarray:
    n = len(images)
    rows = math.ceil(n / cols)
    tile_h, tile_w, channels = images[0].shape
    full_tile_h = tile_h + label_height
    grid = np.full((rows * full_tile_h, cols * tile_w, channels), 32, dtype=np.uint8)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        r, c = divmod(i, cols)
        grid[
            r * full_tile_h + label_height : (r + 1) * full_tile_h,
            c * tile_w : (c + 1) * tile_w,
        ] = image
        _draw_label(grid, r * full_tile_h, c * tile_w, tile_w, label_height, label)
    return grid


@app.command()
def grid(
    out: Annotated[Path, typer.Option(help="grid output path (.png)")],
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
    t: TOpt = 0.0,
    cams: Annotated[
        str | None,
        typer.Option(help="comma-separated camera names; default = every scene camera"),
    ] = None,
    no_free_cam: Annotated[
        bool, typer.Option("--no-free-cam/--free-cam", help="omit the free-cam tile")
    ] = False,
    width: WidthOpt = 480,
    height: HeightOpt = 360,
    az: AzOpt = 135.0,
    el: ElOpt = -20.0,
    dist: DistOpt = 2.5,
    lookat: LookatOpt = "0.30,0.0,0.9",
) -> None:
    """Render several cameras at one sim time and tile them into a grid."""
    ctx = build_scene_and_advance(scene, Seconds(t))
    scene_cams = (
        [c.strip() for c in cams.split(",")]
        if cams
        else [
            mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
            for i in range(ctx.model.ncam)
        ]
    )
    cameras: list[CameraSpec] = list(scene_cams)
    labels = list(scene_cams)
    if not no_free_cam:
        cameras.append(
            build_free_cam(
                FreeCameraPose(
                    azimuth_deg=AzimuthDeg(az),
                    elevation_deg=ElevationDeg(el),
                    distance_m=Metres(dist),
                    lookat=parse_world_point(lookat, field_name="lookat"),
                )
            )
        )
        labels.append("free_cam")
    images = [
        render_frame(ctx.model, ctx.data, camera=cam, width=width, height=height) for cam in cameras
    ]
    iio.imwrite(out, _tile_grid(images, labels))
    typer.echo(f"rendered {len(images)} tiles → {out}")


# ---- plan --------------------------------------------------------------------


@app.command()
def plan(
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
) -> None:
    """Print a scene's task plan as a timeline table."""
    ctx = build_scene_and_advance(scene, Seconds(0.0))
    scene_mod = ctx.scene_module
    cube_body_ids = [
        mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in getattr(scene_mod, "GRIPPABLES", ())
    ]
    task_plan = scene_mod.make_task_plan(ctx.model, ctx.data, ctx.arms, cube_body_ids)
    phase_contracts = getattr(scene_mod, "PHASE_CONTRACTS", ())
    if phase_contracts:
        _print_phase_contracts(phase_contracts)
        typer.echo("")

    label_w = (
        max((len(step.label) for script in task_plan.values() for step in script), default=10) + 1
    )
    phase_w = (
        max(
            (len(step.phase.value) for script in task_plan.values() for step in script),
            default=5,
        )
        + 1
    )
    widths = {
        "side": 7,
        "start": 7,
        "dur": 5,
        "phase": phase_w,
        "grip": 6,
        "label": label_w,
        "aplus": 32,
        "aminus": 32,
        "wplus": 5,
        "wminus": 5,
    }
    header = (
        f"{'side':<{widths['side']}} {'start':>{widths['start']}} "
        f"{'dur':>{widths['dur']}} {'phase':<{widths['phase']}} "
        f"{'grip':<{widths['grip']}} "
        f"{'label':<{widths['label']}} {'attach+':<{widths['aplus']}} "
        f"{'attach-':<{widths['aminus']}} {'weld+':<{widths['wplus']}} "
        f"{'weld-':<{widths['wminus']}}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for side, script in task_plan.items():
        elapsed = 0.0
        for step in script:
            typer.echo(_format_plan_row(side, elapsed, step, widths))
            elapsed += step.duration
        typer.echo("")


def _print_phase_contracts(phase_contracts: tuple[PhaseContract, ...]) -> None:
    typer.echo("phase contracts")
    typer.echo("-" * 80)
    for contract in phase_contracts:
        typer.echo(f"[{contract.phase.value}]")
        typer.echo(f"  start: {contract.starts.description}")
        typer.echo(f"  end:   {contract.ends.description}")
        start_facts = _format_phase_state_facts(contract.starts)
        end_facts = _format_phase_state_facts(contract.ends)
        if start_facts:
            typer.echo(f"  start facts: {start_facts}")
        if end_facts:
            typer.echo(f"  end facts:   {end_facts}")


def _format_phase_state_facts(state: PhaseState) -> str:
    parts: list[str] = []
    if state.active_attachments:
        parts.append("active=" + ",".join(str(name) for name in state.active_attachments))
    if state.inactive_attachments:
        parts.append("inactive=" + ",".join(str(name) for name in state.inactive_attachments))
    if state.base_aux:
        parts.append("base=" + ",".join(f"{name}:{value:+.2f}" for name, value in state.base_aux))
    return " | ".join(parts)


def _format_plan_row(
    side: str,
    start_s: float,
    step: Step,
    widths: dict[str, int],
) -> str:
    def fmt_attach(items: tuple[str, ...]) -> str:
        return ",".join(str(i) for i in items) if items else "-"

    def fmt_weld(idx: int | None) -> str:
        return "-" if idx is None else str(idx)

    return (
        f"{side:<{widths['side']}} "
        f"{start_s:>{widths['start']}.2f} "
        f"{step.duration:>{widths['dur']}.2f} "
        f"{step.phase.value:<{widths['phase']}} "
        f"{step.gripper:<{widths['grip']}} "
        f"{step.label:<{widths['label']}} "
        f"{fmt_attach(step.attach_activate):<{widths['aplus']}} "
        f"{fmt_attach(step.attach_deactivate):<{widths['aminus']}} "
        f"{fmt_weld(step.weld_activate):<{widths['wplus']}} "
        f"{fmt_weld(step.weld_deactivate):<{widths['wminus']}}"
    )


# ---- phase contracts + replay ------------------------------------------------


@app.command()
def contracts(
    out_root: Annotated[
        Path,
        typer.Option(help="root directory for structured run artifacts"),
    ] = Path("results/runs"),
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
    render: Annotated[
        bool,
        typer.Option("--render/--no-render", help="write a PNG at every checked boundary"),
    ] = True,
    rerun_rrd: Annotated[
        Path | None,
        typer.Option(
            "--rerun-rrd",
            help="optional .rrd file with phase-boundary TextLog events",
        ),
    ] = None,
    base_tolerance: Annotated[
        float,
        typer.Option(help="absolute tolerance for base qpos contract checks"),
    ] = 1e-3,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error/--no-fail-on-error", help="exit non-zero on contract failure"
        ),
    ] = True,
    camera: Annotated[
        str | None,
        typer.Option(help="named scene camera for boundary renders"),
    ] = None,
    width: WidthOpt = 960,
    height: HeightOpt = 720,
    az: AzOpt = 40.0,
    el: ElOpt = -15.0,
    dist: DistOpt = 2.2,
    lookat: LookatOpt = "0.30,0.0,0.9",
) -> None:
    """Replay the full task and assert every declared phase boundary."""
    ctx = build_scene_and_advance(scene, Seconds(0.0))
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {scene!r} has no make_task_plan")

    phase_contracts: tuple[PhaseContract, ...] = tuple(
        getattr(ctx.scene_module, "PHASE_CONTRACTS", ())
    )
    if not phase_contracts:
        raise typer.BadParameter(f"scene {scene!r} does not define PHASE_CONTRACTS")

    run_dir = make_run_directory(out_root, scene_name=scene)
    writer = RunArtifactWriter(run_dir, scene_name=scene)
    writer.write_phase_contracts(phase_contracts)
    writer.write_event("run_start", command="contracts")

    camera_spec = _resolve_camera(camera, az, el, dist, lookat) if render else None
    contracts_by_phase = _phase_contracts_by_phase(phase_contracts)
    timeline_state = make_timeline_state(ctx.data, ctx.arms)
    current_time_s = 0.0
    phase_events: list[dict[str, Any]] = []
    all_ok = True

    for window in _sorted_phase_windows(ctx.task_plan):
        contract = contracts_by_phase.get(window.phase)
        if contract is None:
            continue
        timeline_state = _advance_context(ctx, window.start_s - current_time_s, timeline_state)
        current_time_s = window.start_s
        start_report, start_event = _record_phase_boundary(
            ctx,
            writer,
            phase=window.phase,
            boundary="start",
            phase_state=contract.starts,
            camera_spec=camera_spec,
            width=width,
            height=height,
            base_tolerance=base_tolerance,
        )
        phase_events.append(start_event)

        timeline_state = _advance_context(ctx, window.duration_s, timeline_state)
        current_time_s = window.end_s
        end_report, end_event = _record_phase_boundary(
            ctx,
            writer,
            phase=window.phase,
            boundary="end",
            phase_state=contract.ends,
            camera_spec=camera_spec,
            width=width,
            height=height,
            base_tolerance=base_tolerance,
        )
        phase_events.append(end_event)

        phase_ok = start_report.ok and end_report.ok
        all_ok = all_ok and phase_ok
        typer.echo(f"{window.phase.value:<22} {'OK' if phase_ok else 'FAIL'}")
        for report in (start_report, end_report):
            for failure in report.failures:
                typer.echo(f"  - {failure.message}")

    _write_optional_rerun_phase_log(rerun_rrd, scene_name=scene, phase_events=phase_events)
    writer.write_summary(
        {
            "scene": scene,
            "command": "contracts",
            "ok": all_ok,
            "phase_events": phase_events,
        }
    )
    writer.write_event("run_end", ok=all_ok)
    typer.echo(f"artifacts → {run_dir}")
    if not all_ok and fail_on_error:
        raise typer.Exit(code=1)


@app.command("phase")
def replay_phase(
    phase: Annotated[str, typer.Argument(help="phase name, e.g. remove_old_server")],
    out_root: Annotated[
        Path,
        typer.Option(help="root directory for structured phase replay artifacts"),
    ] = Path("results/runs"),
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
    rerun_rrd: Annotated[
        Path | None,
        typer.Option(
            "--rerun-rrd",
            help="optional .rrd file with this phase's boundary TextLog events",
        ),
    ] = None,
    base_tolerance: Annotated[
        float,
        typer.Option(help="absolute tolerance for base qpos contract checks"),
    ] = 1e-3,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error/--no-fail-on-error", help="exit non-zero on contract failure"
        ),
    ] = True,
    camera: Annotated[
        str | None,
        typer.Option(help="named scene camera for boundary renders"),
    ] = None,
    width: WidthOpt = 960,
    height: HeightOpt = 720,
    az: AzOpt = 40.0,
    el: ElOpt = -15.0,
    dist: DistOpt = 2.2,
    lookat: LookatOpt = "0.30,0.0,0.9",
) -> None:
    """Replay one phase, saving before/after snapshots and renders."""
    task_phase = _parse_task_phase(phase)
    ctx = build_scene_and_advance(scene, Seconds(0.0))
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {scene!r} has no make_task_plan")

    phase_contracts: tuple[PhaseContract, ...] = tuple(
        getattr(ctx.scene_module, "PHASE_CONTRACTS", ())
    )
    contracts_by_phase = _phase_contracts_by_phase(phase_contracts)
    contract = contracts_by_phase.get(task_phase)
    if contract is None:
        raise typer.BadParameter(f"scene {scene!r} has no contract for phase {task_phase.value!r}")

    windows = _phase_windows(ctx.task_plan)
    window = windows.get(task_phase)
    if window is None:
        raise typer.BadParameter(f"task plan has no steps for phase {task_phase.value!r}")

    run_dir = make_run_directory(out_root, scene_name=f"{scene}_{task_phase.value}")
    writer = RunArtifactWriter(run_dir, scene_name=scene)
    writer.write_json("phase_contract.json", phase_contract_to_json_dict(contract))
    writer.write_event(
        "run_start",
        command="phase",
        phase=task_phase.value,
        start_s=window.start_s,
        end_s=window.end_s,
    )

    camera_spec = _resolve_camera(camera, az, el, dist, lookat)
    timeline_state = make_timeline_state(ctx.data, ctx.arms)
    timeline_state = _advance_context(ctx, window.start_s, timeline_state)
    start_report, start_event = _record_phase_boundary(
        ctx,
        writer,
        phase=task_phase,
        boundary="start",
        phase_state=contract.starts,
        camera_spec=camera_spec,
        width=width,
        height=height,
        base_tolerance=base_tolerance,
    )

    timeline_state = _advance_context(ctx, window.duration_s, timeline_state)
    end_report, end_event = _record_phase_boundary(
        ctx,
        writer,
        phase=task_phase,
        boundary="end",
        phase_state=contract.ends,
        camera_spec=camera_spec,
        width=width,
        height=height,
        base_tolerance=base_tolerance,
    )

    phase_events = [start_event, end_event]
    _write_optional_rerun_phase_log(rerun_rrd, scene_name=scene, phase_events=phase_events)
    all_ok = start_report.ok and end_report.ok
    writer.write_summary(
        {
            "scene": scene,
            "command": "phase",
            "phase": task_phase.value,
            "start_s": window.start_s,
            "end_s": window.end_s,
            "ok": all_ok,
            "phase_events": phase_events,
        }
    )
    writer.write_event("run_end", ok=all_ok)
    typer.echo(
        f"{task_phase.value}: {'OK' if all_ok else 'FAIL'} "
        f"({window.start_s:.2f}s → {window.end_s:.2f}s)"
    )
    for report in (start_report, end_report):
        for failure in report.failures:
            typer.echo(f"  - {failure.message}")
    typer.echo(f"artifacts → {run_dir}")
    if not all_ok and fail_on_error:
        raise typer.Exit(code=1)


# ---- phase-graph -------------------------------------------------------------


@app.command("phase-graph")
def phase_graph(
    out: Annotated[
        Path | None,
        typer.Option(help="output .dot file (omit for stdout)"),
    ] = None,
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
) -> None:
    """Emit a GraphViz DOT graph of the scene's phase contracts.

    Renders the transition graph declared by `legal_predecessors` on each
    `PhaseContract`. Initial phases (no predecessors) are highlighted in blue,
    making the entry point obvious. Render with: `dot -Tpng phases.dot -o phases.png`.
    """
    scene_module = load_scene(scene)
    contracts: tuple[PhaseContract, ...] = tuple(getattr(scene_module, "PHASE_CONTRACTS", ()))
    if not contracts:
        raise typer.BadParameter(f"scene {scene!r} has no PHASE_CONTRACTS")
    lines = [
        "digraph phases {",
        '  rankdir="LR";',
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    for contract in contracts:
        label = contract.phase.name
        attrs = [f'label="{label}"']
        if not contract.legal_predecessors:
            attrs.append('style="rounded,filled"')
            attrs.append('fillcolor="#dbeafe"')
        lines.append(f"  {contract.phase.name} [{', '.join(attrs)}];")
    for contract in contracts:
        for prev in contract.legal_predecessors:
            lines.append(f"  {prev.name} -> {contract.phase.name};")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    if out is None:
        typer.echo(text, nl=False)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}")


# ---- diff --------------------------------------------------------------------


@app.command()
def diff(
    a: Annotated[Path, typer.Option(help="first PNG")],
    b: Annotated[Path, typer.Option(help="second PNG")],
    out: Annotated[Path, typer.Option(help="diff heat-map output PNG")],
    threshold: Annotated[
        int,
        typer.Option(help="per-channel diff threshold for 'changed' stat (0-255)"),
    ] = 5,
) -> None:
    """Compute a pixel-difference heat-map between two PNG renders."""
    img_a = iio.imread(a)
    img_b = iio.imread(b)
    if img_a.shape != img_b.shape:
        typer.echo(
            f"image shapes differ: {img_a.shape} vs {img_b.shape} "
            "— re-render both at the same width/height before diffing",
            err=True,
        )
        raise typer.Exit(code=1)

    abs_diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).astype(np.uint8)
    heatmap = np.minimum(abs_diff.astype(np.int16) * 8, 255).astype(np.uint8)
    iio.imwrite(out, heatmap)
    max_diff = int(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    changed = (abs_diff > threshold).any(axis=-1)
    typer.echo(f"max per-channel diff: {max_diff}")
    typer.echo(f"mean per-channel diff: {mean_diff:.2f}")
    typer.echo(f"pixels with any-channel diff > {threshold}: {float(changed.mean()) * 100:.1f}%")
    typer.echo(f"heat-map → {out}")


# ---- ik (feasibility sweep) --------------------------------------------------


_IK_TOL_M = 0.02


def _build_ik_seeds(home: np.ndarray) -> list[np.ndarray]:
    """Five-seed spread around the scene's home joint config.

    The home pose is one seed; the other four are small joint-wise
    perturbations (±0.5, ±0.3 rad on a couple of joints) that exercise
    different convergence basins. Built from `home` so the seed shape
    + magnitudes are robot-agnostic — works for Piper's 6-joint arm
    chain just as well as UR10e's.
    """
    home = np.asarray(home, dtype=float)
    perturbations = [
        np.zeros_like(home),
        np.array([+0.5, +0.3, -0.3, 0.0, +0.1, 0.0])[: len(home)],
        np.array([-0.5, -0.3, +0.3, 0.0, -0.1, 0.0])[: len(home)],
        np.array([0.0, +0.5, -0.5, +0.5, +0.3, 0.0])[: len(home)],
        np.array([0.0, -0.5, +0.5, -0.5, -0.3, 0.0])[: len(home)],
    ]
    return [home + p for p in perturbations]


@app.command()
def ik(
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
) -> None:
    """Probe IK feasibility across a task plan's waypoints.

    For each step's arm_q the tool computes the implied TCP target,
    then re-runs IK from a spread of joint seeds. Flags waypoints that
    only converge from the luckily-chosen runtime seed — those are
    fragile to later layout changes.
    """
    from ik import PositionOnly, solve_ik

    ctx = build_scene_and_advance(scene, Seconds(0.0))
    scene_mod = ctx.scene_module
    cube_body_ids = [
        mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in getattr(scene_mod, "GRIPPABLES", ())
    ]
    task_plan = scene_mod.make_task_plan(ctx.model, ctx.data, ctx.arms, cube_body_ids)

    # Scene exposes which joints to lock during IK + a home-pose array
    # to seed the solver with. Both vary by robot kind so they belong
    # to the scene module rather than this tool.
    locked_joint_names: tuple[str, ...] = scene_mod.IK_LOCKED_JOINT_NAMES
    ik_seeds = _build_ik_seeds(np.asarray(scene_mod.IK_SEED_Q, dtype=float))

    typer.echo("status   side    conv  err(mm)  tcp(world)          label")
    typer.echo("-" * 80)
    failed = fragile = robust = 0
    for side, script in task_plan.items():
        arm = ctx.arms[side]
        for step in script:
            for i, idx in enumerate(arm.arm_qpos_idx):
                ctx.data.qpos[idx] = step.arm_q[i]
            mujoco.mj_forward(ctx.model, ctx.data)
            tcp = np.asarray(ctx.data.site_xpos[arm.tcp_site_id], dtype=float).copy()
            best_err = np.inf
            n_conv = 0
            unreachable = False
            for seed in ik_seeds:
                try:
                    _, err = solve_ik(
                        ctx.model,
                        ctx.data,
                        arm,
                        tcp,
                        orientation=PositionOnly(),
                        seed_q=seed,
                        locked_joint_names=locked_joint_names,
                        solver="daqp",
                    )
                except RuntimeError:
                    unreachable = True
                    continue
                best_err = min(best_err, err)
                if err <= _IK_TOL_M:
                    n_conv += 1
            if best_err > _IK_TOL_M:
                status = "FAIL"
                failed += 1
            elif unreachable or n_conv < len(ik_seeds):
                status = "FRAGILE"
                fragile += 1
            else:
                status = "OK"
                robust += 1
            typer.echo(
                f"{status:<8} {side:<7} {n_conv:>2}/{len(ik_seeds)} "
                f"{best_err * 1000:>7.2f} ({tcp[0]:+.2f},{tcp[1]:+.2f},{tcp[2]:+.2f}) "
                f"{step.label}"
            )
    typer.echo("-" * 80)
    typer.echo(
        f"summary: {failed + fragile + robust} waypoints | "
        f"{failed} fail | {fragile} fragile | {robust} robust"
    )


# ---- review (regression-catching keyframe grid + video) ----------------------


_REVIEW_KEYFRAMES_DATA_CENTER: tuple[tuple[float, str], ...] = (
    (0.0, "home"),
    (5.0, "settled at cables"),
    (10.5, "unplugged cable 1"),
    (16.0, "unplugged cable 2"),
    (22.0, "all cables unplugged"),
    (27.0, "server slid out (base reversed)"),
    (30.5, "rotating toward cart"),
    (34.0, "server on cart"),
    (39.5, "gripped new server"),
    (47.5, "new server in rack"),
)
"""Timestamps + labels of the moments a human reviewer cares about
when judging the data-center demo. Pinned to the scene's task plan;
update if step durations shift materially."""


_REVIEW_ANGLES: tuple[tuple[str, FreeCameraPose], ...] = (
    (
        "3q",
        FreeCameraPose(
            azimuth_deg=AzimuthDeg(40.0),
            elevation_deg=ElevationDeg(-15.0),
            distance_m=Metres(2.2),
            lookat=(0.30, 0.0, 0.9),
        ),
    ),
    (
        "side",
        FreeCameraPose(
            # Side-on profile: camera ~1 m to the +y side of the
            # rack-and-robot midline, looking across at the rack-arm
            # interaction. Avoids the wrist-cam mesh dominating the
            # frame (the lookat at the cable-port height keeps the
            # arms and rack interior in profile, not in front of the
            # camera).
            azimuth_deg=AzimuthDeg(90.0),
            elevation_deg=ElevationDeg(-5.0),
            distance_m=Metres(1.5),
            lookat=(0.30, 0.0, 0.85),
        ),
    ),
)
"""Two canonical free-cam poses: a wide 3/4 overview and a rack-
interior view looking back at the robot. Enough to catch "arm is too
high", "server didn't seat", or "shelf protruding" without
multi-window orbiting."""


@app.command()
def review(
    out_dir: Annotated[
        Path, typer.Option(help="directory to write review.png + review.mp4 into")
    ] = Path("/tmp"),
    scene: SceneOpt = "mobile_aloha_ur10e_server_swap",
    video_fps: Annotated[
        float, typer.Option(help="video playback fps (frames sampled every 0.5 s sim-time)")
    ] = 10.0,
    video_end: Annotated[
        float, typer.Option(help="video covers 0..video_end seconds of sim time")
    ] = 45.0,
) -> None:
    """One-shot regression render: grid of keyframes from 2 angles +
    a short mp4 of the whole task.

    Run after every substantive scene change to catch visual issues —
    arm pose wrong, compartment clipping, shelf protruding, etc. —
    before they surface when the user opens the viewer.

    Outputs:
      {out_dir}/review.png — (rows = keyframes) x (cols = angles) grid
      {out_dir}/review.mp4 — timelapse at `video_fps` from 0..video_end
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = 560, 420

    # --- keyframe grid ---
    images: list[np.ndarray] = []
    labels: list[str] = []
    ctx = build_scene_and_advance(scene, Seconds(0.0))
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {scene!r} has no make_task_plan")
    aux_name_to_id = {
        name: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(ctx.scene_module, "AUX_ACTUATOR_NAMES", ())
    }
    sim_dt = float(ctx.model.opt.timestep)
    prev_t = 0.0
    timeline_state = make_timeline_state(ctx.data, ctx.arms)

    for t, label in _REVIEW_KEYFRAMES_DATA_CENTER:
        dt = t - prev_t
        if dt > 0:
            advance_timeline_with_state(
                ctx.model,
                ctx.data,
                ctx.arms,
                ctx.task_plan,
                timeline_state,
                aux_name_to_id,
                sim_dt,
                Seconds(dt),
            )
            mujoco.mj_forward(ctx.model, ctx.data)
        prev_t = t
        for angle_name, pose in _REVIEW_ANGLES:
            img = render_frame(
                ctx.model,
                ctx.data,
                camera=build_free_cam(pose),
                width=width,
                height=height,
            )
            images.append(img)
            labels.append(f"t={t:05.1f} {angle_name} — {label}")

    # Fixed 2-column layout (one per angle) so the review reads
    # top-to-bottom chronologically — visually easier than the
    # square-ish auto-layout `_tile_grid` defaults to.
    grid_img = _tile_grid_cols(images, labels, cols=len(_REVIEW_ANGLES))
    grid_path = out_dir / "review.png"
    iio.imwrite(grid_path, grid_img)
    typer.echo(f"grid → {grid_path} ({len(images)} tiles)")

    # --- video timelapse ---
    ctx = build_scene_and_advance(scene, Seconds(0.0))  # reset sim for clean advance
    if ctx.task_plan is None:
        raise typer.BadParameter(f"scene {scene!r} has no make_task_plan")
    aux_name_to_id = {
        name: mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in getattr(ctx.scene_module, "AUX_ACTUATOR_NAMES", ())
    }
    sim_dt = float(ctx.model.opt.timestep)
    video_task_plan = ctx.task_plan
    prev_t = 0.0
    timeline_state = make_timeline_state(ctx.data, ctx.arms)
    frames: list[np.ndarray] = []
    step_s = 1.0 / video_fps
    for target_t in np.arange(0.0, video_end + 1e-9, step_s):
        dt = float(target_t) - prev_t
        if dt > 0:
            advance_timeline_with_state(
                ctx.model,
                ctx.data,
                ctx.arms,
                video_task_plan,
                timeline_state,
                aux_name_to_id,
                sim_dt,
                Seconds(dt),
            )
            mujoco.mj_forward(ctx.model, ctx.data)
        # Use the wide 3/4 angle for video — easiest for a human
        # reviewer to track the arms against a fixed reference frame.
        img = render_frame(
            ctx.model,
            ctx.data,
            camera=build_free_cam(_REVIEW_ANGLES[0][1]),
            width=640,
            height=480,
        )
        frames.append(img)
        prev_t = float(target_t)

    video_path = out_dir / "review.mp4"
    iio.imwrite(video_path, frames, fps=int(video_fps), codec="libx264")
    typer.echo(f"video → {video_path} ({len(frames)} frames @ {video_fps} fps)")


if __name__ == "__main__":
    app()
