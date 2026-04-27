# bimanual_sim

MuJoCo + Viser sim experiment. Scenes follow `<base>_<arms>_<task>` naming so
the shared infra can host more than one demo without ambiguity:

- `mobile_aloha_ur10e_server_swap` — bimanual UR10e arms with Robotiq 2F-85
  grippers on a Mobile-ALOHA-style base performing a rack server swap: pull
  the old server out, stow it on a service cart, retrieve a replacement
  server, install it in the rack. Live scene.
- `mobile_aloha_piper_indicator_check` — Mobile-ALOHA chassis with bimanual
  Piper arms inspecting a server in a 5×2 rack aisle. Drive in, yaw to face
  the alert rack, both arms reach to "examine" the alert server, indicator
  flips red→green via runtime RGBA update, retract. No manipulation, no
  welds — base motion + scripted bimanual gesture. Wrist + top + free-cam
  views set up for offline video rendering (see "Multi-view video rendering"
  below).
- `tiago_piper_server_cable_swap` — legacy variant on a TIAGo base with
  Piper arms and 3 cable disconnect/reconnect phases bookending the server
  swap. Kept as a backup; not actively maintained.

Scene modules live under `scenes/`. The runner in `runner.py` is scene-agnostic
so additional scenes slot in without touching the shared infra.

## Layout

```
runner.py                    generic main loop + Viser GUI (CLI: --scene NAME)
scene_base.py                Step dataclass + shape aliases + CubeID helper
scene_check.py               compile-time sanity checks + schematic printer
                             (AttachmentConstraint, check_scene, print_schematic)
ik.py                        mink-backed differential IK
arm_handles.py               robot-specific joint/actuator/body lookups + ArmSide enum
cameras.py                   Viser camera-frustum widgets + CameraRole enum
viser_render.py              MuJoCo geom → Viser mesh bridge
welds.py                     equality-weld grasp cheat + generic attachment welds
phase_monitor.py             live phase-contract monitor for runner.py
rerun_stream.py              live/offline Rerun recording helpers
teleop.py                    Viser TCP-drag authoring + JSON replay support
paths.py                     Menagerie path resolution
                             (PIPER_XML / TIAGO_XML / D435I_XML / D405_MESH_STL)
robots/
  mobile_aloha.py            Mobile ALOHA base loader with planar x/y/yaw joints
  piper.py                   Piper arm loader
  tiago.py                   TIAGo base + torso loader
  ur10e.py                   UR10e + 2F-85 loader with namespaced wrist cameras
scenes/
  mobile_aloha_ur10e_server_swap.py        live scene (MJCF build, IK plan, welds)
  mobile_aloha_ur10e_server_swap_layout.py declarative geometry as frozen
                                           dataclasses with cross-component invariants
  mobile_aloha_piper_indicator_check{,_layout}.py  bimanual Piper indicator-check
  tiago_piper_server_cable_swap{,_layout}.py       legacy backup
tools/
  mj.py                      unified debug CLI (typer): snapshot / video / grid /
                             plan / contracts / phase / phase-graph / diff /
                             ik / review
                             — `uv run python tools/mj.py --help`
  render_pov_videos.py       multi-camera offline video renderer for the
                             indicator-check scene: emits forward / left wrist /
                             right wrist / directorial mp4s
  inspect_aloha_body.py      inspect Mobile ALOHA mesh components
  label_aloha_arms.py        render/label mesh component candidates
  strip_aloha_arms.py        remove arm geometry from the ALOHA body mesh
  _runtime.py                shared scene-build + timeline-advance helpers
  observability.py           run artifacts, JSONL events, phase snapshots,
                             executable phase-contract checks
serve.sh                     start/stop/status/logs helper
```

## Debug tools (`tools/mj.py`)

Headless renders + plan inspection for agent-driven debugging. One
typer CLI; every subcommand defaults to
`--scene mobile_aloha_ur10e_server_swap`. Renders go through MuJoCo's native `Renderer`
over EGL (forced before the `mujoco` import so no X display is
needed); on a GPU host the GL driver offloads rendering automatically
with no explicit switch. `_runtime.py` owns the "import scene →
compile spec → apply initial state → advance task plan to time t"
plumbing so each subcommand stays a thin wrapper.

```bash
# Single frame (free-cam orbit: --az/--el/--dist/--lookat)
uv run python tools/mj.py snapshot --az 45 --el -20 --out /tmp/home.png

# Mid-task frame from a named scene camera
uv run python tools/mj.py snapshot --t 22 --camera top_d435i_cam --out /tmp/cable.png

# Time-lapse: one PNG per 0.5 s from 0..30 s (requires --out-prefix + --t > 0)
uv run python tools/mj.py snapshot --t 30 --every 0.5 --out-prefix /tmp/run_

# Stitch a {prefix}*.png sequence into mp4 (libx264) or gif
uv run python tools/mj.py video --prefix /tmp/run_ --fps 20 --out /tmp/run.mp4

# All scene cameras + free-cam tiled into a labeled grid at one sim time
uv run python tools/mj.py grid --t 22 --out /tmp/grid.png

# Task plan as a timeline table: side, start, dur, gripper, label, attach±, weld±
uv run python tools/mj.py plan | less

# Assert every declared phase boundary and write structured artifacts:
# events.jsonl, summary.json, phase_contracts.json, snapshots/*.npz, renders/*.png
uv run python tools/mj.py contracts --out-root results/runs

# Replay one phase and save before/after state snapshots and renders
uv run python tools/mj.py phase remove_old_server --out-root results/runs

# Render the declared phase predecessor graph as GraphViz DOT
uv run python tools/mj.py phase-graph --out /tmp/phases.dot

# Optional: also emit a Rerun recording with phase-boundary TextLog events
uv run python tools/mj.py contracts --rerun-rrd results/runs/server_swap.rrd

# Pixel-diff heat-map between two equal-size renders; prints max/mean/%changed
uv run python tools/mj.py diff --a /tmp/before.png --b /tmp/after.png --out /tmp/d.png

# IK feasibility sweep: replays each waypoint's arm_q, re-solves from 5 seeds,
# labels each step OK / FRAGILE (converges from only some seeds) / FAIL.
uv run python tools/mj.py ik

# Partner-facing regression packet: review.png keyframes + review.mp4 timelapse
uv run python tools/mj.py review --out-dir results/review
```

Run `tools/mj.py --help` or `tools/mj.py <subcommand> --help` for the full
option list. Shared option aliases (`--scene`, `--t`, `--width`, `--height`,
free-cam knobs) live at the top of `mj.py` so they behave identically across
subcommands.

## Prerequisites

1. Clone MuJoCo Menagerie to `~/mujoco_menagerie` (paths can be overridden via
   the `MENAGERIE_PATH` env var — see `paths.py`):
   ```bash
   git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
   ```
2. `uv` installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Python 3.12. `uv sync` will create the project virtualenv from
   `pyproject.toml`.
4. Optional tools for documented workflows:
   - `ffmpeg` for video encoding and 2×2 video grids.
   - GraphViz `dot` for rendering `tools/mj.py phase-graph` output.
   - Rerun viewer for `.rrd` replay or live Rerun streaming.

## Running locally

```bash
cd experiments/bimanual_sim
uv sync                     # first time only
./serve.sh start            # defaults to scene=mobile_aloha_ur10e_server_swap
# open http://localhost:8080
```

Invoke `runner.py` directly for extra knobs:

```bash
uv run python runner.py --scene mobile_aloha_ur10e_server_swap [--speed 1.0] [--render-hz 45] [--max-rate]
```

- `--speed` — multiplier on Step durations (also live-adjustable from the GUI slider).
- `--render-hz` — cap on Viser + physics update rate. Default is 45 Hz; higher
  rates mostly increase websocket/message-buffer CPU.
- `--max-rate` — drop the realtime throttle entirely. Useful for batch
  trajectory generation or headless video capture.
- `--inspect` — compile the scene, run `scene_check`, print the body/geom
  schematic, and exit before starting Viser.
- `--strict` — fail immediately when a phase contract fails instead of collecting
  failures until the demo exits.
- `--rerun-port PORT` — serve a local Rerun gRPC stream. Tunnel it from EC2 with
  `ssh -L PORT:localhost:PORT`, then connect the local viewer to
  `rerun+http://localhost:PORT`.
- `--rerun-connect URL` — push to an already-running Rerun viewer.
- `--rerun-rrd PATH` — write an offline Rerun `.rrd` recording.
- `--rerun-camera-every N` — if `N > 0`, include named-camera frames every N
  render ticks. Keep this at `0` on slow tunnels.
- `--teleop` — replace the scripted task plan with Viser drag handles for live
  TCP IK authoring.
- `--play-recording PATH` — replay a teleop JSON recording as the task plan.
- `--start-phase PHASE` — boot from a hand-authored phase start pose from the
  scene layout's `PHASE_HOMES` map, e.g. `remove_old_server`.

The Viser page exposes runtime controls: ▶ play / ⏸ pause, ↺ reset, speed
slider, 📷 focus on robot (re-anchors the orbit pivot to the chassis),
📸 log cam pose (prints the clicking client's current orbit-cam pose +
sim time to stdout / `runner.log` — used to author directorial keyframes,
see "Multi-view video rendering" below), per-arm status lines.

## Teleop authoring

Teleop is for shaping task motion directly in Viser when scripted IK produces
technically valid but awkward trajectories:

```bash
uv run python runner.py --scene mobile_aloha_ur10e_server_swap --teleop
```

The browser exposes TCP drag handles, per-joint sliders, base x/y/yaw controls,
phase selection, weld/grasp controls, and capture/save buttons. Captured
keyframes are saved under `/tmp/teleop_recordings/<scene>/teleop_<UTC>.json`
and can be replayed with:

```bash
uv run python runner.py \
  --scene mobile_aloha_ur10e_server_swap \
  --play-recording /tmp/teleop_recordings/mobile_aloha_ur10e_server_swap/teleop_YYYYMMDDTHHMMSSZ.json
```

Use `--start-phase PHASE` with either `--teleop` or `--play-recording` when you
want to author or replay from a specific phase home instead of scene start.

## Multi-view video rendering

`tools/render_pov_videos.py` is the offline renderer for the
indicator-check scene. One command produces four mp4s at 1920×1080 /
30 fps with 1-second keyframes (so seeking + playback stay smooth):

- `forward.mp4` — chassis-mounted top D435i (yaws with the base)
- `left_wrist.mp4` — D405 on the left Piper's wrist
- `right_wrist.mp4` — D405 on the right Piper's wrist
- `directorial.mp4` — free-cam cinematic with hard cuts between
  user-captured viewpoints

```bash
# On the host running MuJoCo, with the live viser runner stopped (the
# runner holds the EGL context and the offline renderer can't share
# it — `./serve.sh stop` first).
./serve.sh stop
uv run python tools/render_pov_videos.py --out-dir /tmp/pov

# Knobs (defaults shown):
#   --width 1920 --height 1080 --fps 30 --crf 18 --preset slower
#   --duration-s 0   (0 → full task plan length, ~27 s)
```

The renderer is headless via `MUJOCO_GL=egl` (set internally — no
`DISPLAY` needed). Each camera is rendered in its own pass over the
timeline, so peak memory is one camera's worth of raw frames. Things it
takes care of automatically:

- **Indicator flip** — `Step.set_geom_rgba` (the red→green flip during
  WAIT_AT_SERVER) is applied through `tools/_runtime.advance_timeline_with_state`,
  which mirrors the live runner's logic.
- **RGBA reset between passes** — the initial `model.geom_rgba` is
  snapshotted once and restored before each cam pass, so passes 2–4
  see the alert as red at t=0 (without this, only forward.mp4 would
  show the flip).
- **Hide debug overlays** — TCP site spheres, MJ camera frustum widgets,
  contact-point markers, and the menagerie Piper's translucent `class="collision"`
  finger pads are all suppressed in the rendered output. The live viser
  view is unaffected (it ignores `MjvOption.geomgroup`).
- **Timeline timing** — frame N is rendered at exactly `N/fps` of sim
  time (integer step accounting), so the last frame really does land on
  `task_duration_s`. A naïve `dt = target_t - prev_t` loop drops ~1 ms
  per frame to `int()` truncation and the retract phase ends up
  half-finished.

To compose all four into a single 1080p 2×2 grid (lighter to play than
4K, with 1-second keyframes for jump-resilient seeking):

```bash
ffmpeg -y \
  -i /tmp/pov/directorial.mp4 -i /tmp/pov/forward.mp4 \
  -i /tmp/pov/left_wrist.mp4 -i /tmp/pov/right_wrist.mp4 \
  -filter_complex "
    [0:v]scale=960:540,drawtext=text='directorial':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6[v0];
    [1:v]scale=960:540,drawtext=text='forward (top)':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6[v1];
    [2:v]scale=960:540,drawtext=text='left wrist':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6[v2];
    [3:v]scale=960:540,drawtext=text='right wrist':x=15:y=15:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6[v3];
    [v0][v1]hstack[top];[v2][v3]hstack[bot];[top][bot]vstack" \
  -c:v libx264 -preset slower -crf 18 -pix_fmt yuv420p -g 30 -keyint_min 30 \
  /tmp/pov/grid.mp4
```

### Tuning the camera framings

All camera positions, tilts, FOVs, mesh offsets, and stand height for
the indicator-check scene live as named `_TOP_CAM_*` / `_WRIST_CAM_*` /
`_WRIST_MESH_*` constants near the top of
`scenes/mobile_aloha_piper_indicator_check.py`. Change one number,
re-run the renderer — no other edits needed. Each constant has a
docstring explaining what it represents (not what its current value
*is*), so the prose doesn't drift when the value changes.

The defaults are tuned against the click-pose framing (sim t≈22, both
grippers touching the alert) and roughly match a real ALOHA-style rig:
top D435i at 42° vfov tilted 25° down, wrist D405s at 58° vfov tilted
20° toward the gripper.

### Authoring the directorial cut from viser

Hard-cut keyframes for the directorial track live in
`tools/render_pov_videos.py` as `_DIRECTORIAL_CUTS`:

```python
_DIRECTORIAL_CUTS = (
    (0.0,  (-1.312, -1.617, 2.945), (2.322, -0.839, 0.614)),
    (11.5, (4.174, -0.837, 2.294), (5.258, -0.069, 0.853)),
)
# tuple format: (start_t, position, lookat) — all in world-frame coords.
```

To add or change a cut, capture its pose interactively:

1. Start the runner (`./serve.sh start mobile_aloha_piper_indicator_check`)
   and open the viser page through the SSH tunnel.
2. Pause at the sim-time you want the cut to fire (▶ / ⏸).
3. Orbit / zoom the viser camera to the framing you want.
4. Click **📸 log cam pose**. A line like the following prints to
   stdout / `runner.log`:
   ```
   cam_pose t= 11.50  pos=(4.174, -0.837, 2.294)  lookat=(5.258, -0.069, 0.853)
   ```
5. Paste the `pos` + `lookat` (and the `start_t` you intend) into
   `_DIRECTORIAL_CUTS`. Stop the runner and re-run `tools/render_pov_videos.py`.

The renderer cuts (no interpolation) — at any sim time, the active
keyframe is the one whose `start_t` is the latest that's still
`<= target_t`. Two keyframes already give you the full demo (one for
the drive-in, one held through the rest).

Coordinate conventions match what viser exposes: world-frame `(x, y, z)`
in metres, `+z` up, no quaternion or distance/azimuth needed — the
renderer converts to MuJoCo's az/el/dist internally.

## Phase Debug Workflow

Scripted scenes can define named `TaskPhase` values and `PHASE_CONTRACTS`.
Each contract declares what must be true at the start and end of the phase:
which attachment equalities are active or inactive, and where the planar base
must be. Some contracts also assert expected grippable poses and in-phase
invariants such as held weld state, static joints, gripper state, and MuJoCo
QACC warning count. This gives every demo failure a narrower question: which
phase boundary or invariant changed unexpectedly?

Use `contracts` after changing scene geometry, IK targets, attachment names, or
step timing. Use `phase PHASE_NAME` when one stage looks wrong in Viser and you
want exact before/after MuJoCo snapshots for that stage.

Artifacts are intentionally plain files:

- `events.jsonl` is append-only structured logging for each boundary.
- `summary.json` is the CI-friendly pass/fail summary.
- `phase_contracts.json` records the expected phase states for that run.
- `snapshots/*.npz` stores `qpos`, `qvel`, `ctrl`, `eq_active`, `eq_data`, and
  sim time for exact reproduction.
- `renders/*.png` stores a visual before/after for each checked boundary.

## Running on a remote GPU box

`serve.sh` is location-agnostic — it `cd`s to its own directory via
`$BASH_SOURCE`, so you can drop this tree anywhere on the remote.

**Deploy** (adjust key + host):

```bash
rsync -av --delete \
  --exclude='.venv' --exclude='__pycache__' \
  --exclude='*.log' --exclude='*.pid' --exclude='.ruff_cache' \
  -e "ssh -i ~/.ssh/YOUR_KEY.pem" \
  experiments/bimanual_sim/ \
  user@your.host:/path/on/remote/
```

**Start / stop / inspect on remote**:

```bash
ssh -i ~/.ssh/YOUR_KEY.pem user@your.host /path/on/remote/serve.sh start
ssh -i ~/.ssh/YOUR_KEY.pem user@your.host /path/on/remote/serve.sh status
ssh -i ~/.ssh/YOUR_KEY.pem user@your.host /path/on/remote/serve.sh logs 80
ssh -i ~/.ssh/YOUR_KEY.pem user@your.host /path/on/remote/serve.sh stop
```

**View in a browser from your laptop.** Tunnel the Viser port:

```bash
ssh -i ~/.ssh/YOUR_KEY.pem -L 8080:localhost:8080 user@your.host
# leave that session open, then:
open http://localhost:8080
```

The runner binds to `127.0.0.1` by default — 8080 is never exposed publicly,
the SSH tunnel is the only way to reach it.

For your Malaysia EC2 instance, the same pattern is:

```bash
ssh -i ~/.ssh/openpi-seoul.pem -L 8080:localhost:8080 ubuntu@43.217.252.75
open http://localhost:8080
```

Run phase checks on the EC2 host and keep artifacts in the repo-local
`results/` directory:

```bash
uv run python tools/mj.py contracts --out-root results/runs
uv run python tools/mj.py review --out-dir results/review
```

To inspect artifacts locally without manual sync, either tunnel a simple file
server from the EC2 box:

```bash
# on EC2, from this directory
uv run python -m http.server 8090 --directory results

# on your Mac
ssh -i ~/.ssh/openpi-seoul.pem -L 8090:localhost:8090 ubuntu@43.217.252.75
open http://localhost:8090
```

Or write a Rerun `.rrd` on EC2 and open it through the same tunneled file
server. Rerun can also stream to a viewer in another process; the CLI keeps Rerun
optional so the base demo still runs with only MuJoCo, Viser, and ImageIO.

## Code quality

Run the Python checks before committing behavior changes:

```bash
uv run ruff check --fix
uv run ruff format
uv run ty check
uv run pytest
```

For render/scene changes, also run the relevant executable scene checks:

```bash
uv run python runner.py --scene mobile_aloha_ur10e_server_swap --inspect
uv run python tools/mj.py contracts --out-root results/runs
```

## Scene contract (if adding another)

A scene module under `scenes/` must export:

```python
import mujoco

from arm_handles import ArmSide
from arm_handles import ArmHandles
from cameras import CameraRole
from scene_base import Step

NAME = "my_scene"
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)  # or () for no arm
N_CUBES = 0                          # number of grippable objects

# Optional: grippable body names parallel to N_CUBES.
GRIPPABLES: tuple[str, ...] = ()

# Optional: camera frustum widgets drawn in Viser.
CAMERAS: tuple[tuple[str, CameraRole], ...] = ()

# Optional: scene-owned (non-arm) actuators addressable via Step.aux_ctrl.
AUX_ACTUATOR_NAMES: tuple[str, ...] = ()

def build_spec() -> tuple[mujoco.MjModel, mujoco.MjData]: ...
def apply_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
) -> None: ...

# One of:
def make_task_plan(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arms: dict[ArmSide, ArmHandles],
    cube_body_ids: list[int],
) -> dict[ArmSide, list[Step]]: ...
def step_free_play(t: float, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...
```

`Step` carries `weld_activate`/`weld_deactivate` (grasp cheats indexed by
cube id), `attach_activate`/`attach_deactivate` and `attach_activate_at`
(body↔body welds addressed by MJCF name), `aux_ctrl` (dict of aux actuator
name → target), `set_geom_rgba` (runtime visual state changes), and `phase`
(a `TaskPhase` label for contracts and tools). See
`scenes/mobile_aloha_ur10e_server_swap.py` for a worked example that uses
grasp welds, attachment welds, aux controls, and phase contracts.

Start the new scene with: `./serve.sh start my_scene`.
