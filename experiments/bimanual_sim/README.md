# bimanual_sim

MuJoCo + Viser sim experiment. Scenes follow `<base>_<arms>_<task>` naming so
the shared infra can host more than one demo without ambiguity:

- `mobile_aloha_ur10e_server_swap` — bimanual UR10e arms with Robotiq 2F-85
  grippers on a Mobile-ALOHA-style base performing a rack server swap: pull
  the old server out, stow it on a service cart, retrieve a replacement
  server, install it in the rack. Live scene.
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
paths.py                     Menagerie path resolution
                             (PIPER_XML / TIAGO_XML / D435I_XML / D405_MESH_STL)
robots/
  mobile_aloha.py            Mobile ALOHA base loader with planar x/y/yaw joints
  ur10e.py                   UR10e + 2F-85 loader with namespaced wrist cameras
scenes/
  mobile_aloha_ur10e_server_swap.py        live scene (MJCF build, IK plan, welds)
  mobile_aloha_ur10e_server_swap_layout.py declarative geometry as frozen
                                           dataclasses with cross-component invariants
  tiago_piper_server_cable_swap{,_layout}.py  legacy backup
tools/
  mj.py                      unified debug CLI (typer): snapshot / video / grid /
                             plan / contracts / phase / diff / ik / review
                             — `uv run python tools/mj.py --help`
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

## Running locally

```bash
cd experiments/bimanual_sim
uv sync                     # first time only
./serve.sh start            # defaults to scene=mobile_aloha_ur10e_server_swap
# open http://localhost:8080
```

Invoke `runner.py` directly for extra knobs:

```bash
uv run python runner.py --scene mobile_aloha_ur10e_server_swap [--speed 1.0] [--render-hz 60] [--max-rate]
```

- `--speed` — multiplier on Step durations (also live-adjustable from the GUI slider).
- `--render-hz` — cap on render/physics tick rate. 60 is plenty for the browser;
  higher just burns websocket CPU.
- `--max-rate` — drop the realtime throttle entirely. Useful for batch
  trajectory generation or headless video capture.

The Viser page exposes runtime controls: ▶ play / ⏸ pause, ↺ reset, speed
slider, per-arm status lines.

## Phase Debug Workflow

The data-center scene defines named `TaskPhase` values and `PHASE_CONTRACTS`.
Each contract declares what must be true at the start and end of the phase:
which attachment equalities are active or inactive, and where the planar base
must be. This gives every demo failure a narrower question: did setup,
disconnect, remove, stow, retrieve, install, reconnect, or reset violate its
boundary?

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

## Scene contract (if adding another)

A scene module under `scenes/` must export:

```python
from arm_handles import ArmSide
from cameras import CameraRole
from scene_base import Step

NAME = "my_scene"
ARM_PREFIXES: tuple[ArmSide, ...] = (ArmSide.LEFT, ArmSide.RIGHT)  # or () for no arm
N_CUBES = 0                          # number of grippable objects

# Optional: camera frustum widgets drawn in Viser.
CAMERAS: tuple[tuple[str, CameraRole], ...] = ()

# Optional: scene-owned (non-arm) actuators addressable via Step.aux_ctrl.
AUX_ACTUATOR_NAMES: tuple[str, ...] = ()

def build_spec() -> mujoco.MjSpec: ...
def apply_initial_state(model, data, arms, cube_body_ids) -> None: ...

# One of:
def make_task_plan(model, data, arms, cube_body_ids) -> dict[ArmSide, list[Step]]: ...
def step_free_play(t, model, data) -> None: ...
```

`Step` carries `weld_activate`/`weld_deactivate` (grasp cheats indexed by
cube id), `attach_activate`/`attach_deactivate` (body↔body welds addressed by
MJCF name), and `aux_ctrl` (dict of aux actuator name → target). See
`scenes/mobile_aloha_ur10e_server_swap.py` for a worked example that uses all of them.

Start the new scene with: `./serve.sh start my_scene`.
