# bimanual_sim

MuJoCo + Viser sim experiment. One scene today:

- `data_center` — bimanual AgileX Pipers on a PAL TIAGo mobile base performing
  a rack server swap: unplug 3 color-coded cables, pull the old server out,
  stow in an onboard compartment, slot a replacement in from the upper
  compartment, replug the cables.

Scene modules live under `scenes/`. The runner in `runner.py` is scene-agnostic
so additional scenes slot in without touching the shared infra.

## Layout

```
runner.py                    generic main loop + Viser GUI (CLI: --scene NAME)
scene_base.py                Step dataclass + shape aliases + CubeID helper
scene_check.py               compile-time sanity checks + schematic printer
                             (AttachmentConstraint, check_scene, print_schematic)
ik.py                        mink-backed differential IK
arm_handles.py               Piper-specific joint/actuator/body lookups + ArmSide enum
cameras.py                   Viser camera-frustum widgets + CameraRole enum
viser_render.py              MuJoCo geom → Viser mesh bridge
welds.py                     equality-weld grasp cheat + generic attachment welds
paths.py                     Menagerie path resolution
                             (PIPER_XML / TIAGO_XML / D435I_XML / D405_MESH_STL)
robots/
  tiago.py                   TIAGo loader: TiagoConfig + load_tiago() — strip
                             upstream single arm + head, drop base freejoint,
                             prune orphan excludes. Menagerie-shape assertions.
  piper.py                   Piper loader: PiperConfig + attach_piper() — attach
                             with prefix, override per-joint kp/kv/forcerange.
scenes/
  data_center.py             the scene (MJCF build, IK task plan, weld registry)
  data_center_layout.py      declarative geometry: every dimension/anchor as a
                             frozen dataclass with cross-component invariants
tools/
  mj.py                      unified debug CLI (typer): snapshot / video / grid /
                             plan / diff / ik — `uv run python tools/mj.py --help`
  _runtime.py                shared scene-build + timeline-advance helpers
serve.sh                     start/stop/status/logs helper
```

## Debug tools (`tools/mj.py`)

Headless renders + plan inspection for agent-driven debugging. One
typer CLI, six subcommands; every subcommand defaults to
`--scene data_center`. Renders go through MuJoCo's native `Renderer`
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

# Pixel-diff heat-map between two equal-size renders; prints max/mean/%changed
uv run python tools/mj.py diff --a /tmp/before.png --b /tmp/after.png --out /tmp/d.png

# IK feasibility sweep: replays each waypoint's arm_q, re-solves from 5 seeds,
# labels each step OK / FRAGILE (converges from only some seeds) / FAIL.
uv run python tools/mj.py ik
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
./serve.sh start            # defaults to scene=data_center
# open http://localhost:8080
```

Invoke `runner.py` directly for extra knobs:

```bash
uv run python runner.py --scene data_center [--speed 1.0] [--render-hz 60] [--max-rate]
```

- `--speed` — multiplier on Step durations (also live-adjustable from the GUI slider).
- `--render-hz` — cap on render/physics tick rate. 60 is plenty for the browser;
  higher just burns websocket CPU.
- `--max-rate` — drop the realtime throttle entirely. Useful for batch
  trajectory generation or headless video capture.

The Viser page exposes runtime controls: ▶ play / ⏸ pause, ↺ reset, speed
slider, per-arm status lines.

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
`scenes/data_center.py` for a worked example that uses all of them.

Start the new scene with: `./serve.sh start my_scene`.
