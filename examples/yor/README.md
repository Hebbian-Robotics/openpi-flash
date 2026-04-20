# YOR -> OpenPI Client

Client for running OpenPI inference on a [YOR (Your Own Robot)](https://yourownrobot.ai/) bimanual mobile manipulator.

## Prerequisites

1. A running YOR robot (physical or [MuJoCo simulation](https://github.com/YOR-robot/YOR))
2. An OpenPI inference server accessible via QUIC

### Robot setup

Follow the [YOR setup instructions](https://github.com/YOR-robot/YOR) to get the robot driver running.  The driver exposes an RPC server on port 5557 via `commlink`.

For simulation:
```bash
# In the YOR repo
conda activate yor
python robot/yor_mujoco.py
```

For physical robot:
```bash
# In the YOR repo
conda activate yor
./create_windows.sh  # sets up CAN, drivers, tmux
```

### Client dependencies

The client requires `commlink` (the YOR RPC library):
```bash
pip install commlink
```

## Usage

```bash
# With MuJoCo simulation (robot on localhost)
uv run python examples/yor/yor_client.py \
    --server-host 10.0.0.42 \
    --prompt "pick up the red cup"

# With physical robot on a specific host
uv run python examples/yor/yor_client.py \
    --robot-host 10.0.0.10 \
    --server-host 10.0.0.42 \
    --prompt "fold the towel" \
    --mapping aloha

# Using DROID mapping instead of ALOHA
uv run python examples/yor/yor_client.py \
    --robot-host 10.0.0.10 \
    --server-host 10.0.0.42 \
    --prompt "pick up the red cup" \
    --mapping droid-dual-arm
```

## Mappings

YOR has 6-DOF per arm, which is a native fit for the ALOHA format:

| Mapping | State dims | Cameras | Notes |
|---------|-----------|---------|-------|
| `aloha` (default) | 14 | cam_high + dummy cam_low | Native 6-DOF fit, no joint dropping |
| `droid-dual-arm` | 14 | cam_high | DROID observation format |

## Camera configuration

By default, the client opens OpenCV device `0` as the head camera (`cam_high`).  To add wrist cameras or change device indices, edit `DEFAULT_CAMERA_DEVICES` in `yor_client.py`.

## Architecture

Unlike the Galaxea client (which uses ROS 2 topics), YOR communicates via `commlink` ZMQ RPC:

```
YOR Robot (yor.py / yor_mujoco.py)
    ↕ ZMQ RPC (port 5557)
YOR OpenPI Client (yor_client.py)
    ↕ QUIC (port 5555)
OpenPI Server
```

The client polls joint state and camera images, builds an observation, runs
inference via QUIC, and sends the resulting joint targets back to the robot.
