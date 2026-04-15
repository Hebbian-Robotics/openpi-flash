# Galaxea R1 Pro Integration

Connect a Galaxea R1 Pro robot to a hosted OpenPI inference server. The client runs on the R1 Pro's onboard Jetson AGX Orin and sends observations over QUIC for low-latency policy inference.

## Prerequisites

- A Galaxea R1 Pro with ROS 2 Humble workspace set up ([R1 Pro Software Guide](https://docs.galaxea-dynamics.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/))
- An OpenPI inference server running (see [`docs/aws-manual-setup.md`](../../docs/aws-manual-setup.md) or use Modal)
- Network connectivity between the robot and the server (UDP port 5555)

## Setup on the Jetson

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust toolchain (needed to build the QUIC sidecar)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the hosting repo and build the QUIC sidecar
git clone <your-hosting-repo-url> ~/openpi-hosting
cd ~/openpi-hosting/quic-sidecar
cargo build --release

# Source ROS 2 (must be done before creating the venv so ROS packages are visible)
source /opt/ros/humble/setup.bash
source ~/r1pro_ws/install/setup.bash

# Create a Python 3.11 venv that inherits system ROS packages
cd ~/openpi-hosting
uv venv --system-site-packages --python 3.11

# Install Python client dependencies into the venv
uv pip install openpi-client opencv-python-headless numpy
```

## Running the client

```bash
# Source ROS 2 (if not already sourced in this shell)
source /opt/ros/humble/setup.bash
source ~/r1pro_ws/install/setup.bash

# Make sure the R1 Pro hardware stack is running:
#   ros2 launch HDAS r1pro.py
#   ros2 launch signal_camera_node signal_camera.py
#   ros2 launch mobiman r1_pro_jointTrackerdemo.py

cd ~/openpi-hosting

# Run the client (DROID dual-arm mapping, recommended)
uv run python examples/galaxea_r1_pro/galaxea_r1_pro_client.py \
    --host <server-ip> \
    --prompt "pick up the red cup" \
    --mapping droid-dual-arm \
    --rate 15

# Or specify a custom port
uv run python examples/galaxea_r1_pro/galaxea_r1_pro_client.py \
    --host <server-ip> \
    --port 5555 \
    --prompt "pick up the red cup"
```

## Embodiment mappings

The client uses pluggable mappings to convert between R1 Pro sensor data and OpenPI's observation format. Two mappings are provided:

### `droid-dual-arm` (recommended)

Extends the DROID 7-DOF format to dual arms. Best match for the R1 Pro's 7 joints per arm.

| Property      | Value                                             |
| ------------- | ------------------------------------------------- |
| State dims    | 16 (7 joints + gripper per arm)                   |
| Cameras       | 3 (head, left wrist, right wrist)                 |
| Image format  | HWC uint8, 224x224                                |
| Server config | `pi05_droid` or `pi05_base` with DROID norm stats |

### `aloha`

Force-fits into ALOHA's 6-DOF dual-arm format by dropping one joint per arm. Provided for experimentation — the dropped joint must be held at a fixed position.

| Property      | Value                                            |
| ------------- | ------------------------------------------------ |
| State dims    | 14 (6 joints + gripper per arm)                  |
| Cameras       | 4 (head, front chassis, left wrist, right wrist) |
| Image format  | CHW uint8, 224x224                               |
| Server config | `pi05_aloha` or `pi0_aloha`                      |

## Adding a custom mapping

Create a new `EmbodimentMapping` subclass in `examples/r1_pro_embodiment_mappings.py`:

```python
@dataclass(frozen=True)
class MyCustomMapping(EmbodimentMapping):
    def camera_topic_mapping(self) -> dict[str, str]:
        return {
            "cam_head": "/hdas/camera_head/left_raw/image_raw_color/compressed",
            # ... your camera topics
        }

    def image_size(self) -> tuple[int, int]:
        return (224, 224)

    def build_observation(self, *, left_arm_positions, right_arm_positions,
                          left_gripper_position, right_gripper_position,
                          images, prompt) -> dict:
        # Map to your chosen OpenPI observation format
        ...

    def unpack_actions(self, server_response) -> R1ProActionCommand:
        # Extract joint commands from server response
        ...

# Register it
AVAILABLE_MAPPINGS["my-custom"] = MyCustomMapping()
```

Then run with `--mapping my-custom`.

## R1 Pro ROS 2 topics reference

### Sensor topics (subscribed by the client)

| Purpose              | Topic                                                   | Message Type                  |
| -------------------- | ------------------------------------------------------- | ----------------------------- |
| Left arm joints      | `/hdas/feedback_arm_left`                               | `sensor_msgs/JointState`      |
| Right arm joints     | `/hdas/feedback_arm_right`                              | `sensor_msgs/JointState`      |
| Left gripper         | `/hdas/feedback_gripper_left`                           | `sensor_msgs/JointState`      |
| Right gripper        | `/hdas/feedback_gripper_right`                          | `sensor_msgs/JointState`      |
| Head camera          | `/hdas/camera_head/left_raw/image_raw_color/compressed` | `sensor_msgs/CompressedImage` |
| Left wrist camera    | `/hdas/camera_wrist_left/color/image_raw/compressed`    | `sensor_msgs/CompressedImage` |
| Right wrist camera   | `/hdas/camera_wrist_right/color/image_raw/compressed`   | `sensor_msgs/CompressedImage` |
| Front chassis camera | `/hdas/camera_chassis_front_left/rgb/compressed`        | `sensor_msgs/CompressedImage` |

### Command topics (published by the client)

| Purpose            | Topic                                            | Message Type             |
| ------------------ | ------------------------------------------------ | ------------------------ |
| Left arm commands  | `/motion_target/target_joint_state_arm_left`     | `sensor_msgs/JointState` |
| Right arm commands | `/motion_target/target_joint_state_arm_right`    | `sensor_msgs/JointState` |
| Left gripper       | `/motion_control/position_control_gripper_left`  | `std_msgs/Float32`       |
| Right gripper      | `/motion_control/position_control_gripper_right` | `std_msgs/Float32`       |

All sensor topics use `BEST_EFFORT` reliability, `KEEP_LAST(1)` history, and `VOLATILE` durability.

## Choosing a server config

The server-side OpenPI config must match the observation format your mapping produces:

| Client mapping   | Compatible server configs | Norm stats |
| ---------------- | ------------------------- | ---------- |
| `droid-dual-arm` | `pi05_droid`, `pi05_base` | `droid`    |
| `aloha`          | `pi05_aloha`, `pi0_aloha` | `trossen`  |

For best results with the R1 Pro's 7-DOF arms, use `pi05_base` with `droid` normalization stats and a custom data transform that handles all 16 state dimensions. See the [OpenPI norm stats docs](https://github.com/physical-intelligence/openpi/blob/main/docs/norm_stats.md) for details on action space definitions.

## Troubleshooting

**No sensor data received**: Verify the R1 Pro hardware stack is running (`ros2 topic list | grep hdas`). Check `ROS_DOMAIN_ID` matches (R1 Pro default: 72).

**Connection refused / timeout**: Ensure the server is running and the firewall allows UDP 5555. The QUIC sidecar will retry connections automatically. Test server reachability with `curl http://<server-ip>:8000/healthz` (the HTTP health endpoint).

**Sidecar binary not found**: The client looks for the sidecar at `quic-sidecar/target/release/openpi-quic-sidecar` or `/usr/local/bin/openpi-quic-sidecar`. Override with `OPENPI_QUIC_SIDECAR_BINARY=/path/to/binary`.

**Image decode failures**: The R1 Pro publishes compressed JPEG images. Ensure `opencv-python-headless` is installed.
