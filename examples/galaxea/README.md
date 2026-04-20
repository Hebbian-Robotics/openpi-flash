# Galaxea Robot Integration

Connect Galaxea R1-series robots to a hosted OpenPI inference server. Each client runs on the robot's onboard computer and sends observations over QUIC for low-latency policy inference.

## Supported Robots

| Robot | Arm DOF | Cameras | Compute |
| ----- | ------- | ------- | ------- |
| R1 Pro | 7 per arm | Head (mono) + 2 wrist + 5 chassis | Jetson AGX Orin |
| R1 Lite | 6 per arm | Head (binocular) + 2 wrist | Intel NUC i9 |

Both robots are served by a single client: `galaxea_client.py --robot r1-pro` or `--robot r1-lite`.

## Prerequisites

- A Galaxea R1 robot with ROS 2 Humble workspace set up
  - [R1 Pro Software Guide](https://docs.galaxea-dynamics.com/Guide/R1Pro/software_introduction/R1Pro_Software_Guide_ROS2/)
  - [R1 Lite Software Guide](https://docs.galaxea-dynamics.com/Guide/R1Lite/software_introduction/ros2/R1Lite_Software_Introduction_ros2/)
- An OpenPI inference server running (see [`docs/aws-manual-setup.md`](../../docs/aws-manual-setup.md) or use Modal)
- Network connectivity between the robot and the server (UDP port 5555)

## Setup on the Robot

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Rust toolchain (needed to build the QUIC sidecar)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the hosting repo and build the transport binary
git clone <your-hosting-repo-url> ~/openpi-hosting
cd ~/openpi-hosting/flash-transport
cargo build --release

# Source ROS 2 (must be done before creating the venv so ROS packages are visible)
source /opt/ros/humble/setup.bash
source ~/r1_ws/install/setup.bash  # r1pro_ws or r1lite_ws depending on your robot

# Create a Python 3.11 venv that inherits system ROS packages
cd ~/openpi-hosting
uv venv --system-site-packages --python 3.11

# Install Python client dependencies into the venv
uv pip install openpi-client opencv-python-headless numpy
```

## Running the Client

### R1 Pro

```bash
source /opt/ros/humble/setup.bash
source ~/r1pro_ws/install/setup.bash

# Make sure the R1 Pro hardware stack is running:
#   ros2 launch HDAS r1pro.py
#   ros2 launch signal_camera_node signal_camera.py
#   ros2 launch mobiman r1_pro_jointTrackerdemo.py

cd ~/openpi-hosting

# DROID dual-arm mapping (recommended for R1 Pro — preserves all 7 joints)
uv run python examples/galaxea/galaxea_client.py \
    --robot r1-pro \
    --host <server-ip> \
    --prompt "pick up the red cup" \
    --mapping droid-dual-arm \
    --rate 15
```

### R1 Lite

```bash
source /opt/ros/humble/setup.bash
source ~/r1lite_ws/install/setup.bash

# Make sure the R1 Lite hardware stack is running:
#   ros2 launch HDAS r1lite.py

cd ~/openpi-hosting

# ALOHA mapping (recommended for R1 Lite — native 6-DOF fit, no joint dropping)
uv run python examples/galaxea/galaxea_client.py \
    --robot r1-lite \
    --host <server-ip> \
    --prompt "pick up the red cup" \
    --mapping aloha \
    --rate 15

# DROID mapping (leverages binocular head — 4 cameras)
uv run python examples/galaxea/galaxea_client.py \
    --robot r1-lite \
    --host <server-ip> \
    --prompt "pick up the red cup" \
    --mapping droid-dual-arm
```

## Embodiment Mappings

### R1 Pro Mappings

#### `droid-dual-arm` (recommended)

Extends the DROID 7-DOF format to dual arms. Best match for the R1 Pro's 7 joints per arm.

| Property      | Value                                             |
| ------------- | ------------------------------------------------- |
| State dims    | 16 (7 joints + gripper per arm)                   |
| Cameras       | 3 (head, left wrist, right wrist)                 |
| Image format  | HWC uint8, 224x224                                |
| Server config | `pi05_droid` or `pi05_base` with DROID norm stats |

#### `aloha`

Force-fits into ALOHA's 6-DOF dual-arm format by dropping one joint per arm. The dropped joint is held at a fixed position.

| Property      | Value                                            |
| ------------- | ------------------------------------------------ |
| State dims    | 14 (6 joints + gripper per arm)                  |
| Cameras       | 4 (head, front chassis, left wrist, right wrist) |
| Image format  | CHW uint8, 224x224                               |
| Server config | `pi05_aloha` or `pi0_aloha`                      |

### R1 Lite Mappings

#### `droid-dual-arm`

Maps the R1 Lite's 6-DOF arms into the DROID format, leveraging the binocular head camera for two exterior views.

| Property      | Value                                                  |
| ------------- | ------------------------------------------------------ |
| State dims    | 14 (6 joints + gripper per arm)                        |
| Cameras       | 4 (head left eye, head right eye, left wrist, right wrist) |
| Image format  | HWC uint8, 224x224                                     |
| Server config | `pi05_base` with custom 14-dim norm stats              |

#### `aloha` (recommended)

Native fit for ALOHA's 6-DOF format. No joints are dropped or restored.

| Property      | Value                                       |
| ------------- | ------------------------------------------- |
| State dims    | 14 (6 joints + gripper per arm)             |
| Cameras       | 3 (head, left wrist, right wrist)           |
| Image format  | CHW uint8, 224x224                          |
| Server config | `pi05_aloha` or `pi0_aloha` (3-camera variant) |

Note: Standard ALOHA uses 4 cameras. The R1 Lite has no chassis camera, so this mapping provides 3. Pass `--mapping aloha` and ensure your server config accepts 3 cameras.

## ROS 2 Topics Reference

### Sensor topics (subscribed by the client)

| Purpose              | Topic                                                   | Message Type                  | R1 Pro | R1 Lite |
| -------------------- | ------------------------------------------------------- | ----------------------------- | ------ | ------- |
| Left arm joints      | `/hdas/feedback_arm_left`                               | `sensor_msgs/JointState`      | 7 pos  | 6 pos   |
| Right arm joints     | `/hdas/feedback_arm_right`                              | `sensor_msgs/JointState`      | 7 pos  | 6 pos   |
| Left gripper         | `/hdas/feedback_gripper_left`                           | `sensor_msgs/JointState`      | yes    | yes     |
| Right gripper        | `/hdas/feedback_gripper_right`                          | `sensor_msgs/JointState`      | yes    | yes     |
| Head camera (left)   | `/hdas/camera_head/left_raw/image_raw_color/compressed` | `sensor_msgs/CompressedImage` | yes    | yes     |
| Head camera (right)  | `/hdas/camera_head/right_raw/image_raw_color/compressed`| `sensor_msgs/CompressedImage` | -      | yes     |
| Left wrist camera    | `/hdas/camera_wrist_left/color/image_raw/compressed`    | `sensor_msgs/CompressedImage` | yes    | yes     |
| Right wrist camera   | `/hdas/camera_wrist_right/color/image_raw/compressed`   | `sensor_msgs/CompressedImage` | yes    | yes     |
| Front chassis camera | `/hdas/camera_chassis_front_left/rgb/compressed`        | `sensor_msgs/CompressedImage` | yes    | -       |

### Command topics (published by the client)

| Purpose            | Topic                                            | Message Type             |
| ------------------ | ------------------------------------------------ | ------------------------ |
| Left arm commands  | `/motion_target/target_joint_state_arm_left`     | `sensor_msgs/JointState` |
| Right arm commands | `/motion_target/target_joint_state_arm_right`    | `sensor_msgs/JointState` |
| Left gripper       | `/motion_control/position_control_gripper_left`  | `std_msgs/Float32`       |
| Right gripper      | `/motion_control/position_control_gripper_right` | `std_msgs/Float32`       |

All sensor topics use `BEST_EFFORT` reliability, `KEEP_LAST(1)` history, and `VOLATILE` durability.

## Choosing a Server Config

| Client | Mapping | Compatible server configs | Norm stats |
| ------ | ------- | ------------------------- | ---------- |
| R1 Pro | `droid-dual-arm` | `pi05_droid`, `pi05_base` | `droid` |
| R1 Pro | `aloha` | `pi05_aloha`, `pi0_aloha` | `trossen` |
| R1 Lite | `droid-dual-arm` | `pi05_base` | custom (14-dim) |
| R1 Lite | `aloha` | `pi05_aloha`, `pi0_aloha` | `trossen` |

## Troubleshooting

**No sensor data received**: Verify the hardware stack is running (`ros2 topic list | grep hdas`). Check `ROS_DOMAIN_ID` matches (R1 Pro default: 72).

**Connection refused / timeout**: Ensure the server is running and the firewall allows UDP 5555. The QUIC sidecar will retry connections automatically. Test server reachability with `curl http://<server-ip>:8000/healthz`.

**Transport binary not found**: The client looks for the binary at `flash-transport/target/release/openpi-flash-transport` or `/usr/local/bin/openpi-flash-transport`. Override with `OPENPI_FLASH_TRANSPORT_BINARY=/path/to/binary`.

**Image decode failures**: Both robots publish compressed JPEG images. Ensure `opencv-python-headless` is installed.
