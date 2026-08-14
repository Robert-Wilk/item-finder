# On-Device Object Search Robot

Arm Create: AI Optimization Challenge 2026 - Track 1: Physical AI

Created by: Robert Wilk

MIT - see [LICENSE](./LICENSE)


## Project Overview
This project controls a mecanum-wheeled robot that searches its environment for a user-specified object (from a known household-item vocabulary) using an on-device COCO object detector, then autonomously navigates to it. All perception and decision-making runs locally on a Raspberry Pi 3B+ (Arm Cortex-A53) with no cloud round-trip. The detector has been quantized to int8 to reduce memory bandwidth and exploit ARM NEON acceleration, making real-time inference feasible on the Pi's constrained CPU.

## Functionality / Output
- User provides a target object (e.g. "find the remote") via `/find`
- `query_matcher_node` resolves free text to a known object class
- `perception_node` runs a pretrained COCO detector on-device (Arm Cortex-A53) on live camera frames
- `search_state_node` drives a state machine: SEARCHING -> APPROACH_VERIFY -> FOUND / TIMEOUT
- Robot autonomously sweeps the room, detects the target, and drives to it using the mecanum base's holonomic strafing

**Optimization results** (fill in after running `scripts/benchmark_model.py`):

| Model | Mean latency | FPS |
|---|---|---|
| fp32 baseline | TODO ms | TODO |
| int8 quantized | TODO ms | TODO |

## Hardware
- Raspberry Pi 3B+ (Arm Cortex-A53, quad-core)
- Mecanum wheel base
- IMU
- Front-facing ultrasonic sensor
- Pi Camera Module

## Architecture
```
camera_node -> perception_node -> /detections (JSON) -\
                                                          -> search_state_node -> /cmd_vel_raw -> safety_node -> base_driver
              query_matcher_node -> /target_class ------/                            ^
 /find (user input)                                                    ultrasonic_node -/
```

## Project structure
- item_finder - Includes path-planning, perception, and interface code that publishes messages to robot
- mecanum - Includes sensing and controller code as well as sets up pub/sub topics for controlling the robot

## Setup Instructions

### 1. Prerequisites
- ROS 2 (Humble or later), Ubuntu Server 22.04 64-bit
- Python 3.10+
- `pip install ultralytics opencv-python --break-system-packages`

### 2. Clone and build
```bash
# clone the repo
mkdir -p ~/item_finder_ws/src
cd ~/item_finder_ws/src
git clone <YOUR_REPO_URL> item_finder

# build the robot's operating workspace
cd ~/mecanum/mecanum_pca9685_ws
colcon build --symlink-install
source install/setup.bash

# build the AI item finder workspace
cd ~/item_finder/item_finder_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. TODO Run the mecanum launch to start pub/sub topics

### 4. Verify the detector against your objects (do this before anything else)
```bash
python3 src/item_finder/scripts/test_detector.py --source 0
```

### 5. Launch
```bash
# Terminal 1: your existing base robot stack (camera, IMU, ultrasonic, base_driver, safety_node, teleop)
# In this case mecanum.launch.py
ros2 launch <your_base_pkg> mecanum.launch.py

# Terminal 2: item_finder nodes
ros2 launch item_finder item_finder.launch.py

# Terminal 3: issue a search
ros2 topic pub /find std_msgs/String "data: 'tv remote'" --once
```

### 6. Watch it work
```bash
ros2 topic echo /search_state
ros2 topic echo /find_result
```

## Future Work
- Persisted topological map of explored areas across runs
- Curiosity-driven exploration using the same embedding/novelty approach for autonomous inspection
- Open-vocabulary matching (CLIP-style) for objects outside the fixed COCO vocabulary
- Replace the deterministic alias/fuzzy matcher with a small on-device LLM performing constrained classification, mapping arbitrary descriptive input (e.g. "red round blob") to the nearest valid COCO class.
