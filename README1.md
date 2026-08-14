# [Project Name] -- On-Device Object Search Robot

Arm Create: AI Optimization Challenge 2026 -- Track 1: Physical AI

## Project Overview
<!-- 2-4 sentences: what is this, and why should it win?
     e.g. "A mecanum-wheeled robot that searches a room for a
     user-described object using an on-device COCO detector, then
     drives to it -- all inference and decision-making runs locally
     on a Raspberry Pi 3B+ (Arm Cortex-A53), no cloud round-trip." -->

TODO

## Functionality / Output
<!-- What does it actually do, end to end? What's the final output?
     Describe the sense -> decide -> act loop explicitly -- that's
     literally the track's eligibility definition, make it easy for
     judges to see you hit it. -->

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
camera_node (/image_raw) -> perception_node -> /detections (JSON) -\
                                                                      -> search_state_node -> /cmd_vel -> base_driver
             query_matcher_node -> /target_class -------------------/         ^
 /find (user input)                                                            |
                                              ultrasonic (/ultrasonic/range) --/
```
`search_state_node` includes an inline ultrasonic hard-stop (no separate
safety_node in this robot's stack) -- forward motion is blocked whenever
the ultrasonic reading is below `ULTRASONIC_STOP_DISTANCE_M` or the
reading is stale. Note the ultrasonic is front-facing only, so this
guards forward approach, not lateral strafing.

## Engineering Notes
<!-- This is real content for your writeup, not just internal notes --
     a genuine hardware-constraint story is more credible to judges
     than a hypothetical optimization narrative. -->

During development we initially built the detector on Ultralytics
YOLOv8n (PyTorch backend). On the Pi 3B+ (899MB usable RAM, no swap
configured), loading PyTorch alongside our already-running base nodes
(camera, IMU, ultrasonic, base_driver -- collectively ~150MB+ RSS)
exhausted available memory, causing the perception process to hang
under memory pressure rather than fail cleanly.

We switched to **tflite-runtime** with a pre-quantized (int8) SSD
MobileNet v1 COCO model. This wasn't just a speed optimization --
**it was the difference between the perception pipeline running at
all versus not fitting in the device's memory budget.** This is a
direct, concrete demonstration of why Arm-specific model optimization
(quantization, lightweight runtimes) matters for real edge deployment,
not just a benchmark exercise.

| Approach | Result on Pi 3B+ |
|---|---|
| PyTorch + YOLOv8n (fp32) | Did not complete -- memory exhaustion under concurrent ROS2 node load |
| tflite-runtime + int8 SSD MobileNet v1 | TODO: fill in mean inference time from test run |

## Setup Instructions
<!-- Step-by-step, assume a judge is setting this up on their own Arm device -->

### 1. Prerequisites
- ROS 2 (Humble or later), Ubuntu Server 22.04 64-bit
- Python 3.10+
- Add swap (Pi 3B+ has no swap by default -- without it, memory
  pressure can hang a process instead of failing cleanly):
  ```bash
  sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  ```
- `pip install opencv-python numpy --break-system-packages`
- TFLite backend -- try the lightweight option first, it may not have
  a wheel for your Python/platform (confirmed unavailable for
  cp312/aarch64 at time of writing):
  ```bash
  pip install tflite-runtime --break-system-packages
  # if that fails with "No matching distribution":
  pip install tensorflow --break-system-packages   # heavier, but has
                                                      # aarch64/cp312 wheels
                                                      # and an identical API
  ```

### 2. Clone and build
```bash
mkdir -p ~/item_finder_ws/src
cd ~/item_finder_ws/src
git clone <YOUR_REPO_URL> item_finder
cd ~/item_finder_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Download the detection model
```bash
cd src/item_finder
mkdir -p models && cd models
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```
This is Google's pre-converted, pre-quantized (int8) SSD MobileNet v1
COCO detector -- chosen specifically because it's small enough (a few
MB) to run comfortably within a Raspberry Pi 3B+'s ~899MB usable RAM.

### 4. Verify the detector against your objects (do this before anything else)
```bash
python3 scripts/test_detector.py --frames 20
```

### 4. Launch
```bash
# Terminal 1: your existing base robot stack (camera, IMU, ultrasonic, base_driver, safety_node, teleop)
ros2 launch <your_base_pkg> base.launch.py

# Terminal 2: item_finder nodes
ros2 launch item_finder item_finder.launch.py

# Terminal 3: issue a search
ros2 topic pub /find std_msgs/String "data: 'tv remote'" --once
```

### 5. Watch it work
```bash
ros2 topic echo /search_state
ros2 topic echo /find_result
```

## Future Work
<!-- Cheap, honest way to show depth without needing to build it under deadline -->
- Persisted topological map of explored areas across runs
- Curiosity-driven exploration using the same embedding/novelty approach for autonomous inspection
- Open-vocabulary matching (CLIP-style) for objects outside the fixed COCO vocabulary

## License
MIT -- see [LICENSE](./LICENSE)