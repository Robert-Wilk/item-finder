"""
perception_node
----------------
Subscribes to the camera topic, runs a PRE-QUANTIZED TFLite COCO
detector on incoming frames (throttled -- you do NOT need every
frame), and publishes detections as a JSON string.

Backend: tflite-runtime + Google's pre-quantized SSD MobileNet v1,
NOT PyTorch/Ultralytics. On a Pi 3B+ (899MB usable RAM, no swap),
full PyTorch's baseline memory footprint alone can exhaust available
RAM once your base ROS2 nodes (camera, IMU, ultrasonic, base_driver)
are already running -- this showed up as the perception process
getting stuck/OOM-killed with zero warning. tflite-runtime is a much
smaller inference-only package with no such issue.

Kept dependency-light and JSON-based on purpose: defining custom .msg
types costs real setup time you don't have this weekend. std_msgs/String
with a JSON payload is a fine, honest engineering trade-off for a
hackathon timeline.

TODO before running:
  - Confirm your camera topic name/type matches the parameter below.
  - Download the model (see README / models/ dir) before first run:
      wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
      unzip into src/item_finder/models/
  - pip install tflite-runtime --break-system-packages
  - Test standalone first (scripts/test_detector.py) against your
    ACTUAL demo objects before trusting this in the full pipeline.
"""

from ament_index_python.packages import get_package_share_directory

import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # tflite-runtime has no published wheel for some newer Python/aarch64
    # combos (confirmed: none for cp312 aarch64 at time of writing).
    # tensorflow's built-in tf.lite.Interpreter has an identical API and
    # DOES publish aarch64/cp312 wheels -- heavier install, but it's the
    # only backend that supports this model's custom
    # TFLite_Detection_PostProcess op (cv2.dnn's TFLite importer does not).
    import tensorflow.lite as tflite


# --- Model IO conventions for this specific SSD MobileNet v1 quant model ---
# Input: 300x300x3, uint8 (already quantized -- no float normalization needed)
# Output tensors (in this fixed order for this model):
#   0: bounding boxes [N, 4] as [ymin, xmin, ymax, xmax], normalized 0-1
#   1: class indices [N]  (0-indexed into labelmap, offset by 1 -- see load_labels)
#   2: confidence scores [N]
#   3: number of valid detections
MODEL_INPUT_SIZE = 300


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        pkg_dir = get_package_share_directory('item_finder')
        default_model = os.path.join(
            pkg_dir, 'models', 'detect.tflite'
        )
        default_labels = os.path.join(
            pkg_dir, 'models', 'labelmap.txt'
        )
        
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('labels_path', default_labels)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('inference_hz', 2.0)

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        labels_path = self.get_parameter('labels_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        inference_hz = self.get_parameter('inference_hz').value

        self._min_period = 1.0 / inference_hz
        self._last_inference_time = 0.0

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading TFLite model: {model_path} ...')
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = load_labels(labels_path)
        self.get_logger().info(f'Model loaded. {len(self.labels)} labels.')

        self.detections_pub = self.create_publisher(String, '/detections', 10)

        self.image_sub = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data
        )

        self.get_logger().info(
            f'perception_node ready. Subscribed to {image_topic}, '
            f'throttled to {inference_hz} Hz.'
        )

    def on_image(self, msg: Image):
        now = time.time()
        if now - self._last_inference_time < self._min_period:
            return
        self._last_inference_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        import cv2
        resized = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

        frame_h, frame_w = frame.shape[:2]
        detections = []
        for i in range(num_detections):
            score = float(scores[i])
            if score < self.conf_thresh:
                continue
            class_id = int(classes[i])
            if class_id < 0 or class_id >= len(self.labels):
                continue
            class_name = self.labels[class_id]

            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({
                'class': class_name,
                'confidence': score,
                # convert normalized coords back to pixel coords in the ORIGINAL frame
                'bbox': [
                    float(xmin * frame_w), float(ymin * frame_h),
                    float(xmax * frame_w), float(ymax * frame_h),
                ],
            })

        out = String()
        out.data = json.dumps({'stamp': now, 'detections': detections})
        self.detections_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()