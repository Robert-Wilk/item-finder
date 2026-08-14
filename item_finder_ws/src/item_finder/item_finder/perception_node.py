"""
perception_node
----------------
Subscribes to the camera topic, runs a pretrained COCO object detector
on incoming frames (throttled - you do NOT need every frame), and
publishes detections as a JSON string.

Kept dependency-light and JSON-based on purpose: defining custom .msg
types costs real setup time you don't have this weekend. std_msgs/String
with a JSON payload is a fine, honest engineering trade-off for a
hackathon timeline - note it as a deliberate choice, not an oversight,
if it comes up.

TODO before running:
  - Confirm your camera topic name/type (raw Image vs CompressedImage)
    and adjust the subscription below accordingly.
  - pip install ultralytics --break-system-packages
  - Test this file's detection quality standalone (see scripts/test_detector.py)
    against your ACTUAL demo objects before wiring it into the full stack.
"""

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

# cv_bridge converts ROS Image <-> numpy/cv2 arrays
from cv_bridge import CvBridge

# --- Detector backend -------------------------------------------------
# Default: Ultralytics YOLOv8n (easy install, pretrained on COCO,
# reasonable speed on Pi-class ARM CPUs at low resolution / low rate).
from ultralytics import YOLO


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # --- Parameters (override via launch file or CLI) ---
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model_path', 'yolov8n.pt')  # auto-downloads on first run
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('inference_hz', 2.0)  # throttle - don't run every frame
        self.declare_parameter('image_size', 320)  # smaller = faster on Pi CPU

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        inference_hz = self.get_parameter('inference_hz').value
        self.imgsz = self.get_parameter('image_size').value

        self._min_period = 1.0 / inference_hz
        self._last_inference_time = 0.0

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading model: {model_path} ...')
        self.model = YOLO(model_path)
        self.get_logger().info('Model loaded.')

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
            return  # drop this frame, not time for inference yet
        self._last_inference_time = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf_thresh, verbose=False
        )

        detections = []
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],  # pixel coords in the resized frame
            })

        out = String()
        out.data = json.dumps({
            'stamp': now,
            'detections': detections,
        })
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