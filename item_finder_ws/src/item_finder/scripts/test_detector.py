#!/usr/bin/env python3
"""
Run this FIRST, before wiring perception_node into the full launch file.

Subscribes to the SAME /image_raw topic your real camera_node already
publishes, using the SAME tflite-runtime backend as perception_node
(pre-quantized SSD MobileNet, not PyTorch -- see perception_node.py's
docstring for why: full PyTorch's memory footprint doesn't reliably
fit a Pi 3B+'s ~899MB usable RAM alongside your other running nodes).

Requires:
  - Your base robot's camera_node already running and publishing /image_raw
  - Model downloaded into src/item_finder/models/ (see README)
  - pip install tflite-runtime --break-system-packages

Usage:
  python3 scripts/test_detector.py
  python3 scripts/test_detector.py --frames 20
  python3 scripts/test_detector.py --conf 0.3
"""

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from item_finder.target_classes import TARGET_CLASSES

MODEL_INPUT_SIZE = 300


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


class DetectorTestNode(Node):
    def __init__(self, topic, model_path, labels_path, conf_thresh, max_frames):
        super().__init__('detector_test_node')
        self.bridge = CvBridge()
        self.conf_thresh = conf_thresh
        self.max_frames = max_frames
        self.frame_count = 0

        self.get_logger().info(f'Loading TFLite model: {model_path} ...')
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = load_labels(labels_path)

        self.get_logger().info(f'Target classes to verify: {TARGET_CLASSES}')
        self.get_logger().info(f'Subscribing to {topic} -- waiting for frames...')

        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        t0 = time.time()
        resized = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
        dt = time.time() - t0

        found = []
        for i in range(num_detections):
            score = float(scores[i])
            if score < self.conf_thresh:
                continue
            class_id = int(classes[i])
            if class_id < 0 or class_id >= len(self.labels):
                continue
            found.append((self.labels[class_id], round(score, 2)))

        target_hits = [f for f in found if f[0] in TARGET_CLASSES]
        other = [f for f in found if f[0] not in TARGET_CLASSES]

        print(f'\n--- frame {self.frame_count} ---')
        print(f'  inference time: {dt*1000:.0f} ms')
        print(f'  TARGET matches: {target_hits}' if target_hits else '  no target-class matches')
        if other:
            print(f'  other detections: {other}')

        self.frame_count += 1
        if self.max_frames and self.frame_count >= self.max_frames:
            self.get_logger().info(f'Reached {self.max_frames} frames, shutting down.')
            rclpy.shutdown()


def main():
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model = os.path.join(pkg_dir, 'models', 'detect.tflite')
    default_labels = os.path.join(pkg_dir, 'models', 'labelmap.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/image_raw')
    parser.add_argument('--model', default=default_model)
    parser.add_argument('--labels', default=default_labels)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--frames', type=int, default=0, help='stop after N frames (0 = run forever)')
    args = parser.parse_args()

    rclpy.init()
    node = DetectorTestNode(args.topic, args.model, args.labels, args.conf, args.frames)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()