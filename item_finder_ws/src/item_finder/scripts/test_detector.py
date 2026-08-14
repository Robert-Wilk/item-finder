#!/usr/bin/env python3
"""
Run this FIRST, before wiring perception_node into the full launch file.

Subscribes to the SAME /image_raw topic your real camera_node already
publishes (rather than opening the camera device directly), so this:
  - avoids device-contention errors (cv2.VideoCapture fighting your
    already-running camera_node for the same /dev/video device)
  - tests the exact same data path perception_node will actually use

Requires your base robot's camera_node to already be running and
publishing /image_raw (confirm with `ros2 topic list`).

Usage:
  python3 scripts/test_detector.py                      # default /image_raw, live loop
  python3 scripts/test_detector.py --topic /image_raw
  python3 scripts/test_detector.py --frames 20           # stop after N frames instead of running forever
  python3 scripts/test_detector.py --save                # save annotated frames to ./test_output/
"""

import argparse
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from item_finder.target_classes import TARGET_CLASSES


class DetectorTestNode(Node):
    def __init__(self, topic, model_path, conf_thresh, max_frames, save):
        super().__init__('detector_test_node')
        self.bridge = CvBridge()
        self.conf_thresh = conf_thresh
        self.max_frames = max_frames
        self.save = save
        self.frame_count = 0

        if self.save:
            os.makedirs('test_output', exist_ok=True)

        self.get_logger().info(f'Loading {model_path} ...')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Target classes to verify: {TARGET_CLASSES}')
        self.get_logger().info(f'Subscribing to {topic} - waiting for frames...')

        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)

    def on_image(self, msg: Image):
        t_recv = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        t0 = time.time()
        results = self.model.predict(frame, imgsz=320, conf=self.conf_thresh, verbose=False)
        dt = time.time() - t0

        r = results[0]
        found = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]
            conf = float(box.conf[0])
            found.append((name, round(conf, 2)))

        target_hits = [f for f in found if f[0] in TARGET_CLASSES]
        other = [f for f in found if f[0] not in TARGET_CLASSES]

        print(f'\n--- frame {self.frame_count} ---')
        print(f'  inference time: {dt*1000:.0f} ms')
        print(f'  TARGET matches: {target_hits}' if target_hits else '  no target-class matches')
        if other:
            print(f'  other detections: {other}')

        if self.save:
            annotated = r.plot()
            cv2.imwrite(os.path.join('test_output', f'frame_{self.frame_count:04d}.jpg'), annotated)

        self.frame_count += 1
        if self.max_frames and self.frame_count >= self.max_frames:
            self.get_logger().info(f'Reached {self.max_frames} frames, shutting down.')
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/image_raw')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--frames', type=int, default=0, help='stop after N frames (0 = run forever)')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    node = DetectorTestNode(args.topic, args.model, args.conf, args.frames, args.save)
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