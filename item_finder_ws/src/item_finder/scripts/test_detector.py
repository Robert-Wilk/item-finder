#!/usr/bin/env python3
"""
Run this FIRST, before wiring anything into ROS 2.

Points the Pi camera (or a USB webcam / a folder of test photos) at your
actual demo objects and checks the detector actually sees them reliably.
This is the highest-risk unknown in the whole build - find out early
whether your target objects are being detected consistently, or whether
you need to narrow your TARGET_CLASSES list.

Usage:
  python3 scripts/test_detector.py --source 0        # webcam/picam index
  python3 scripts/test_detector.py --source photo.jpg # single image
  python3 scripts/test_detector.py --source ./test_photos/  # folder
"""

import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO

from item_finder.target_classes import TARGET_CLASSES


def run_on_frame(model, frame, conf_thresh):
    t0 = time.time()
    results = model.predict(frame, imgsz=320, conf=conf_thresh, verbose=False)
    dt = time.time() - t0

    r = results[0]
    found = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])
        found.append((name, conf))

    target_hits = [f for f in found if f[0] in TARGET_CLASSES]
    other = [f for f in found if f[0] not in TARGET_CLASSES]

    print(f'  inference time: {dt*1000:.0f} ms')
    if target_hits:
        print(f'  TARGET matches: {target_hits}')
    else:
        print('  no target-class matches')
    if other:
        print(f'  other detections: {other}')

    return r.plot()  # annotated frame, for optional visual check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='camera index, image path, or folder')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--save-annotated', action='store_true',
                         help='save annotated frames to ./test_output/')
    args = parser.parse_args()

    print(f'Loading {args.model} ...')
    model = YOLO(args.model)
    print(f'Target classes to verify: {TARGET_CLASSES}\n')

    if args.save_annotated:
        os.makedirs('test_output', exist_ok=True)

    # Folder of images
    if os.path.isdir(args.source):
        paths = sorted(glob.glob(os.path.join(args.source, '*')))
        for p in paths:
            frame = cv2.imread(p)
            if frame is None:
                continue
            print(f'--- {p} ---')
            annotated = run_on_frame(model, frame, args.conf)
            if args.save_annotated:
                cv2.imwrite(os.path.join('test_output', os.path.basename(p)), annotated)
        return

    # Single image file
    if os.path.isfile(args.source):
        frame = cv2.imread(args.source)
        print(f'--- {args.source} ---')
        annotated = run_on_frame(model, frame, args.conf)
        if args.save_annotated:
            cv2.imwrite(os.path.join('test_output', 'result.jpg'), annotated)
        return

    # Live camera (index)
    cam_index = int(args.source)
    cap = cv2.VideoCapture(cam_index)
    print('Press Ctrl+C to stop.\n')
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Failed to read frame.')
                break
            print(f'--- frame {frame_no} ---')
            annotated = run_on_frame(model, frame, args.conf)
            if args.save_annotated:
                cv2.imwrite(os.path.join('test_output', f'frame_{frame_no:04d}.jpg'), annotated)
            frame_no += 1
            time.sleep(0.5)  # don't hammer the CPU while eyeballing results
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == '__main__':
    main()
