#!/usr/bin/env python3
"""
Produces the before/after optimization numbers for your submission writeup.
Judging criteria explicitly names "model speed" and "Arm-specific
optimization" - this script is cheap to run and gives you real numbers
to cite instead of a vague "it's optimized" claim.

Run on the Pi itself (not your laptop) - the numbers only mean
something as Arm/Cortex-A results.

Usage:
  python3 scripts/benchmark_model.py --model yolov8n.pt --runs 30
  python3 scripts/benchmark_model.py --model yolov8n_int8.tflite --runs 30

Export an int8 TFLite model first (one-time, do this on a laptop, copy
the resulting file to the Pi):
  from ultralytics import YOLO
  YOLO('yolov8n.pt').export(format='tflite', int8=True, imgsz=320)
"""

import argparse
import time

import numpy as np


def benchmark_ultralytics(model_path, runs, imgsz):
    from ultralytics import YOLO
    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup - first call includes model init overhead, don't count it
    for _ in range(3):
        model.predict(dummy, imgsz=imgsz, verbose=False)

    times = []
    for _ in range(runs):
        t0 = time.time()
        model.predict(dummy, imgsz=imgsz, verbose=False)
        times.append(time.time() - t0)

    return times


def benchmark_tflite(model_path, runs, imgsz):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    dtype = input_details[0]['dtype']
    dummy = np.random.randint(0, 255, input_details[0]['shape']).astype(dtype)

    for _ in range(3):
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()

    times = []
    for _ in range(runs):
        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        times.append(time.time() - t0)

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=320)
    args = parser.parse_args()

    print(f'Benchmarking {args.model} on this device ({args.runs} runs, imgsz={args.imgsz})...')

    if args.model.endswith('.tflite'):
        times = benchmark_tflite(args.model, args.runs, args.imgsz)
    else:
        times = benchmark_ultralytics(args.model, args.runs, args.imgsz)

    times_ms = [t * 1000 for t in times]
    print(f'\nModel: {args.model}')
    print(f'  mean:   {np.mean(times_ms):.1f} ms  ({1000/np.mean(times_ms):.1f} FPS)')
    print(f'  median: {np.median(times_ms):.1f} ms')
    print(f'  p95:    {np.percentile(times_ms, 95):.1f} ms')
    print(f'  min/max: {np.min(times_ms):.1f} / {np.max(times_ms):.1f} ms')
    print('\nRun this again with the quantized model and put both numbers in your writeup.')


if __name__ == '__main__':
    main()
