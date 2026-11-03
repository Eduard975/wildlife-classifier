"""Detector step 2 (tensorflow env): ONNX -> SavedModel (onnx2tf) -> FP32 and int8 TFLite.

Calibrates on letterboxed train-split frames (input in [0, 1], NHWC). int8 input and
output, per-channel weights (per-tensor is also written for the NPU speed comparison).

Usage: .venv-export/bin/python scripts/quantize_detector.py --run MDV6-yolov10-c_640
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def letterbox(im: Image.Image, size: int):
    """Resize keeping aspect, pad bottom/right with grey. Returns array in [0, 1] and scale."""
    s = size / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, (0, 0))
    return np.asarray(canvas, np.float32) / 255.0, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib", type=int, default=200)
    args = ap.parse_args()
    d = ROOT / "runs/detector" / args.run
    size = int(args.run.rsplit("_", 1)[1])
    sm = d / "saved_model"
    if not (sm / "saved_model.pb").exists():
        subprocess.run([str(Path(sys.executable).parent / "onnx2tf"), "-i", str(d / "model.onnx"),
                        "-o", str(sm), "-osd", "-n"], check=True, cwd=d)
    paths = sorted((ROOT / "data/frames/calib").glob("*.jpg"))[: args.calib]

    def rep():
        for p in paths:
            yield [letterbox(Image.open(p).convert("RGB"), size)[0][None]]

    (d / "fp32.tflite").write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(sm)).convert())
    for scheme in ("per_channel", "per_tensor"):
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if scheme == "per_tensor":
            conv._experimental_disable_per_channel = True
        out = d / f"int8_{scheme}.tflite"
        out.write_bytes(conv.convert())
        print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
