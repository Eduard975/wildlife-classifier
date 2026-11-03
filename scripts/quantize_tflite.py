"""Phase 4 step 2 (tensorflow env): ONNX -> SavedModel (onnx2tf) -> int8 TFLite.

Full-integer quantization with int8 input/output, calibrated on calib.npy.
Produces both per-channel (TFLite default) and per-tensor weight variants so the
NPU speed/accuracy difference can be measured rather than assumed.

Usage: .venv-export/bin/python scripts/quantize_tflite.py --arch efficientnet_lite0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    args = ap.parse_args()
    d = ROOT / "runs/export" / args.arch
    sm = d / "saved_model"
    if not (sm / "saved_model.pb").exists():
        subprocess.run([str(Path(sys.executable).parent / "onnx2tf"), "-i", str(d / "model.onnx"),
                        "-o", str(sm), "-osd", "-n"], check=True, cwd=d)
    calib = (np.load(d / "calib.npy") - MEAN) / STD  # NHWC, normalized like training

    def rep():
        for i in range(len(calib)):
            yield [calib[i:i + 1].astype(np.float32)]

    for scheme in ("per_channel", "per_tensor"):
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        if scheme == "per_tensor":
            conv._experimental_disable_per_channel = True
        out = d / f"{args.arch}_int8_{scheme}.tflite"
        out.write_bytes(conv.convert())
        print(out.name, out.stat().st_size // 1024, "KB")
    fconv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    (d / f"{args.arch}_fp32.tflite").write_bytes(fconv.convert())


if __name__ == "__main__":
    main()
