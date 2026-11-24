"""Export the frozen ImageNet EfficientNet-Lite0 backbone (pooled features) to FP32 TFLite.

Step 1 (.venv):        python scripts/export_backbone.py onnx --arch efficientnet_lite0
Step 2 (.venv-export): python scripts/export_backbone.py tflite --arch efficientnet_lite0
Writes runs/export_backbone/<arch>/{backbone.onnx, saved_model, backbone_fp32.tflite}.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_onnx(arch, size, d):
    import timm
    import torch
    from rafl.models.classifier import TIMM_NAMES
    m = timm.create_model(TIMM_NAMES[arch], pretrained=True, num_classes=0).eval()
    torch.onnx.export(m, torch.zeros(1, 3, size, size), d / "backbone.onnx", input_names=["input"],
                      output_names=["features"], opset_version=17, dynamo=False)


def to_tflite(d):
    import tensorflow as tf
    sm = d / "saved_model"
    if not (sm / "saved_model.pb").exists():
        subprocess.run([str(Path(sys.executable).parent / "onnx2tf"), "-i", str(d / "backbone.onnx"),
                        "-o", str(sm), "-osd", "-n"], check=True, cwd=d)
    out = d / "backbone_fp32.tflite"
    out.write_bytes(tf.lite.TFLiteConverter.from_saved_model(str(sm)).convert())
    print(out, out.stat().st_size // 1024, "KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["onnx", "tflite"])
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    d = ROOT / "runs/export_backbone" / args.arch
    d.mkdir(parents=True, exist_ok=True)
    to_onnx(args.arch, args.size, d) if args.step == "onnx" else to_tflite(d)


if __name__ == "__main__":
    main()
