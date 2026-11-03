"""Detector step 1 (torch env): Ultralytics .pt -> ONNX with separate box and score outputs.

Ultralytics concatenates pixel boxes and 0-1 scores into one tensor. In int8 that tensor
gets one scale and every score rounds to zero, so the head returns the two parts
separately and boxes are divided by the image size.

Usage: .venv/bin/python scripts/export_detector.py --weights models/detector/MDV6-yolov10-c.pt --imgsz 640
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxslim
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


class SplitHead(torch.nn.Module):
    """Returns (boxes xywh / imgsz, class probabilities) without Ultralytics' final concat."""

    def __init__(self, model, imgsz):
        super().__init__()
        self.model, self.imgsz = model, imgsz
        head = model.model[-1]
        head._inference = lambda x: (head._get_decode_boxes(x), x["scores"].sigmoid())

    def forward(self, x):
        boxes, scores = self.model(x)
        return boxes / self.imgsz, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    w = ROOT / args.weights
    yolo = YOLO(str(w))
    m = yolo.model.fuse().eval()
    head = m.model[-1]
    head.export, head.format = True, "onnx"
    if hasattr(head, "end2end"):
        head.end2end = False  # raw (4 + nc, N) output; NMS runs in post-processing
    out = ROOT / "runs/detector" / f"{w.stem}_{args.imgsz}"
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(SplitHead(m, args.imgsz), torch.zeros(1, 3, args.imgsz, args.imgsz),
                      out / "model.onnx", opset_version=17, input_names=["images"],
                      output_names=["boxes", "scores"], dynamo=False)
    onnx.save(onnxslim.slim(onnx.load(out / "model.onnx")), out / "model.onnx")  # onnx2tf needs folded anchors
    (out / "names.txt").write_text("\n".join(yolo.names[i] for i in sorted(yolo.names)))
    print(out / "model.onnx", sum(p.numel() for p in m.parameters()), "params")


if __name__ == "__main__":
    main()
