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

    def __init__(self, model, imgsz, level_decode=False):
        super().__init__()
        self.model, self.imgsz, self.level_decode = model, imgsz, level_decode
        head = model.model[-1]
        if level_decode:
            head._inference = lambda x: (self._level_boxes(head, x), x["scores"].sigmoid())
        else:
            head._inference = lambda x: (head._get_decode_boxes(x), x["scores"].sigmoid())

    def _level_boxes(self, head, x):
        from ultralytics.utils.tal import make_anchors

        feats = x["feats"]
        out = []
        for f, b, s in zip(feats, x["boxes"].split([f.shape[2] * f.shape[3] for f in feats], 2), head.stride):
            k = float(s) / self.imgsz
            a = make_anchors([f], [s], 0.5)[0].transpose(0, 1).unsqueeze(0) * k
            d = head.dfl(b) * k
            lt, rb = d[:, :2], d[:, 2:]
            out.append(torch.cat((a + (rb - lt) / 2, lt + rb), 1))
        return torch.cat(out, 2)

    def forward(self, x):
        boxes, scores = self.model(x)
        return (boxes if self.level_decode else boxes / self.imgsz), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--root", default="runs/detector")
    ap.add_argument("--state")
    ap.add_argument("--level-decode", action="store_true", help="decode boxes per pyramid level in normalized units")
    args = ap.parse_args()
    w = ROOT / args.weights
    yolo = YOLO(str(w))
    m = yolo.model.fuse().eval()
    if args.state:
        print(m.load_state_dict(torch.load(args.state, map_location="cpu"), strict=True))
    head = m.model[-1]
    head.export, head.format = True, "onnx"
    if hasattr(head, "end2end"):
        head.end2end = False  # raw (4 + nc, N) output; NMS runs in post-processing
    out = ROOT / args.root / f"{w.stem}_{args.imgsz}"
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(SplitHead(m, args.imgsz, args.level_decode), torch.zeros(1, 3, args.imgsz, args.imgsz),
                      out / "model.onnx", opset_version=17, input_names=["images"],
                      output_names=["boxes", "scores"], dynamo=False)
    onnx.save(onnxslim.slim(onnx.load(out / "model.onnx")), out / "model.onnx")  # onnx2tf needs folded anchors
    (out / "names.txt").write_text("\n".join(yolo.names[i] for i in sorted(yolo.names)))
    print(out / "model.onnx", sum(p.numel() for p in m.parameters()), "params")


if __name__ == "__main__":
    main()
