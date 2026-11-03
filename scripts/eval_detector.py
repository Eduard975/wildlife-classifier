"""Detector step 3: accuracy of the TFLite detectors on test-split full frames.

Reference = LILA's MegaDetector v5 boxes (conf >= 0.2). Per class: the fraction of
frames where the detector finds an animal box with IoU >= 0.5 to a reference box.
On frames where MegaDetector v5 found nothing, the fraction with any animal detection.

Usage: .venv-export/bin/python scripts/eval_detector.py --run MDV6-yolov10-c_640
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from quantize_detector import letterbox

ROOT = Path(__file__).resolve().parents[1]
COCO_ANIMALS = list(range(14, 24))  # bird .. giraffe
THRESHOLDS = (0.1, 0.25, 0.4)


def run_model(interp, x):
    inp = interp.get_input_details()[0]
    if inp["dtype"] == np.int8:
        s, z = inp["quantization"]
        x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x[None])
    interp.invoke()
    outs = {}
    for o in interp.get_output_details():
        y = interp.get_tensor(o["index"])[0]
        if o["dtype"] == np.int8:
            s, z = o["quantization"]
            y = (y.astype(np.float32) - z) * s
        outs[y.shape[0] if y.shape[0] < y.shape[-1] else y.shape[-1]] = y if y.shape[0] < y.shape[-1] else y.T
    boxes = outs.pop(4)
    return boxes, next(iter(outs.values()))  # (4, N) xywh / imgsz, (nc, N)


def animal_scores(scores, names):
    return scores[COCO_ANIMALS].max(0) if len(names) == 80 else scores[names.index("animal")]


def nms(xyxy, conf, iou=0.45, top=20):
    order, keep = np.argsort(-conf), []
    while len(order) and len(keep) < top:
        i, order = order[0], order[1:]
        keep.append(i)
        order = order[iou_one(xyxy[i], xyxy[order]) < iou]
    return keep


def iou_one(a, b):
    tl, br = np.maximum(a[:2], b[:, :2]), np.minimum(a[2:], b[:, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 1)
    return inter / (np.prod(a[2:] - a[:2]) + np.prod(b[:, 2:] - b[:, :2], 1) - inter)


def detect(interp, path, size, names, thr):
    im = Image.open(path).convert("RGB")
    x, s = letterbox(im, size)
    t = time.perf_counter()
    boxes, scores = run_model(interp, x)
    ms = (time.perf_counter() - t) * 1000
    conf = animal_scores(scores, names)
    m = conf >= thr
    cx, cy, w, h = boxes[:, m] * size / s / np.array([[im.width], [im.height], [im.width], [im.height]])
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    k = nms(xyxy, conf[m]) if m.any() else []
    return xyxy[k], conf[m][k], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    d = ROOT / "runs/detector" / args.run
    size = int(args.run.rsplit("_", 1)[1])
    names = (d / "names.txt").read_text().split("\n")
    frames = pd.read_parquet(ROOT / "data/manifests/frames.parquet").query("role == 'eval'")
    ref = pd.read_parquet(ROOT / "data/manifests/frame_boxes.parquet")
    ref_by = {fp: g[["x", "y", "w", "h"]].to_numpy() for fp, g in ref.groupby("frame_path")}
    result = {}
    for variant in ("fp32", "int8_per_channel", "int8_per_tensor"):
        interp = tf.lite.Interpreter(str(d / f"{variant}.tflite"), num_threads=4)
        interp.allocate_tensors()
        rows, lat = [], []
        for f in frames.itertuples():
            xyxy, conf, ms = detect(interp, ROOT / "data/frames" / f.frame_path, size, names, min(THRESHOLDS))
            lat.append(ms)
            r = ref_by.get(f.frame_path)
            if r is not None:
                r = np.c_[r[:, :2], r[:, :2] + r[:, 2:]]
                best = np.array([iou_one(b, r).max() for b in xyxy]) if len(xyxy) else np.zeros(0)
            for t in THRESHOLDS:
                sel = conf >= t
                rows.append({"label": f.label, "has_ref": r is not None, "thr": t,
                             "hit": bool(r is not None and (best[sel] >= 0.5).any()), "any": bool(sel.any())})
        df = pd.DataFrame(rows)
        pos, neg = df[df.has_ref], df[~df.has_ref]
        result[variant] = {
            "frame_recall": {str(t): pos[pos.thr == t].groupby("label").hit.mean().round(3).to_dict()
                             for t in THRESHOLDS},
            "detect_rate_on_md_empty": neg.groupby("thr")["any"].mean().round(3).to_dict(),
            "mac_cpu_ms": round(float(np.median(lat[1:])), 1),
        }
        print(args.run, variant, json.dumps(result[variant]))
    (d / "eval.json").write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
