"""Phase 4 step 3 (tensorflow env): accuracy of the exported TFLite models on the
test split (excluding hand-marked 'n'), same preprocessing as on the board.

Usage: .venv-export/bin/python scripts/eval_tflite.py --arch efficientnet_lite0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def run(path: Path, imgs: np.ndarray) -> np.ndarray:
    it = tf.lite.Interpreter(str(path), num_threads=8)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in imgs:
        x = ((x - MEAN) / STD)[None]
        if i["dtype"] == np.int8:
            s, z = i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], x)
        it.invoke()
        y = it.get_tensor(o["index"])[0].astype(np.float32)
        if o["dtype"] == np.int8:
            s, z = o["quantization"]
            y = (y - z) * s
        out.append(y)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    t = df[df.ok & (df.split == "test") & df.label.isin(classes) & (df.verified != "n")].reset_index(drop=True)
    y = t.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    imgs = np.stack([np.asarray(Image.open(ROOT / p).convert("RGB"), np.float32) / 255.0 for p in t.crop_path])

    d = ROOT / "runs/export" / args.arch
    res, ref = {}, None
    for name in ("fp32", "int8_per_channel", "int8_per_tensor"):
        logits = run(d / f"{args.arch}_{name}.tflite", imgs)
        pred = logits.argmax(1)
        ref = pred if ref is None else ref
        r = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(classes)}
        ver = {c: float((pred[(t.label == c) & (t.verified == "y")] == classes.index(c)).mean()) for c in ("wolf", "bear")}
        res[name] = {"bacc": float(np.mean(list(r.values()))), "recall": r, "verified_recall": ver,
                     "agree_with_fp32": float((pred == ref).mean())}
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()})
                                for k, v in res[name].items()}), flush=True)
    json.dump(res, open(d / "tflite_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
