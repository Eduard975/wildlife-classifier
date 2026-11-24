"""Reproduction check for the on-board quantizer (tensorflow env, Mac).

Calibrates the FP32 model with the converter's own calibration inputs, quantizes with rafl.edge.requant
from the FP32 weights, writes the result into the int8 template, and compares it with the converter's
file: scales, zero points, int8 weights, int32 biases, and model outputs on held-out inputs.

  .venv-export/bin/python scripts/check_requant.py --model cls
  .venv-export/bin/python scripts/check_requant.py --model det
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rafl.edge import requant  # noqa: E402
from quantize_detector import letterbox  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODELS = {
    "cls": ("runs/qat/efficientnet_lite0/efficientnet_lite0_int8_per_tensor.tflite",
            "runs/qat/efficientnet_lite0/efficientnet_lite0_fp32.tflite", "runs/requant/cls_quant_map.json"),
    "det": ("runs/qat_detector/MDV6-yolov10-c_416/int8_per_tensor.tflite",
            "runs/qat_detector/MDV6-yolov10-c_416/fp32.tflite", "runs/requant/det_quant_map.json"),
}


def calib_inputs(model):
    if model == "cls":
        c = (np.load(ROOT / "runs/qat/efficientnet_lite0/calib.npy") - MEAN) / STD
        return [c[i:i + 1].astype(np.float32) for i in range(len(c))]
    paths = sorted((ROOT / "data/frames/calib").glob("*.jpg"))[:200]
    return [letterbox(Image.open(p).convert("RGB"), 416)[0][None] for p in paths]


def eval_inputs(model, n):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    if model == "cls":
        df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
        df = df[df.ok & (df.split == "test")].sample(n, random_state=0)
        out = []
        for p in df.crop_path:
            im = Image.open(ROOT / p).convert("RGB").resize((224, 224), Image.BICUBIC)
            out.append(((np.asarray(im, np.float32) / 255.0 - MEAN) / STD)[None])
        return out
    frames = pd.read_parquet(ROOT / "data/manifests/frames.parquet").query("role == 'eval'").head(n)
    return [letterbox(Image.open(ROOT / "data/frames" / p).convert("RGB"), 416)[0][None] for p in frames.frame_path]


def run_int8(content, xs):
    it = tf.lite.Interpreter(model_content=content, num_threads=8)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    s, z = i["quantization"]
    res = []
    for x in xs:
        it.set_tensor(i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8))
        it.invoke()
        outs = []
        for o in it.get_output_details():
            so, zo = o["quantization"]
            outs.append((it.get_tensor(o["index"]).astype(np.float32) - zo) * so)
        res.append(outs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--n-eval", type=int, default=200)
    args = ap.parse_args()
    i8, f32, qmp = (ROOT / p for p in MODELS[args.model])
    qm = requant.load_map(qmp)
    tmpl = i8.read_bytes()

    t0 = time.time()
    cal = requant.calibrate(f32, qm, calib_inputs(args.model), num_threads=8)
    t_cal = time.time() - t0
    q = requant.quantize(qm, tmpl, cal["ranges"], requant.float_weights(f32.read_bytes(), qm))
    new = requant.write(tmpl, qm, q)

    old_p, new_p = requant.read_params(tmpl, qm), requant.read_params(new, qm)
    rel, zpd = {}, {}
    for t, e in qm["tensors"].items():
        so, zo = old_p[t]
        sn, zn = new_p[t]
        rel.setdefault(e["kind"], []).append(abs(sn - so) / max(so, 1e-12))
        zpd.setdefault(e["kind"], []).append(abs(zn - zo))
    wd, bd = [], []
    for w in qm["weighted"]:
        a = requant._data(tmpl, qm["tensors"][w["w"]]["data"], np.int8).astype(int)
        b = requant._data(new, qm["tensors"][w["w"]]["data"], np.int8).astype(int)
        wd.append(np.abs(a - b).max())
        if w["b"] is not None:
            a = requant._data(tmpl, qm["tensors"][w["b"]]["data"], "<i4").astype(np.int64)
            b = requant._data(new, qm["tensors"][w["b"]]["data"], "<i4").astype(np.int64)
            bd.append(np.abs(a - b).max())

    xs = eval_inputs(args.model, args.n_eval)
    (ROOT / f"runs/requant/{args.model}_requant.tflite").write_bytes(new)
    ya, yb = run_int8(tmpl, xs), run_int8(new, xs)
    res = {"model": args.model, "calib_inputs": cal["n"], "calib_s": round(t_cal, 1),
           "identical_file": new == tmpl,
           "scale_rel_diff_max": {k: float(np.max(v)) for k, v in rel.items()},
           "scale_rel_diff_median": {k: float(np.median(v)) for k, v in rel.items()},
           "zp_diff_max": {k: int(np.max(v)) for k, v in zpd.items()},
           "weight_lsb_diff_max": int(np.max(wd)), "weights_identical_ops": int(sum(d == 0 for d in wd)), "weighted_ops": len(wd),
           "bias_diff_max": int(np.max(bd)) if bd else None}
    if args.model == "cls":
        pa = np.array([y[0].reshape(-1).argmax() for y in ya])
        pb = np.array([y[0].reshape(-1).argmax() for y in yb])
        res["top1_agreement"] = float((pa == pb).mean())
        res["logit_max_abs_diff"] = float(max(np.abs(a[0] - b[0]).max() for a, b in zip(ya, yb)))
    else:
        nout = min(len(ya[0]), len(yb[0]))
        res["n_outputs"] = [len(ya[0]), len(yb[0])]
        res["output_max_abs_diff"] = [float(max(np.abs(a[k] - b[k]).max() for a, b in zip(ya, yb))) for k in range(nout)]
        res["output_mean_abs_diff"] = [float(np.mean([np.abs(a[k] - b[k]).mean() for a, b in zip(ya, yb)])) for k in range(nout)]
    out = ROOT / f"runs/requant/check_{args.model}.json"
    out.write_text(json.dumps(res, indent=1))
    (ROOT / f"runs/requant/{args.model}_requant.tflite").write_bytes(new)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
