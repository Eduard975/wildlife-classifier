"""Cache the input of the deployed classifier's last block for federating more than the head (tensorflow env).

The deployed int8 EfficientNet-Lite0 ends in
    [7x7x320 int8] -> CONV_2D 1x1 320->1280 + ReLU6 -> MEAN -> FULLY_CONNECTED 1280->5.
This script adds the 7x7x320 tensor as a model output, runs every crop once on the CPU, and stores
it as int8 (data/features/qat_block_q.npy, 469 MB for 29,895 crops). It also stores the dequantized
1x1 conv weights and bias and the head as the starting point (runs/fl_hetero/init_block.npz).
A client in the "block" tier trains the 1x1 conv and the head (416,005 parameters); a client in the
"head" tier trains only the head (6,405 parameters) on top of the current conv.

  .venv-export/bin/python scripts/build_block_features.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def ops_of(model):
    sg = model.subgraphs[0]
    codes = {i: oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode for i, oc in enumerate(model.operatorCodes)}
    return sg, [(codes[op.opcodeIndex], op) for op in sg.operators]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qat/efficientnet_lite0/efficientnet_lite0_int8_per_tensor.tflite")
    ap.add_argument("--out", default="runs/fl_hetero")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = flatbuffer_utils.read_model(str(ROOT / args.model))
    sg, ops = ops_of(model)
    B = schema.BuiltinOperator
    assert [c for c, _ in ops[-3:]] == [B.CONV_2D, B.MEAN, B.FULLY_CONNECTED], "unexpected graph tail"
    conv, fc = ops[-3][1], ops[-1][1]
    x_t, wc_t, bc_t = (int(i) for i in conv.inputs)
    z_t = int(conv.outputs[0])
    feat_t, w_t, b_t = (int(i) for i in fc.inputs)
    y_t = int(fc.outputs[0])
    sg.outputs = np.array([y_t, x_t, feat_t], np.int32)
    dst = out / "classifier_block_out.tflite"
    flatbuffer_utils.write_model(model, str(dst))

    it = tf.lite.Interpreter(str(dst), num_threads=8)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    q = {k: td[i]["quantization"] for k, i in (("x", x_t), ("wc", wc_t), ("bc", bc_t), ("z", z_t), ("feat", feat_t),
                                                ("w", w_t), ("b", b_t))}
    Wc = it.get_tensor(wc_t).reshape(1280, 320).astype(np.float32).T * q["wc"][0]   # (320, 1280)
    bc = it.get_tensor(bc_t).astype(np.float32) * q["bc"][0]
    W = it.get_tensor(w_t).astype(np.float32).T * q["w"][0]                          # (1280, 5)
    b = it.get_tensor(b_t).astype(np.float32) * q["b"][0]
    np.savez(out / "init_block.npz", Wc=Wc, bc=bc, W=W, b=b)
    json.dump({k: [float(v[0]), int(v[1])] for k, v in q.items()}, open(out / "block_quant.json", "w"), indent=1)

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & df.label.isin(cfg["models"]["classifier"]["classes"])].reset_index(drop=True)
    if args.limit:
        df = df.sample(args.limit, random_state=0).reset_index(drop=True)
    i_in = it.get_input_details()[0]
    outs = {d["index"]: d for d in it.get_output_details()}
    s_in, z_in = i_in["quantization"]
    xs = np.zeros((len(df), 49, 320), np.int8)
    fq = np.zeros((len(df), 1280), np.int8)
    t0 = time.time()
    for k, p in enumerate(df.crop_path):
        im = Image.open(ROOT / p).convert("RGB")
        if im.size != (224, 224):
            im = im.resize((224, 224), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        it.set_tensor(i_in["index"], np.clip(np.round(x / s_in + z_in), -128, 127).astype(np.int8)[None])
        it.invoke()
        xs[k] = it.get_tensor(x_t).reshape(49, 320)
        fq[k] = it.get_tensor(feat_t).reshape(1280)
    print(f"{len(df)} crops in {time.time() - t0:.0f} s", flush=True)

    # float recompute of the pooled features from the cached block input must match the int8 model
    xf = (xs[:500].astype(np.float32) - q["x"][1]) * q["x"][0]
    f_rec = np.clip(xf @ Wc + bc, 0, 6).mean(1)
    f_int8 = (fq[:500].astype(np.float32) - q["feat"][1]) * q["feat"][0]
    cos = float(np.mean(np.sum(f_rec * f_int8, 1) / (np.linalg.norm(f_rec, axis=1) * np.linalg.norm(f_int8, axis=1))))
    agree = float(((f_rec @ W + b).argmax(1) == (f_int8 @ W + b).argmax(1)).mean())
    print(f"float recompute vs int8 features: cosine {cos:.5f}, same prediction on {100 * agree:.1f}% of 500 crops")

    tag = "qat_block" if not args.limit else f"qat_block_limit{args.limit}"
    np.save(ROOT / f"data/features/{tag}_q.npy", xs)
    df[["crop_id", "label", "split", "dataset", "location_id", "seq_key", "verified"]].to_parquet(
        ROOT / f"data/features/{tag}_meta.parquet")
    print("wrote", ROOT / f"data/features/{tag}_q.npy")


if __name__ == "__main__":
    main()
