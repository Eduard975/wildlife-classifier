"""Prepare the deployed int8 classifier for federated learning on the NPU (tensorflow env).

1. Adds the pooled-feature tensor (input of the final FULLY_CONNECTED) as a second model
   output, so one NPU call returns logits and the features the FL head trains on.
2. Records the byte offsets of the head's weight and bias data and of the scales and zero
   points that must change when a new head is written in (patch_map.json). The board patches
   these bytes with numpy only (rafl.fl.npu_head); no flatbuffer library is needed there.
3. Computes int8 features of every crop with the new file on the CPU, plus the calibration
   features that fix the output range of a re-quantized head.

  .venv-export/bin/python scripts/build_fl_npu_model.py
Writes runs/fl_npu/{classifier_fl.tflite, patch_map.json, init_head.npz, calib_features_q.npy},
data/features/qat_int8_q.npy (int8) and data/features/qat_int8_meta.parquet.
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
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rafl.fl import npu_head  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def find_head(model):
    """Index of the final FULLY_CONNECTED op and its (input, weight, bias, output) tensors."""
    sg = model.subgraphs[0]
    codes = {i: oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode for i, oc in enumerate(model.operatorCodes)}
    fc = [op for op in sg.operators if codes[op.opcodeIndex] == schema.BuiltinOperator.FULLY_CONNECTED]
    assert len(fc) == 1, f"expected one FULLY_CONNECTED, found {len(fc)}"
    x, w, b = (int(i) for i in fc[0].inputs)
    y = int(fc[0].outputs[0])
    assert y in [int(o) for o in sg.outputs], "FC output is not the model output"
    return x, w, b, y


def offsets(buf: bytes, tensors: dict) -> dict:
    """Absolute byte offsets of buffer data and quantization vectors, read with the flatbuffer tables."""
    m = schema.Model.GetRootAs(buf, 0)
    sg = m.Subgraphs(0)
    out = {}
    for name, ti in tensors.items():
        t = sg.Tensors(ti)
        bt = m.Buffers(t.Buffer())
        ent = {"tensor": ti, "shape": t.ShapeAsNumpy().tolist()}
        o = bt._tab.Offset(4)
        if o:
            ent["data"] = [bt._tab.Vector(o), bt._tab.VectorLen(o)]
        q = t.Quantization()
        for field, slot in (("scale", 8), ("zero_point", 10)):
            o = q._tab.Offset(slot)
            ent[field] = [q._tab.Vector(o), q._tab.VectorLen(o)]
        out[name] = ent
    return out


def run_crops(path, files, size=224):
    it = (tf.lite.Interpreter(model_content=path, num_threads=8) if isinstance(path, bytes)
          else tf.lite.Interpreter(str(path), num_threads=8))
    it.allocate_tensors()
    i = it.get_input_details()[0]
    outs = it.get_output_details()
    s, z = i["quantization"]
    logits, feats = [], []
    for f in files:
        im = Image.open(f).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        it.set_tensor(i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8)[None])
        it.invoke()
        logits.append(it.get_tensor(outs[0]["index"]).reshape(-1).copy())
        if len(outs) > 1:
            feats.append(it.get_tensor(outs[1]["index"]).reshape(-1).copy())
    return np.array(logits), (np.array(feats) if feats else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qat/efficientnet_lite0/efficientnet_lite0_int8_per_tensor.tflite")
    ap.add_argument("--calib", default="runs/qat/efficientnet_lite0/calib.npy")
    ap.add_argument("--out", default="runs/fl_npu")
    ap.add_argument("--limit", type=int, default=0, help="only featurize this many crops (testing)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / args.model

    model = flatbuffer_utils.read_model(str(src))
    x_t, w_t, b_t, y_t = find_head(model)
    sg = model.subgraphs[0]
    sg.outputs = np.array([y_t, x_t], np.int32)
    dst = out / "classifier_fl.tflite"
    flatbuffer_utils.write_model(model, str(dst))
    buf = dst.read_bytes()

    pm = {"source": args.model, "feature_tensor": x_t,
          "offsets": offsets(buf, {"feat": x_t, "weight": w_t, "bias": b_t, "out": y_t})}
    it = tf.lite.Interpreter(str(dst))
    td = {t["index"]: t for t in it.get_tensor_details()}
    for k, ti in (("feat", x_t), ("weight", w_t), ("bias", b_t), ("out", y_t)):
        s, z = td[ti]["quantization"]
        pm[k] = {"scale": float(s), "zero_point": int(z), "shape": td[ti]["shape"].tolist()}
    (out / "patch_map.json").write_text(json.dumps(pm, indent=1))

    # the offsets must point at the values the interpreter reports
    W, b = npu_head.read_head(buf, pm)
    Wt, bt = it.get_tensor(w_t), it.get_tensor(b_t)
    assert np.allclose(W, Wt.astype(np.float32).T * pm["weight"]["scale"]), "weight offset wrong"
    assert np.allclose(b, bt.astype(np.float32) * pm["bias"]["scale"]), "bias offset wrong"
    np.savez(out / "init_head.npz", W=W, b=b)

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    df = df[df.ok & df.label.isin(cfg["models"]["classifier"]["classes"])].reset_index(drop=True)
    if args.limit:
        df = df.sample(args.limit, random_state=0).reset_index(drop=True)

    # same logits as the deployed file, and the extra output is the FC input
    probe = [ROOT / p for p in df.crop_path[:64]]
    l_old, _ = run_crops(src, probe)
    l_new, f_new = run_crops(dst, probe)
    assert np.array_equal(l_old, l_new), "two-output model changed the logits"
    emu = npu_head.int8_logits(f_new, *npu_head.head_q(buf, pm), pm)
    agree = float((emu == l_new).mean())
    print(f"int8 head emulation equals TFLite on {100 * agree:.2f}% of logits", flush=True)

    calib = np.load(ROOT / args.calib)
    it2 = tf.lite.Interpreter(str(dst), num_threads=8)
    it2.allocate_tensors()
    i, outs = it2.get_input_details()[0], it2.get_output_details()
    s, z = i["quantization"]
    cq = []
    for x in calib:
        it2.set_tensor(i["index"], np.clip(np.round(((x - MEAN) / STD) / s + z), -128, 127).astype(np.int8)[None])
        it2.invoke()
        cq.append(it2.get_tensor(outs[1]["index"]).reshape(-1).copy())
    cq = np.array(cq, np.int8)
    np.save(out / "calib_features_q.npy", cq)

    # self-test: write a perturbed head into the file and compare TFLite with the emulation
    rng = np.random.default_rng(0)
    W2 = W * (1 + 0.2 * rng.standard_normal(W.shape)).astype(np.float32)
    q = npu_head.quantize_head(W2, b + 0.1, pm, cq)
    l_p, f_p = run_crops(npu_head.patch(buf, pm, q), probe)
    assert np.array_equal(f_p, f_new), "patching changed the features"
    agree_p = float((npu_head.emulate(f_new, q, pm) == l_p).mean())
    within1 = float((np.abs(npu_head.emulate(f_new, q, pm).astype(int) - l_p.astype(int)) <= 1).mean())
    print(f"patched head: emulation equals TFLite on {100 * agree_p:.2f}% of logits, within 1 LSB on {100 * within1:.2f}%")
    assert within1 == 1.0, "patched model does not match the emulation"

    t0 = time.time()
    _, fq = run_crops(dst, [ROOT / p for p in df.crop_path])
    print(f"features {fq.shape} in {time.time() - t0:.0f} s", flush=True)
    tag = "qat_int8" if not args.limit else f"qat_int8_limit{args.limit}"
    np.save(ROOT / f"data/features/{tag}_q.npy", fq.astype(np.int8))
    df[["crop_id", "label", "split", "dataset", "location_id", "seq_key", "verified"]].to_parquet(
        ROOT / f"data/features/{tag}_meta.parquet")
    print("wrote", dst, out / "patch_map.json")


if __name__ == "__main__":
    main()
