"""Check rafl.fl.npu_block on the Mac (tensorflow env): write a block and head into the int8 classifier,
calibrated on one camera location's crops only, and compare int8 accuracy with the float block.

  .venv-export/bin/python scripts/check_npu_block.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rafl.edge import requant  # noqa: E402
from rafl.fl import block, npu_block  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def int8_preds(content, paths):
    it = tf.lite.Interpreter(model_content=content, num_threads=8)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    out = []
    for p in paths:
        im = Image.open(ROOT / p).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        it.set_tensor(i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8)[None])
        it.invoke()
        out.append(it.get_tensor(o["index"]).reshape(-1).argmax())
    return np.array(out)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    qm = requant.load_map(ROOT / "runs/fl_block/quant_map.json")
    tmpl = (ROOT / "runs/fl_block/classifier_block.tflite").read_bytes()
    xq = np.load(ROOT / "data/features/qat_block_q.npy", mmap_mode="r")
    meta = pd.read_parquet(ROOT / "data/features/qat_block_meta.parquet").reset_index(drop=True)
    man = pd.read_parquet(ROOT / cfg["paths"]["manifest"]).set_index("crop_id")
    xp = npu_block.block_input_params(tmpl, qm)
    y = meta.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    cal = np.flatnonzero(((meta.location_id == "NACTI : lebec_CA-42") & (meta.split == "train")).to_numpy())
    te = np.flatnonzero(((meta.split == "test") & (meta.verified != "n")).to_numpy())
    paths = man.loc[meta.crop_id.iloc[te], "crop_path"].tolist()
    x_cal = npu_block.dequant(xq[cal], xp)
    x_te = npu_block.dequant(xq[te], xp)
    d = np.load(ROOT / "runs/fl_hetero/init_block.npz")
    p0 = {k: d[k].astype(np.float32) for k in block.NAMES}
    rng = np.random.default_rng(0)
    p1 = {k: (v * (1 + 0.05 * rng.standard_normal(v.shape))).astype(np.float32) for k, v in p0.items()}
    res = {"calib_crops": len(cal), "test_crops": len(te)}
    for name, p in (("deployed", p0), ("perturbed", p1)):
        new = npu_block.deployable(p, x_cal, qm, tmpl)
        pf = block.logits(p, x_te).argmax(1)
        pi = int8_preds(new, paths)
        bacc = lambda pr: float(np.mean([(pr[y[te] == i] == i).mean() for i in range(len(classes))]))  # noqa: E731
        old_p, new_p = requant.read_params(tmpl, qm), requant.read_params(new, qm)
        res[name] = {"float_bacc": bacc(pf), "int8_bacc": bacc(pi), "int8_vs_float_agree": float((pf == pi).mean()),
                     "tensors_changed": sum(old_p[t] != new_p[t] for t in qm["tensors"]),
                     "bytes_changed": sum(a != b for a, b in zip(new, tmpl))}
        print(name, json.dumps(res[name]), flush=True)
    res["template_int8_bacc"] = float(np.mean([(int8_preds(tmpl, paths)[y[te] == i] == i).mean() for i in range(len(classes))]))
    (ROOT / "runs/fl_block/check_npu_block.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
