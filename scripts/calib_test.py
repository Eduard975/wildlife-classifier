"""Does calibrating int8 on a camera's own frames beat generic calibration? (D2 hypothesis test, tensorflow env)

For each test-split camera location, the FP32 model is calibrated on frames from that location only
(no labels needed), re-quantized with rafl.edge.requant into the per-tensor template, and evaluated on
that location's held-out frames. The baseline is the same quantizer calibrated on the generic set the
converter used, so calibration data is the only difference. Results are pooled over locations.

Detector: calibration images are other frames of the same location (sources of its crops, downloaded
from LILA, not in the eval set); evaluation as in eval_detector.py (MegaDetector v5 reference boxes).
Classifier: each location's test sequences are split in half; one half calibrates, the other evaluates.

  .venv-export/bin/python scripts/calib_test.py --model det --min-eval 4 --calib-per-loc 24
  .venv-export/bin/python scripts/calib_test.py --model cls
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
import eval_detector as ed  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DET = ROOT / "runs/qat_detector/MDV6-yolov10-c_416"
CLS = ROOT / "runs/qat/efficientnet_lite0"


def fetch(url, dst: Path):
    if dst.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            im = Image.open(io.BytesIO(r.read())).convert("RGB")
        im.thumbnail((1280, 1280))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=90)
        return True
    except Exception as e:  # noqa: BLE001
        print("fetch failed", url, e, flush=True)
        return False


def requantize(f32, qm, tmpl, xs):
    cal = requant.calibrate(f32, qm, xs, num_threads=8)
    return requant.write(tmpl, qm, requant.quantize(qm, tmpl, cal["ranges"])), cal["ranges"]


def det_eval(content, paths, ref_by, names, thr=0.25):
    it = tf.lite.Interpreter(model_content=content, num_threads=8) if isinstance(content, bytes) else \
        tf.lite.Interpreter(str(content), num_threads=8)
    it.allocate_tensors()
    rows = []
    for fp, label in paths:
        xyxy, conf, _ = ed.detect(it, ROOT / "data/frames" / fp, 416, names, thr)
        r = ref_by.get(fp)
        hit = False
        if r is not None and len(xyxy):
            r = np.c_[r[:, :2], r[:, :2] + r[:, 2:]]
            hit = bool(max(ed.iou_one(b, r).max() for b in xyxy) >= 0.5)
        rows.append({"frame": fp, "label": label, "has_ref": r is not None, "hit": hit, "any": bool(len(xyxy))})
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    pos, neg = df[df.has_ref], df[~df.has_ref]
    rec = pos.groupby("label").hit.mean()
    return {"frames": len(df), "positives": len(pos), "hits": int(pos.hit.sum()), "recall_pooled": float(pos.hit.mean()),
            "recall_mean_over_classes": float(rec.mean()), "recall_by_class": rec.round(3).to_dict(),
            "empty_frames": len(neg), "empty_detect_rate": float(neg["any"].mean()) if len(neg) else None}


def run_det(args):
    names = (DET / "names.txt").read_text().split("\n")
    qm = requant.load_map(ROOT / "runs/requant/det_quant_map.json")
    tmpl = (DET / "int8_per_tensor.tflite").read_bytes()
    f32 = DET / "fp32.tflite"
    frames = pd.read_parquet(ROOT / "data/manifests/frames.parquet")
    ev = frames[frames.role == "eval"]
    ref = pd.read_parquet(ROOT / "data/manifests/frame_boxes.parquet")
    ref_by = {fp: g[["x", "y", "w", "h"]].to_numpy() for fp, g in ref.groupby("frame_path")}
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    crops = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    counts = ev.location_id.value_counts()
    locs = sorted(counts[counts >= args.min_eval].index)
    eval_ids = set(ev.image_id)
    rng = np.random.default_rng(0)
    plan = {}
    for loc in locs:
        src = crops[(crops.location_id == loc) & ~crops.image_id.isin(eval_ids)].drop_duplicates("image_id")
        if len(src) < args.min_calib:
            continue
        src = src.iloc[rng.permutation(len(src))[: args.calib_per_loc]]
        plan[loc] = [(u, ROOT / "data/frames/loc_calib" / (hashlib.sha1(u.encode()).hexdigest()[:16] + ".jpg")) for u in src.url]
    todo = [(u, p) for v in plan.values() for u, p in v]
    print(f"{len(plan)} locations, {len(todo)} calibration frames to fetch or reuse", flush=True)
    with ThreadPoolExecutor(16) as ex:
        ok = dict(zip([p for _, p in todo], ex.map(lambda a: fetch(*a), todo)))

    generic = [letterbox(Image.open(p).convert("RGB"), 416)[0][None] for p in sorted((ROOT / "data/frames/calib").glob("*.jpg"))[:200]]
    gen_file, gen_ranges = requantize(f32, qm, tmpl, generic)
    res = {"fp32": [], "converter": [], "generic": [], "local": [], "generic_plus_local": []}
    per_loc = {}
    for loc, items in plan.items():
        xs = [letterbox(Image.open(p).convert("RGB"), 416)[0][None] for _, p in items if ok.get(p)]
        loc_file, loc_ranges = requantize(f32, qm, tmpl, xs)
        both = {t: (min(gen_ranges[t][0], loc_ranges[t][0]), max(gen_ranges[t][1], loc_ranges[t][1])) for t in gen_ranges}
        both_file = requant.write(tmpl, qm, requant.quantize(qm, tmpl, both))
        paths = list(ev[ev.location_id == loc][["frame_path", "label"]].itertuples(index=False, name=None))
        per_loc[loc] = {"calib_frames": len(xs), "eval_frames": len(paths)}
        for k, c in (("fp32", f32), ("converter", tmpl), ("generic", gen_file), ("local", loc_file), ("generic_plus_local", both_file)):
            rows = det_eval(c, paths, ref_by, names)
            res[k] += [dict(r, location=loc) for r in rows]
            per_loc[loc][k] = int(sum(r["hit"] for r in rows))
        print(loc, per_loc[loc], flush=True)
    return {k: summarize(v) for k, v in res.items()}, per_loc, res


def run_cls(args):
    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    qm = requant.load_map(ROOT / "runs/requant/cls_quant_map.json")
    tmpl = (CLS / "efficientnet_lite0_int8_per_tensor.tflite").read_bytes()
    f32 = CLS / "efficientnet_lite0_fp32.tflite"
    m = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    t = m[m.ok & (m.split == "test") & m.label.isin(classes) & (m.verified != "n")]
    g = t.groupby("location_id").agg(n=("crop_id", "size"), seq=("seq_key", "nunique"))
    locs = sorted(g[(g.n >= args.min_crops) & (g.seq >= 20)].index)

    def load(paths):
        return [((np.asarray(Image.open(ROOT / p).convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255.0 - MEAN) / STD)[None]
                for p in paths]

    def predict(content, xs):
        it = tf.lite.Interpreter(model_content=content, num_threads=8) if isinstance(content, bytes) else \
            tf.lite.Interpreter(str(content), num_threads=8)
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        out = []
        for x in xs:
            if i["dtype"] == np.int8:
                s, z = i["quantization"]
                x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
            it.set_tensor(i["index"], x.astype(i["dtype"]))
            it.invoke()
            out.append(it.get_tensor(o["index"]).reshape(-1).astype(np.float32).argmax())
        return np.array(out)

    gc = (np.load(CLS / "calib.npy") - MEAN) / STD
    gen_file, _ = requantize(f32, qm, tmpl, [gc[i:i + 1].astype(np.float32) for i in range(len(gc))])
    rng = np.random.default_rng(0)
    preds = {k: [] for k in ("fp32", "converter", "generic", "local")}
    ys = []
    per_loc = {}
    for loc in locs:
        d = t[t.location_id == loc]
        seqs = d.seq_key.unique()
        rng.shuffle(seqs)
        cal_s = set(seqs[: len(seqs) // 2])
        dc, de = d[d.seq_key.isin(cal_s)], d[~d.seq_key.isin(cal_s)]
        loc_file, _ = requantize(f32, qm, tmpl, load(dc.crop_path))
        xe = load(de.crop_path)
        y = de.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
        ys.append(y)
        per_loc[loc] = {"calib_crops": len(dc), "eval_crops": len(de)}
        for k, c in (("fp32", f32), ("converter", tmpl), ("generic", gen_file), ("local", loc_file)):
            p = predict(c, xe)
            preds[k].append(p)
            per_loc[loc][k] = float((p == y).mean())
        print(loc, per_loc[loc], flush=True)
    y = np.concatenate(ys)
    summ = {}
    for k, v in preds.items():
        p = np.concatenate(v)
        rec = {c: float((p[y == i] == i).mean()) for i, c in enumerate(classes) if (y == i).any()}
        summ[k] = {"crops": len(y), "accuracy": float((p == y).mean()), "bacc": float(np.mean(list(rec.values()))), "recall": rec}
    return summ, per_loc, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("det", "cls"), required=True)
    ap.add_argument("--min-eval", type=int, default=4)
    ap.add_argument("--min-calib", type=int, default=12)
    ap.add_argument("--calib-per-loc", type=int, default=24)
    ap.add_argument("--min-crops", type=int, default=100)
    args = ap.parse_args()
    summ, per_loc, rows = (run_det if args.model == "det" else run_cls)(args)
    out = ROOT / "runs/requant/calib_test"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "summary": summ, "per_location": per_loc}, open(out / f"{args.model}.json", "w"), indent=1)
    if rows:
        pd.DataFrame([dict(r, variant=k) for k, v in rows.items() for r in v]).to_csv(out / f"{args.model}_frames.csv", index=False)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
