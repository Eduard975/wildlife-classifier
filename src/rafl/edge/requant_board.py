"""Re-quantize a model on the board from local images, then check the new file on the NPU.

Needs only numpy, PIL and tflite_runtime. Calibrates the FP32 model on the CPU (all tensors kept),
writes new int8 parameters into the per-tensor template with rafl.edge.requant, compiles the result
on the NPU, and compares NPU outputs with the CPU int8 run of the same file and with the template.

  python3 requant_board.py --model det --dir ~/rafl/requant --images ~/rafl/requant/calib_det --n 200
  python3 requant_board.py --model cls --dir ~/rafl/requant --calib-npy ~/rafl/requant/cls_calib_u8.npy
Writes <dir>/<model>_requant_board.tflite and <dir>/<model>_requant_board.json.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rafl.edge import requant  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = {"cls": 224, "det": 416}


def letterbox(im, size):
    s = size / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, (0, 0))
    return np.asarray(canvas, np.float32) / 255.0


def prep(model, x):
    """Model-ready float input from an HWC [0, 1] image."""
    return ((x - MEAN) / STD)[None].astype(np.float32) if model == "cls" else x[None].astype(np.float32)


def load_images(model, args):
    if args.calib_npy:
        a = np.load(args.calib_npy, mmap_mode="r")[: args.n]
        return [a[i].astype(np.float32) / 255.0 for i in range(len(a))]
    paths = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        out.append(letterbox(im, SIZE[model]) if model == "det" else
                   np.asarray(im.resize((224, 224), Image.BICUBIC), np.float32) / 255.0)
    return out


def interp(content, delegate=None, cache=None):
    from tflite_runtime.interpreter import Interpreter, load_delegate
    if delegate:
        os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
        os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = str(cache)
        it = Interpreter(model_content=content, experimental_delegates=[load_delegate(delegate)])
    else:
        it = Interpreter(model_content=content, num_threads=4)
    it.allocate_tensors()
    return it


def run(it, xs):
    i = it.get_input_details()[0]
    s, z = i["quantization"]
    res, t = [], []
    for x in xs:
        it.set_tensor(i["index"], np.clip(np.round(x / s + z), -128, 127).astype(np.int8))
        t0 = time.perf_counter()
        it.invoke()
        t.append(time.perf_counter() - t0)
        res.append([it.get_tensor(o["index"]).copy() for o in it.get_output_details()])
    return res, float(np.median(t[1:]) * 1000) if len(t) > 1 else float(t[0] * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=SIZE, required=True)
    ap.add_argument("--dir", required=True, help="holds <model>_fp32.tflite, <model>_i8pt.tflite, <model>_quant_map.json")
    ap.add_argument("--images")
    ap.add_argument("--calib-npy")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--check", type=int, default=30, help="inputs for the NPU check (taken from the calibration set)")
    ap.add_argument("--weights", action="store_true", help="also re-quantize all weights from the FP32 file")
    ap.add_argument("--delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--tag", default="board")
    args = ap.parse_args()
    d = Path(args.dir).expanduser()
    f32, tmpl_p, qm_p = d / f"{args.model}_fp32.tflite", d / f"{args.model}_i8pt.tflite", d / f"{args.model}_quant_map.json"
    qm = requant.load_map(qm_p)
    tmpl = tmpl_p.read_bytes()

    t0 = time.time()
    imgs = load_images(args.model, args)
    xs = [prep(args.model, x) for x in imgs]
    t_load = time.time() - t0
    t0 = time.time()
    cal = requant.calibrate(f32, qm, xs)
    t_cal = time.time() - t0
    t0 = time.time()
    w = requant.float_weights(f32.read_bytes(), qm) if args.weights else None
    q = requant.quantize(qm, tmpl, cal["ranges"], w)
    new = requant.write(tmpl, qm, q)
    t_q = time.time() - t0
    out = d / f"{args.model}_requant_{args.tag}.tflite"
    out.write_bytes(new)

    chk = xs[: args.check]
    cache = d / f"vxcache_{args.model}_{args.tag}_{int(time.time())}"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    npu = interp(new, args.delegate, cache)
    y_npu, _ = run(npu, chk[:1])
    t_compile = time.time() - t0
    y_npu, ms_npu = run(npu, chk)
    y_cpu, ms_cpu = run(interp(new), chk)
    y_old, _ = run(interp(tmpl), chk)
    cache_t = d / f"vxcache_{args.model}_template"
    cache_t.mkdir(parents=True, exist_ok=True)
    y_old_npu, ms_old_npu = run(interp(tmpl, args.delegate, cache_t), chk)

    def cmp(ya_all, yb_all):
        diffs = [np.abs(a.astype(int) - b.astype(int)) for ya, yb in zip(ya_all, yb_all) for a, b in zip(ya, yb)]
        return int(max(x.max() for x in diffs)), float(np.mean([np.mean(x <= 1) for x in diffs]))

    lsb, within1 = cmp(y_npu, y_cpu)
    lsb_t, within1_t = cmp(y_old_npu, y_old)
    old_p, new_p = requant.read_params(tmpl, qm), requant.read_params(new, qm)
    changed = sum(1 for t in qm["tensors"] if old_p[t] != new_p[t])
    res = {"model": args.model, "tag": args.tag, "n_calib": cal["n"], "weights_requantized": bool(args.weights),
           "load_s": round(t_load, 1), "calibrate_s": round(t_cal, 1), "quantize_write_s": round(t_q, 2),
           "npu_compile_s": round(t_compile, 1), "npu_ms": round(ms_npu, 2), "cpu_int8_ms": round(ms_cpu, 1),
           "npu_vs_cpu_max_lsb": lsb, "npu_vs_cpu_within1": round(within1, 5),
           "template_npu_ms": round(ms_old_npu, 2),
           "template_npu_vs_cpu_max_lsb": lsb_t, "template_npu_vs_cpu_within1": round(within1_t, 5),
           "tensors_changed": changed, "tensors": len(qm["tensors"]),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "digest": requant.digest(new), "template_digest": requant.digest(tmpl)}
    if args.model == "cls":
        res["top1_agree_with_template"] = float(np.mean([a[0].argmax() == b[0].argmax() for a, b in zip(y_cpu, y_old)]))
    (d / f"{args.model}_requant_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
