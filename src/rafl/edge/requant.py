"""On-board int8 quantizer for per-tensor TFLite models. numpy only (plus tflite_runtime for calibration).

The board holds three files per model: the FP32 .tflite, the int8 per-tensor template written once by
the converter, and `quant_map.json` (scripts/build_quant_map.py) with the byte offsets of every
quantized tensor's data, scale and zero point, the int8 -> FP32 tensor correspondence, the ops with
weights, and the parameter constraints of the int8 kernels:
  - tie groups: PAD, RESHAPE, STRIDED_SLICE, TRANSPOSE, CONCATENATION, MAX_POOL_2D and
    RESIZE_NEAREST_NEIGHBOR keep their input's scale and zero point on their output;
  - fixed outputs: LOGISTIC and SOFTMAX use scale 1/256, zero point -128.

`calibrate` runs the FP32 model on the CPU with every intermediate tensor kept and records per-tensor
min/max over the given inputs. `quantize` turns ranges (and optionally new float weights) into int8
parameters the way the TFLite converter does: asymmetric int8 activations with the range nudged to
include 0, symmetric per-tensor int8 weights in [-127, 127], int32 biases at input scale x weight
scale. `write` puts them into a copy of the template; the layout of the file never changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

QMIN, QMAX = -128, 127
FUSED_RANGE = {"RELU": (0.0, None), "RELU6": (0.0, 6.0), "RELU_N1_TO_1": (-1.0, 1.0)}


def load_map(path) -> dict:
    m = json.loads(Path(path).read_text())
    m["tensors"] = {int(k): v for k, v in m["tensors"].items()}
    return m


def _view(buf, off, dtype):
    """Quantization vectors: off[1] counts elements."""
    return np.frombuffer(buf, dtype=dtype, count=off[1], offset=off[0])


def _data(buf, off, dtype):
    """Buffer data: off[1] counts bytes."""
    return np.frombuffer(buf, dtype=np.uint8, count=off[1], offset=off[0]).view(dtype)


def read_params(buf, qm) -> dict:
    """Current (scale, zero point) of every quantized tensor in an int8 file."""
    return {i: (float(_view(buf, t["scale"], "<f4")[0]), int(_view(buf, t["zp"], "<i8")[0]))
            for i, t in qm["tensors"].items()}


def float_weights(fp32_buf, qm) -> dict:
    """{op key: (W, b)} float weights read from the FP32 file."""
    out = {}
    for w in qm["weighted"]:
        W = _data(fp32_buf, w["fp32_w"], "<f4").reshape(w["w_shape"]).copy()
        b = _data(fp32_buf, w["fp32_b"], "<f4").copy() if w["fp32_b"] else None
        out[w["key"]] = (W, b)
    return out


def calibrate(fp32_path, qm, inputs, num_threads=4) -> dict:
    """Per-tensor (min, max) of every mapped activation over `inputs` (iterable of model-ready arrays)."""
    try:
        from tflite_runtime.interpreter import Interpreter, OpResolverType
    except ImportError:  # Mac
        import tensorflow as tf
        Interpreter, OpResolverType = tf.lite.Interpreter, tf.lite.experimental.OpResolverType
    # without the default XNNPACK delegate: tensors inside a delegated partition are not written back
    it = Interpreter(str(fp32_path), num_threads=num_threads, experimental_preserve_all_tensors=True,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    inp = it.get_input_details()[0]["index"]
    acts = {i: t["fp32"] for i, t in qm["tensors"].items() if t["kind"] == "act" and t["fp32"] is not None}
    lo = {i: np.inf for i in acts}
    hi = {i: -np.inf for i in acts}
    n = 0
    for x in inputs:
        it.set_tensor(inp, np.asarray(x, np.float32))
        it.invoke()
        for i, fi in acts.items():
            a = it.get_tensor(fi)
            lo[i] = min(lo[i], float(a.min()))
            hi[i] = max(hi[i], float(a.max()))
        n += 1
    return {"n": n, "ranges": {i: (lo[i], hi[i]) for i in acts}}


def act_params(rmin, rmax):
    """Asymmetric int8 scale and zero point as the converter nudges them."""
    rmin, rmax = min(float(rmin), 0.0), max(float(rmax), 0.0)
    if rmax - rmin < 1e-12:
        return 1.0, 0
    scale = (rmax - rmin) / (QMAX - QMIN)
    zp = int(np.clip(np.round(QMIN - rmin / scale), QMIN, QMAX))
    return float(np.float32(scale)), zp


def _groups(qm):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for g in qm["ties"]:
        for t in g[1:]:
            parent[find(t)] = find(g[0])
    groups = {}
    for i, t in qm["tensors"].items():
        if t["kind"] in ("act", "input"):
            groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def quantize(qm, template, ranges, weights=None) -> dict:
    """New int8 parameters. `ranges`: {tensor: (min, max)} for activations (missing ones keep the
    template's parameters). `weights`: {op key: (W, b)} float weights (missing ones keep the template's
    int8 weights). Returns {"params": {tensor: (scale, zp)}, "data": {tensor: ndarray}}."""
    old = read_params(template, qm)
    params = {}
    fixed = {int(k): tuple(v) for k, v in qm["fixed"].items()}
    for g in _groups(qm):
        if any(t in fixed for t in g):
            p = next(fixed[t] for t in g if t in fixed)
        elif any(qm["tensors"][t]["kind"] == "input" for t in g):
            p = next(old[t] for t in g if qm["tensors"][t]["kind"] == "input")
        else:
            rs = [ranges[t] for t in g if t in ranges]
            if not rs:
                continue
            lo, hi = min(r[0] for r in rs), max(r[1] for r in rs)
            for t in g:
                fr = qm["tensors"][t].get("fused")
                if fr in FUSED_RANGE:
                    a, b = FUSED_RANGE[fr]
                    lo = max(lo, a)
                    hi = min(hi, b) if b is not None else hi
            p = act_params(lo, hi)
        for t in g:
            params[t] = p
    data = {}
    for w in qm["weighted"]:
        x, wi, bi = w["x"], w["w"], w["b"]
        new_w = weights is not None and w["key"] in weights
        if not new_w and params.get(x, old[x]) == old[x]:
            continue  # neither the weights nor the input scale change: keep the template's bytes
        s_x = params.get(x, old[x])[0]
        if new_w:
            W, b = weights[w["key"]]
            s_w = float(np.float32(max(np.abs(W).max(), 1e-8) / 127.0))
            data[wi] = np.clip(np.round(W / s_w), -127, 127).astype(np.int8)
            params[wi] = (s_w, 0)
        else:
            s_w = old[wi][0]
            b = None
            if bi is not None:  # keep the template's float bias
                b = _data(template, qm["tensors"][bi]["data"], "<i4").astype(np.float64) * old[bi][0]
        if bi is not None:
            if b is None:
                b = np.zeros(qm["tensors"][bi]["shape"][0])
            s_b = float(np.float32(s_x * s_w))
            data[bi] = np.clip(np.round(np.asarray(b, np.float64) / s_b), -2**31, 2**31 - 1).astype("<i4")
            params[bi] = (s_b, 0)
    return {"params": params, "data": data}


def write(template: bytes, qm, q) -> bytes:
    out = bytearray(template)

    def put(off, arr, nbytes):
        a = np.ascontiguousarray(arr).tobytes()
        assert len(a) == nbytes, "size mismatch"
        out[off[0]:off[0] + len(a)] = a

    for t, (s, z) in q["params"].items():
        e = qm["tensors"][t]
        put(e["scale"], np.array([s], "<f4"), 4 * e["scale"][1])
        put(e["zp"], np.array([z], "<i8"), 8 * e["zp"][1])
    for t, arr in q["data"].items():
        off = qm["tensors"][t]["data"]
        put(off, np.asarray(arr).reshape(-1), off[1])
    return bytes(out)


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]
