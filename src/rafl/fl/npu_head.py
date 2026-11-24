"""Write a federated head into the deployed int8 TFLite classifier. numpy only, runs on the board.

The classifier ends in MEAN -> FULLY_CONNECTED (per-tensor int8). `patch_map.json`, written by
scripts/build_fl_npu_model.py, holds the byte offsets of the FC weight and bias data and of the
scale / zero-point fields of the weight, bias and output tensors. A new head is quantized the
way the TFLite converter does it (symmetric int8 weights, int32 bias at feat_scale * w_scale,
asymmetric int8 output range from calibration features) and written into a copy of the file.
The NPU graph must then be recompiled (use a new graph-cache directory).

Head layout follows rafl.fl.sim: W has shape (dim, k), logits = f @ W + b, where f is the
dequantized pooled feature vector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_map(path) -> dict:
    return json.loads(Path(path).read_text())


def _view(buf, off, dtype):
    start, n = off
    return np.frombuffer(buf, dtype=dtype, count=n, offset=start)


def head_q(buf, pm):
    """Raw int8 weights (k, dim) and int32 bias from the file."""
    o = pm["offsets"]
    k, dim = pm["weight"]["shape"]
    W = _view(buf, o["weight"]["data"], np.int8).reshape(k, dim)
    b = _view(buf, o["bias"]["data"], np.uint8).view("<i4")
    return W, b


def scales(buf, pm):
    o = pm["offsets"]
    return {n: (float(_view(buf, o[n]["scale"], "<f4")[0]), int(_view(buf, o[n]["zero_point"], "<i8")[0]))
            for n in ("feat", "weight", "bias", "out")}


def read_head(buf, pm):
    """Float head (W (dim, k), b (k,)) stored in the file."""
    Wq, bq = head_q(buf, pm)
    s = scales(buf, pm)
    return (Wq.astype(np.float32).T * s["weight"][0]).astype(np.float32), (bq.astype(np.float32) * s["bias"][0])


def dequant_features(fq, pm):
    s, z = pm["feat"]["scale"], pm["feat"]["zero_point"]
    return ((np.asarray(fq, np.float32) - z) * s).astype(np.float32)


def quantize_head(W, b, pm, calib_fq):
    """int8 weights (k, dim), int32 bias and the scales, as the converter would produce them."""
    s_x = pm["feat"]["scale"]
    w_scale = float(max(np.abs(W).max(), 1e-8) / 127.0)
    Wq = np.clip(np.round(W.T / w_scale), -127, 127).astype(np.int8)
    b_scale = s_x * w_scale
    bq = np.round(b / b_scale).astype(np.int64).clip(-2**31, 2**31 - 1).astype("<i4")
    # output range from the exact integer accumulators, so every machine writes identical bytes
    acc = (np.asarray(calib_fq, np.int64) - pm["feat"]["zero_point"]) @ Wq.astype(np.int64).T + bq.astype(np.int64)
    lo = min(int(acc.min()), 0) * np.float64(np.float32(b_scale))
    hi = max(int(acc.max()), 0) * np.float64(np.float32(b_scale))
    o_scale = float(np.float32((hi - lo) / 255.0))
    o_zp = int(np.clip(np.round(-128 - lo / o_scale), -128, 127))
    return {"Wq": Wq, "bq": bq, "w_scale": float(np.float32(w_scale)), "b_scale": float(np.float32(b_scale)),
            "o_scale": o_scale, "o_zp": o_zp}


def patch(buf: bytes, pm, q) -> bytes:
    """Copy of the model bytes with the quantized head `q` written in place."""
    out = bytearray(buf)
    o = pm["offsets"]

    def put(off, arr, elem_size):
        a = np.ascontiguousarray(arr).tobytes()
        assert len(a) == off[1] * elem_size, "size mismatch"
        out[off[0]:off[0] + len(a)] = a

    put(o["weight"]["data"], q["Wq"].reshape(-1), 1)
    put(o["bias"]["data"], q["bq"].view(np.uint8), 1)
    put(o["weight"]["scale"], np.array([q["w_scale"]], "<f4"), 4)
    put(o["bias"]["scale"], np.array([q["b_scale"]], "<f4"), 4)
    put(o["out"]["scale"], np.array([q["o_scale"]], "<f4"), 4)
    put(o["out"]["zero_point"], np.array([q["o_zp"]], "<i8"), 8)
    return bytes(out)


def int8_logits(fq, Wq, bq, pm, o_scale=None, o_zp=None, s=None):
    """Integer FC as TFLite computes it (float multiplier instead of the fixed-point one)."""
    s_x, z_x = pm["feat"]["scale"], pm["feat"]["zero_point"]
    w_scale = s["weight"][0] if s else pm["weight"]["scale"]
    o_scale = o_scale if o_scale is not None else (s["out"][0] if s else pm["out"]["scale"])
    o_zp = o_zp if o_zp is not None else (s["out"][1] if s else pm["out"]["zero_point"])
    acc = (np.asarray(fq, np.int32) - z_x) @ Wq.astype(np.int32).T + bq.astype(np.int64)
    y = np.floor(acc * (s_x * w_scale / o_scale) + 0.5) + o_zp
    return np.clip(y, -128, 127).astype(np.int8)


def emulate(fq, q, pm):
    """int8 logits of a quantized head `q` on int8 features."""
    s_x, z_x = pm["feat"]["scale"], pm["feat"]["zero_point"]
    acc = (np.asarray(fq, np.int32) - z_x) @ q["Wq"].astype(np.int32).T + q["bq"].astype(np.int64)
    y = np.floor(acc * (s_x * q["w_scale"] / q["o_scale"]) + 0.5) + q["o_zp"]
    return np.clip(y, -128, 127).astype(np.int8)


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]
