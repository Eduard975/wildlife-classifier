"""Write a federated block (1x1 conv) and head into the int8 classifier on the board. numpy only.

The classifier ends in [7x7x320 int8] -> CONV_2D 1x1 + ReLU6 -> MEAN -> FULLY_CONNECTED. The template
(runs/fl_hetero/classifier_block_out.tflite) also outputs the 7x7x320 block input, which does not
change when the block does, so the board computes it once per crop on the NPU.

`deployable` re-quantizes the tail the way the converter would, from the board's own data: it runs the
FP32 block and head in numpy on the block inputs of the board's local crops, takes the ranges of the
conv output, the pooled features and the logits, and writes int8 weights, int32 biases and the new
scales with rafl.edge.requant. Tensors before the block keep the template's bytes.

Parameters follow rafl.fl.block: Wc (320, 1280), bc (1280,), W (1280, k), b (k,).
"""
from __future__ import annotations

import numpy as np

from rafl.edge import requant


def tail(qm):
    conv, fc = qm["weighted"][-2], qm["weighted"][-1]
    assert conv["key"].endswith("CONV_2D") and fc["key"].endswith("FULLY_CONNECTED"), "unexpected graph tail"
    assert fc["x"] != conv["y"], "expected MEAN between the conv and the FC"
    return conv, fc


def block_input_params(template, qm):
    conv, _ = tail(qm)
    return requant.read_params(template, qm)[conv["x"]]


def read_block(template, qm) -> dict:
    """Float block and head stored in an int8 file (Wc (320, 1280), bc, W (1280, k), b)."""
    conv, fc = tail(qm)
    pr = requant.read_params(template, qm)
    t = qm["tensors"]

    def w(i, shape):
        return requant._data(template, t[i]["data"], np.int8).astype(np.float32).reshape(shape) * pr[i][0]

    def b(i):
        return requant._data(template, t[i]["data"], "<i4").astype(np.float32) * pr[i][0]
    Wc = w(conv["w"], conv["w_shape"]).reshape(conv["w_shape"][0], -1).T
    W = w(fc["w"], fc["w_shape"]).T
    return {"Wc": np.ascontiguousarray(Wc), "bc": b(conv["b"]), "W": np.ascontiguousarray(W), "b": b(fc["b"])}


def dequant(xq, params):
    s, z = params
    return (np.asarray(xq, np.float32) - z) * s


def tail_ranges(p, x, chunk=128):
    """Ranges of conv output (after ReLU6), pooled features and logits over block inputs x (N, 49, 320)."""
    lo = {"z": np.inf, "f": np.inf, "y": np.inf}
    hi = {"z": -np.inf, "f": -np.inf, "y": -np.inf}
    for s in range(0, len(x), chunk):
        z = np.clip(x[s:s + chunk] @ p["Wc"] + p["bc"], 0.0, 6.0)
        f = z.mean(1)
        y = f @ p["W"] + p["b"]
        for k, v in (("z", z), ("f", f), ("y", y)):
            lo[k], hi[k] = min(lo[k], float(v.min())), max(hi[k], float(v.max()))
    return {k: (lo[k], hi[k]) for k in lo}


def deployable(p: dict, x_cal: np.ndarray, qm: dict, template: bytes) -> bytes:
    """int8 model bytes with block and head `p`, calibrated on block inputs `x_cal` (float, (N, 49, 320))."""
    conv, fc = tail(qm)
    r = tail_ranges(p, x_cal)
    ranges = {conv["y"]: r["z"], fc["x"]: r["f"], fc["y"]: r["y"]}
    Wc = np.asarray(p["Wc"], np.float32)
    weights = {conv["key"]: (Wc.T.reshape(conv["w_shape"]), np.asarray(p["bc"], np.float32)),
               fc["key"]: (np.asarray(p["W"], np.float32).T.reshape(fc["w_shape"]), np.asarray(p["b"], np.float32))}
    return requant.write(template, qm, requant.quantize(qm, template, ranges, weights))
