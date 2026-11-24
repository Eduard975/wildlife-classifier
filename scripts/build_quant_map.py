"""Write quant_map.json for the on-board quantizer (rafl.edge.requant). tensorflow env, run once on the Mac.

Records, for an int8 per-tensor template and its FP32 twin:
  - byte offsets of every quantized tensor's scale, zero point and (for constants) data;
  - the FP32 tensor with the same name, for calibration on the board;
  - ops with weights (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED) with the offsets of their FP32
    weight and bias data, matched by op position (both files have the same op sequence once
    QUANTIZE / DEQUANTIZE are removed);
  - tie groups and fixed outputs required by the int8 kernels.

  .venv-export/bin/python scripts/build_quant_map.py --int8 <int8_per_tensor.tflite> --fp32 <fp32.tflite> --out <quant_map.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

TIE_FIRST = {"PAD", "RESHAPE", "STRIDED_SLICE", "TRANSPOSE", "MAX_POOL_2D", "RESIZE_NEAREST_NEIGHBOR"}
TIE_ALL = {"CONCATENATION"}
FIXED = {"LOGISTIC": (1.0 / 256, -128), "SOFTMAX": (1.0 / 256, -128)}
WEIGHTED = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}
SKIP = {"QUANTIZE", "DEQUANTIZE"}
ACT = {0: None, 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6", 4: "TANH", 5: "SIGN_BIT"}
OPNAME = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}


def _vec(tab, slot):
    o = tab.Offset(slot)
    return [tab.Vector(o), tab.VectorLen(o)] if o else None


def read(path):
    buf = Path(path).read_bytes()
    m = schema.Model.GetRootAs(buf, 0)
    sg = m.Subgraphs(0)
    codes = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        codes.append(OPNAME[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())])
    tensors = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        b = m.Buffers(t.Buffer())
        q = t.Quantization()
        tensors.append({
            "name": t.Name().decode(), "shape": t.ShapeAsNumpy().tolist() if t.ShapeLength() else [],
            "type": t.Type(), "data": _vec(b._tab, 4),
            "scale": _vec(q._tab, 8) if q else None, "zp": _vec(q._tab, 10) if q else None,
        })
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        name = codes[op.OpcodeIndex()]
        fused = None
        opt = op.BuiltinOptions()
        if opt is not None and name in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "ADD", "SUB", "MUL",
                                         "CONCATENATION", "MAX_POOL_2D", "AVERAGE_POOL_2D"):
            cls = {"CONV_2D": schema.Conv2DOptions, "DEPTHWISE_CONV_2D": schema.DepthwiseConv2DOptions,
                   "FULLY_CONNECTED": schema.FullyConnectedOptions, "ADD": schema.AddOptions,
                   "SUB": schema.SubOptions, "MUL": schema.MulOptions,
                   "CONCATENATION": schema.ConcatenationOptions, "MAX_POOL_2D": schema.Pool2DOptions,
                   "AVERAGE_POOL_2D": schema.Pool2DOptions}[name]()
            cls.Init(opt.Bytes, opt.Pos)
            fused = ACT.get(cls.FusedActivationFunction())
        ops.append({"op": name, "in": op.InputsAsNumpy().tolist(), "out": op.OutputsAsNumpy().tolist(), "fused": fused})
    return buf, tensors, ops, sg.InputsAsNumpy().tolist(), sg.OutputsAsNumpy().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _, T, ops, ins, outs = read(args.int8)
    _, F, fops, _, _ = read(args.fp32)
    fname = {t["name"]: i for i, t in enumerate(F)}

    qt = {i for i, t in enumerate(T) if t["scale"] and t["scale"][1] == 1}
    multi = [i for i, t in enumerate(T) if t["scale"] and t["scale"][1] > 1]
    assert not multi, f"per-channel tensors present ({len(multi)}): use a per-tensor template"

    produced = {o for op in ops for o in op["out"]}
    kind = {}
    for i in qt:
        t = T[i]
        kind[i] = "input" if i in ins else ("act" if i in produced else "const")
    fused_of = {}
    for op in ops:
        for o in op["out"]:
            fused_of[o] = op["fused"]

    a_ops = [op for op in ops if op["op"] not in SKIP]
    b_ops = [op for op in fops if op["op"] not in SKIP]
    assert [o["op"] for o in a_ops] == [o["op"] for o in b_ops], "op sequences differ"
    weighted = []
    for k, (a, b) in enumerate(zip(a_ops, b_ops)):
        if a["op"] not in WEIGHTED:
            continue
        x, w = a["in"][0], a["in"][1]
        bias = a["in"][2] if len(a["in"]) > 2 and a["in"][2] >= 0 else None
        fw, fb = b["in"][1], (b["in"][2] if len(b["in"]) > 2 and b["in"][2] >= 0 else None)
        assert T[w]["shape"] == F[fw]["shape"], f"weight shape differs at op {k}"
        kind[w] = "weight"
        if bias is not None:
            kind[bias] = "bias"
        weighted.append({"key": f"{k}:{a['op']}", "x": x, "w": w, "b": bias, "y": a["out"][0],
                         "w_shape": F[fw]["shape"], "fp32_w": F[fw]["data"], "fp32_b": F[fb]["data"] if fb is not None else None})

    ties, fixed = [], {}
    for op in ops:
        acts = [i for i in op["in"] if i in kind and kind[i] in ("act", "input")]
        if op["op"] in TIE_FIRST and acts and op["in"][0] in acts:
            ties.append([op["in"][0], op["out"][0]])
        elif op["op"] in TIE_ALL:
            ties.append(acts + op["out"])
        if op["op"] in FIXED:
            fixed[op["out"][0]] = FIXED[op["op"]]

    tensors = {}
    for i in sorted(qt):
        t = T[i]
        fi = fname.get(t["name"]) if kind[i] in ("act", "input") else None
        tensors[i] = {"name": t["name"], "shape": t["shape"], "kind": kind[i], "scale": t["scale"], "zp": t["zp"],
                      "data": t["data"], "fp32": fi, "fused": fused_of.get(i)}
    qm = {"int8": str(args.int8), "fp32": str(args.fp32), "inputs": ins, "outputs": outs, "tensors": tensors,
          "weighted": weighted, "ties": ties, "fixed": fixed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(qm))
    n = {k: sum(1 for t in tensors.values() if t["kind"] == k) for k in ("input", "act", "weight", "bias", "const")}
    unmapped = [t["name"] for t in tensors.values() if t["kind"] == "act" and t["fp32"] is None]
    print(f"{args.out}: {n}, weighted ops {len(weighted)}, tie groups {len(ties)}, fixed {len(fixed)}, "
          f"activations without FP32 twin {len(unmapped)} {unmapped[:3]}")


if __name__ == "__main__":
    main()
