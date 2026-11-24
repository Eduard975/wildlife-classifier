from pathlib import Path

import numpy as np
import pytest

from rafl.edge import requant

ROOT = Path(__file__).resolve().parents[1]
CLS_MAP = ROOT / "runs/requant/cls_quant_map.json"
CLS_TMPL = ROOT / "runs/qat/efficientnet_lite0/efficientnet_lite0_int8_per_tensor.tflite"
CLS_FP32 = ROOT / "runs/qat/efficientnet_lite0/efficientnet_lite0_fp32.tflite"


def test_act_params_include_zero_and_span_range():
    s, z = requant.act_params(0.5, 6.0)  # range is widened to include 0
    assert z == -128 and s == pytest.approx(6.0 / 255, rel=1e-6)
    s, z = requant.act_params(-1.0, 1.0)
    assert (QLO := (-128 - z) * s) == pytest.approx(-1.0, abs=s) and (127 - z) * s == pytest.approx(1.0, abs=s)
    assert QLO <= 0


needs_files = pytest.mark.skipif(not (CLS_MAP.exists() and CLS_TMPL.exists() and CLS_FP32.exists()),
                                 reason="needs runs/requant and the QAT classifier files")


@needs_files
def test_template_ranges_reproduce_template():
    qm = requant.load_map(CLS_MAP)
    tmpl = CLS_TMPL.read_bytes()
    p = requant.read_params(tmpl, qm)
    # ranges that map exactly onto the template's own activation parameters
    ranges = {t: ((-128 - p[t][1]) * p[t][0], (127 - p[t][1]) * p[t][0]) for t, e in qm["tensors"].items() if e["kind"] == "act"}
    new = requant.write(tmpl, qm, requant.quantize(qm, tmpl, ranges))
    q = requant.read_params(new, qm)
    for t, e in qm["tensors"].items():
        if e["kind"] == "act":
            assert q[t][1] == p[t][1]
            assert q[t][0] == pytest.approx(p[t][0], rel=1e-5)
    assert len(new) == len(tmpl)


@needs_files
def test_fp32_weights_quantize_to_template_weights():
    qm = requant.load_map(CLS_MAP)
    tmpl = CLS_TMPL.read_bytes()
    w = requant.float_weights(CLS_FP32.read_bytes(), qm)
    q = requant.quantize(qm, tmpl, {}, w)
    for op in qm["weighted"]:
        a = requant._data(tmpl, qm["tensors"][op["w"]]["data"], np.int8)
        assert np.array_equal(a, np.asarray(q["data"][op["w"]]).reshape(-1)), op["key"]


BLOCK_MAP = ROOT / "runs/fl_block/quant_map.json"
BLOCK_TMPL = ROOT / "runs/fl_block/classifier_block.tflite"
BLOCK_INIT = ROOT / "runs/fl_hetero/init_block.npz"


@pytest.mark.skipif(not (BLOCK_MAP.exists() and BLOCK_TMPL.exists() and BLOCK_INIT.exists()), reason="needs runs/fl_block")
def test_block_requant_changes_only_the_tail():
    from rafl.fl import block, npu_block
    qm = requant.load_map(BLOCK_MAP)
    tmpl = BLOCK_TMPL.read_bytes()
    d = np.load(BLOCK_INIT)
    p = {k: d[k].astype(np.float32) * 1.1 for k in block.NAMES}
    rng = np.random.default_rng(0)
    x = npu_block.dequant(rng.integers(-128, 128, (16, 49, 320)), npu_block.block_input_params(tmpl, qm))
    new = npu_block.deployable(p, x, qm, tmpl)
    conv, fc = npu_block.tail(qm)
    tail = {conv["w"], conv["b"], conv["y"], fc["w"], fc["b"], fc["x"], fc["y"]}
    old_p, new_p = requant.read_params(tmpl, qm), requant.read_params(new, qm)
    changed = {t for t in qm["tensors"] if old_p[t] != new_p[t]}
    assert changed and changed <= tail
    assert len(new) == len(tmpl)
