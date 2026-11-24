from pathlib import Path

import numpy as np
import pytest

from rafl.fl.quant import ErrorFeedback, decode, encode, pack_codes, unpack_codes

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_pack_roundtrip(bits):
    rng = np.random.default_rng(bits)
    q = rng.integers(0, 1 << bits, 6405).astype(np.uint8)
    b = pack_codes(q, bits)
    assert len(b) == -(-6405 * bits // 8)
    assert np.array_equal(unpack_codes(b, bits, 6405), q)


@pytest.mark.parametrize("bits", [2, 4, 8, 16, 32])
def test_encode_decode_matches_sender(bits):
    rng = np.random.default_rng(0)
    d = [rng.standard_normal((1280, 5)).astype(np.float32), rng.standard_normal(5).astype(np.float32)]
    payload, deq = encode(d, bits, rng)
    out, b = decode(payload)
    assert b == bits
    assert all(np.array_equal(a, c) for a, c in zip(out, deq))
    if bits == 4:
        assert len(payload) == 3 + (1 + 8 + 8 + 3200) + (1 + 4 + 8 + 3)


def test_error_feedback_packed_tracks_residual():
    rng = np.random.default_rng(1)
    ef = ErrorFeedback()
    d = [rng.standard_normal((40, 5)).astype(np.float32)]
    p = ef.compress_packed(d, 2, rng)
    out, _ = decode(p)
    assert np.allclose(ef.residual[0], d[0] - out[0])


@pytest.mark.skipif(not (ROOT / "runs/fl_npu/patch_map.json").exists(), reason="needs runs/fl_npu")
def test_patch_roundtrip_reads_back_head():
    from rafl.fl import npu_head
    pm = npu_head.load_map(ROOT / "runs/fl_npu/patch_map.json")
    base = (ROOT / "runs/fl_npu/classifier_fl.tflite").read_bytes()
    calib = np.load(ROOT / "runs/fl_npu/calib_features_q.npy")
    W, b = npu_head.read_head(base, pm)
    q = npu_head.quantize_head(W * 1.3, b - 0.2, pm, calib)
    patched = npu_head.patch(base, pm, q)
    assert len(patched) == len(base)
    W2, b2 = npu_head.read_head(patched, pm)
    assert np.allclose(W2, W * 1.3, atol=q["w_scale"])
    assert np.allclose(b2, b - 0.2, atol=q["b_scale"])
    s = npu_head.scales(patched, pm)
    assert s["out"] == (pytest.approx(q["o_scale"]), q["o_zp"])
