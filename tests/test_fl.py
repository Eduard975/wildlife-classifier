import numpy as np
import pandas as pd

from rafl.fl.quant import ErrorFeedback, quantize
from rafl.fl.sim import SimConfig, fixed_policy, run


def test_quantize_unbiased_and_payload():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000).astype(np.float32)
    qs = np.mean([quantize(x, 2, rng)[0] for _ in range(200)], axis=0)
    assert abs((qs - x).mean()) < 1e-2
    assert quantize(x, 2, rng)[1] == 10_000 * 2 // 8 + 8
    assert quantize(x, 32, rng)[1] == 40_000
    q8, _ = quantize(x, 8, rng)
    assert np.abs(q8 - x).max() <= (x.max() - x.min()) / 255 + 1e-6


def test_error_feedback_residual_bounded():
    rng = np.random.default_rng(1)
    ef, d = ErrorFeedback(), [np.full(100, 0.3, np.float32), rng.standard_normal(100).astype(np.float32)]
    sent = np.zeros(200)
    for _ in range(50):
        q, _ = ef.compress(d, 2, rng)
        sent += np.concatenate(q)
    # sum of what was sent tracks sum of true deltas to within one residual
    assert np.allclose(sent, 50 * np.concatenate(d), atol=np.abs(np.concatenate(ef.residual)).max() + 1e-4)


def _toy(n_loc=12, per=40, dim=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 3
    rows, xs = [], []
    for loc in range(n_loc):
        split = "train" if loc < 8 else ("val" if loc < 10 else "test")
        for _ in range(per):
            c = rng.integers(k)
            xs.append(centers[c] + rng.standard_normal(dim))
            rows.append({"label": f"c{c}", "split": split, "location_id": f"L{loc}", "verified": ""})
    return np.array(xs, np.float32), pd.DataFrame(rows), [f"c{i}" for i in range(k)]


def test_sim_learns_and_quantized_uploads_are_smaller():
    x, meta, classes = _toy()
    cfg = SimConfig(rounds=20, clients_per_round=4, label_noise=0.0, eval_every=20)
    h32 = run(x, meta, classes, cfg, fixed_policy(32), log=lambda r: None)
    h2 = run(x, meta, classes, cfg, fixed_policy(2), log=lambda r: None)
    assert h32[-1]["test_bacc"] > 0.9
    assert h2[-1]["test_bacc"] > 0.8
    assert h2[-1]["cum_upload_mb"] < h32[-1]["cum_upload_mb"]
    w = np.zeros((1280, 5), np.float32)
    assert quantize(w + np.arange(5), 2, np.random.default_rng(0))[1] < w.size * 4 / 15
