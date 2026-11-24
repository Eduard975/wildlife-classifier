import numpy as np

from rafl.fl import block, hetero
from rafl.fl.quant import decode, encode


def small_params(rng, cin=8, cout=16, k=5):
    return {"Wc": rng.standard_normal((cin, cout)).astype(np.float32) * 0.3,
            "bc": rng.standard_normal(cout).astype(np.float32) * 0.1,
            "W": rng.standard_normal((cout, k)).astype(np.float32) * 0.1,
            "b": np.zeros(k, np.float32)}


def loss(p, x, y):
    z = block.logits(p, x)
    z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def test_block_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    p = small_params(rng)
    x = rng.uniform(0, 3, (6, 4, 8)).astype(np.float32)
    y = rng.integers(0, 5, 6)
    lr = 1e-3
    new = block.local_train(p, x, y, "block", 1, lr, 0.0, 6, np.random.default_rng(1), wd=0.0)
    grad = (p["Wc"] - new["Wc"]) / lr
    eps = 1e-3
    i, j = 2, 5
    pp = {k: v.astype(np.float64) for k, v in p.items()}
    pp["Wc"][i, j] += eps
    lp = loss(pp, x.astype(np.float64), y)
    pp["Wc"][i, j] -= 2 * eps
    lm = loss(pp, x.astype(np.float64), y)
    assert abs(grad[i, j] - (lp - lm) / (2 * eps)) < 1e-3


def test_head_tier_leaves_conv_unchanged():
    rng = np.random.default_rng(2)
    p = small_params(rng)
    x = rng.uniform(0, 3, (10, 4, 8)).astype(np.float32)
    new = block.local_train(p, x, rng.integers(0, 5, 10), "head", 2, 0.01, 0.9, 4, rng)
    assert np.array_equal(new["Wc"], p["Wc"]) and not np.array_equal(new["W"], p["W"])


def test_split_join_roundtrip_through_packed_payload():
    rng = np.random.default_rng(3)
    p = {"Wc": rng.standard_normal((320, 1280)).astype(np.float32), "bc": rng.standard_normal(1280).astype(np.float32),
         "W": rng.standard_normal((1280, 5)).astype(np.float32), "b": rng.standard_normal(5).astype(np.float32)}
    parts = hetero.split(p, "block", 64)
    assert len(parts) == 1280 // 64 + 3
    payload, deq = encode(parts, 32, rng)
    back = hetero.join(decode(payload)[0], "block", p)
    assert all(np.array_equal(back[k], p[k]) for k in p)


def test_resource_aware_policy_skips_busy_and_falls_back_to_head():
    pol = hetero.resource_aware()
    sizes = {("block", 4): 210_000, ("block", 2): 105_000, ("head", 4): 3_300, ("head", 2): 1_700}
    s = {"battery_soc": 0.9, "cpu_load": 0.1, "temp_c": 40.0, "uplink_kbps": 48.0, "deadline_s": 120.0, "n_samples": 100}
    assert pol(1, "c", s, sizes).tier == "block"
    assert pol(1, "c", {**s, "uplink_kbps": 5.0}, sizes).tier == "head"
    assert pol(1, "c", {**s, "cpu_load": 0.9}, sizes).tier == "skip"
    assert pol(1, "c", {**s, "battery_soc": 0.4}, sizes).tier == "head"
