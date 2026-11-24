"""Two-tier trainable model for resource-aware FL: the last block of the deployed classifier plus its head.

    x (49 positions x 320, frozen int8 backbone output)
      -> 1x1 conv Wc (320 x 1280) + bc, ReLU6 -> mean over positions -> f (1280)
      -> head W (1280 x 5) + b -> logits

Tiers (what a client trains in one round):
    "head":  W, b                   (6,405 parameters)
    "block": Wc, bc, W, b           (417,285 parameters)

Parameters are a dict name -> array. numpy only, so it runs on the board's A53 cores.
FedProx: `mu` adds the proximal term (mu/2)||w - w_global||^2 to the local objective.
"""
from __future__ import annotations

import numpy as np

NAMES = ("Wc", "bc", "W", "b")
TIERS = {"head": ("W", "b"), "block": ("Wc", "bc", "W", "b")}


def n_params(tier: str, p: dict) -> int:
    return int(sum(p[k].size for k in TIERS[tier]))


def pooled(p, x):
    """x: (B, 49, 320) float -> pooled features (B, 1280)."""
    return np.clip(x @ p["Wc"] + p["bc"], 0.0, 6.0).mean(1)


def logits(p, x, chunk=512):
    out = []
    for s in range(0, len(x), chunk):
        out.append(pooled(p, x[s:s + chunk]) @ p["W"] + p["b"])
    return np.concatenate(out)


def _softmax_grad(z, y):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    pr = e / e.sum(1, keepdims=True)
    pr[np.arange(len(y)), y] -= 1.0
    return pr / len(y)


def local_train(p_global, x, y, tier, epochs, lr, momentum, batch, rng, mu=0.0, wd=1e-4):
    """SGD with momentum on the parameters of `tier`; others stay at the global values.
    For the head tier the pooled features are computed once with the current global conv."""
    p = {k: v.copy() for k, v in p_global.items()}
    train = TIERS[tier]
    vel = {k: np.zeros_like(p[k]) for k in train}
    n = len(y)
    f_cache = None
    if tier == "head":
        f_cache = np.concatenate([pooled(p, x[s:s + 256]) for s in range(0, n, 256)])
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            i = order[s:s + batch]
            g = {}
            if tier == "head":
                f = f_cache[i]
                dl = _softmax_grad(f @ p["W"] + p["b"], y[i])
            else:
                xb = x[i]
                zc = xb @ p["Wc"] + p["bc"]
                a = np.clip(zc, 0.0, 6.0)
                f = a.mean(1)
                dl = _softmax_grad(f @ p["W"] + p["b"], y[i])
                da = (dl @ p["W"].T)[:, None, :] / a.shape[1]
                dz = da * ((zc > 0) & (zc < 6))
                g["Wc"] = xb.reshape(-1, xb.shape[-1]).T @ dz.reshape(-1, dz.shape[-1])
                g["bc"] = dz.sum((0, 1))
            g["W"] = f.T @ dl
            g["b"] = dl.sum(0)
            for k in train:
                gk = g[k] + wd * p[k] * (k in ("W", "Wc"))
                if mu:
                    gk = gk + mu * (p[k] - p_global[k])
                vel[k] = momentum * vel[k] + gk
                p[k] -= lr * vel[k]
    return p


def metrics(p, x, y, classes):
    pred = logits(p, x).argmax(1)
    out, rec = {}, []
    for i, c in enumerate(classes):
        m = y == i
        r = float((pred[m] == i).mean()) if m.any() else float("nan")
        out[f"recall_{c}"] = r
        rec.append(r)
    out["bacc"] = float(np.nanmean(rec))
    return out
