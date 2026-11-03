"""Federated simulation of head-only classifier training over camera locations.

Each train-split camera location is one client (natural non-IID split). The
backbone is frozen, so clients train a softmax head on cached features
(scripts/extract_features.py). Uploads are model deltas quantized by a
per-round policy; aggregation is Flower's sample-weighted FedAvg.

Clients are flwr NumPyClient instances, so the same class runs on the board
against a real Flower server in Phase 8. The simulation itself runs in-process
(no Ray), which keeps memory flat and runs deterministically from the seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from flwr.client import NumPyClient
from flwr.server.strategy.aggregate import aggregate

from rafl.fl.quant import ErrorFeedback


# ---------- model: softmax regression head (W, b) ----------

def init_head(dim: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [(rng.standard_normal((dim, k)) * 0.01).astype(np.float32), np.zeros(k, np.float32)]


def logits(params, x):
    return x @ params[0] + params[1]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sgd(params, x, y, k, epochs, lr, momentum, batch, rng, wd=1e-4):
    W, b = params[0].copy(), params[1].copy()
    vW, vb = np.zeros_like(W), np.zeros_like(b)
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            i = order[s:s + batch]
            p = softmax(x[i] @ W + b)
            p[np.arange(len(i)), y[i]] -= 1.0
            p /= len(i)
            gW, gb = x[i].T @ p + wd * W, p.sum(0)
            vW = momentum * vW + gW
            vb = momentum * vb + gb
            W -= lr * vW
            b -= lr * vb
    return [W, b]


# ---------- per-round upload policy (Phase 6 plugs the controller in here) ----------

@dataclass
class RoundPlan:
    participate: bool = True
    bits: int = 32
    local_epochs: int = 1


Policy = Callable[[int, str, dict], RoundPlan]  # (round, client_id, resource_state) -> plan


def fixed_policy(bits: int, local_epochs: int = 1) -> Policy:
    return lambda rnd, cid, state: RoundPlan(True, bits, local_epochs)


# ---------- Flower client ----------

class HeadClient(NumPyClient):
    def __init__(self, cid: str, x: np.ndarray, y: np.ndarray, k: int, lr: float,
                 momentum: float, batch: int, seed: int, error_feedback: bool):
        self.cid, self.x, self.y, self.k = cid, x, y, k
        self.lr, self.momentum, self.batch = lr, momentum, batch
        self.rng = np.random.default_rng(seed)
        self.ef = ErrorFeedback()
        self.use_ef = error_feedback

    def fit(self, parameters, config):
        new = sgd(parameters, self.x, self.y, self.k, int(config.get("local_epochs", 1)),
                  self.lr, self.momentum, self.batch, self.rng)
        delta = [n - p for n, p in zip(new, parameters)]
        qdelta, nbytes = self.ef.compress(delta, int(config.get("bits", 32)), self.rng, self.use_ef)
        upload = [p + d for p, d in zip(parameters, qdelta)]
        return upload, len(self.y), {"upload_bytes": nbytes}

    def evaluate(self, parameters, config):
        p = softmax(logits(parameters, self.x))
        loss = float(-np.log(p[np.arange(len(self.y)), self.y] + 1e-9).mean())
        return loss, len(self.y), {"accuracy": float((p.argmax(1) == self.y).mean())}


# ---------- evaluation ----------

def metrics(params, x, y, classes) -> dict:
    pred = logits(params, x).argmax(1)
    out = {}
    rec = []
    for i, c in enumerate(classes):
        m = y == i
        r = float((pred[m] == i).mean()) if m.any() else float("nan")
        out[f"recall_{c}"] = r
        rec.append(r)
    out["bacc"] = float(np.nanmean(rec))
    out["acc"] = float((pred == y).mean())
    return out


# ---------- simulation driver ----------

@dataclass
class SimConfig:
    rounds: int = 100
    clients_per_round: int = 10
    min_client_samples: int = 20
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    batch: int = 32
    label_noise: float = 0.1
    error_feedback: bool = True
    seed: int = 1337
    eval_every: int = 5
    extra: dict = field(default_factory=dict)


def build_clients(x, meta: pd.DataFrame, classes, cfg: SimConfig):
    """One client per train-split location with enough crops. Label noise models
    on-device labels coming from ranger confirmations + burst-vote pseudo-labels."""
    rng = np.random.default_rng(cfg.seed)
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    tr = meta[meta.split == "train"]
    clients = {}
    for j, (loc, g) in enumerate(sorted(tr.groupby("location_id"), key=lambda t: t[0])):
        if len(g) < cfg.min_client_samples:
            continue
        y = g.label.map(idx).to_numpy()
        flip = rng.random(len(y)) < cfg.label_noise
        y = np.where(flip, rng.integers(0, k, len(y)), y)
        clients[loc] = HeadClient(loc, x[g.index.to_numpy()].astype(np.float32), y, k, cfg.lr,
                                  cfg.momentum, cfg.batch, cfg.seed + j, cfg.error_feedback)
    return clients


def run(x: np.ndarray, meta: pd.DataFrame, classes: list[str], cfg: SimConfig,
        policy: Policy, resource_state: Callable[[int, str], dict] | None = None,
        log: Callable[[dict], None] = print, env_factory=None) -> list[dict]:
    """env_factory(client_ids) -> ResourceEnv (rafl.fl.resources). With an env,
    offline cameras cannot train, and uploads that miss the deadline are
    discarded by the server after their bytes are spent."""
    meta = meta.reset_index(drop=True)
    # standardize features with train-split stats (server-side constant, shipped with the model)
    mu = x[meta.split.to_numpy() == "train"].astype(np.float32).mean(0)
    sd = x[meta.split.to_numpy() == "train"].astype(np.float32).std(0) + 1e-6
    x = ((x.astype(np.float32) - mu) / sd).astype(np.float32)

    clients = build_clients(x, meta, classes, cfg)
    ids = sorted(clients)
    idx = {c: i for i, c in enumerate(classes)}
    ev = {s: (x[(meta.split == s).to_numpy()], meta[meta.split == s].label.map(idx).to_numpy())
          for s in ("val", "test")}
    rng = np.random.default_rng(cfg.seed)
    params = init_head(x.shape[1], len(classes), rng)
    env = env_factory(ids) if env_factory else None
    total_bytes, wasted_bytes, history = 0, 0, []
    log({"clients": len(ids), "samples": int(sum(len(c.y) for c in clients.values()))})

    for rnd in range(1, cfg.rounds + 1):
        if env:
            env.step(rnd)
        chosen = rng.choice(ids, size=min(cfg.clients_per_round, len(ids)), replace=False)
        results, rbytes, rwaste, plans, offline, missed = [], 0, 0, [], 0, 0
        for cid in chosen:
            if env and not env.online(cid):
                offline += 1
                continue
            if env:
                state = {**env.observe(cid), "n_samples": len(clients[cid].y)}
            else:
                state = resource_state(rnd, cid) if resource_state else {}
            plan = policy(rnd, cid, state)
            if not plan.participate:
                continue
            plans.append(plan)
            new, n, m = clients[cid].fit(params, {"bits": plan.bits, "local_epochs": plan.local_epochs})
            rbytes += m["upload_bytes"]
            if env and not env.settle(cid, n, plan.local_epochs, m["upload_bytes"]).accepted:
                missed += 1
                rwaste += m["upload_bytes"]
                continue
            results.append((new, n))
        if results:
            params = [a.astype(np.float32) for a in aggregate(results)]
        total_bytes += rbytes
        wasted_bytes += rwaste
        row = {"round": rnd, "participants": len(results), "offline": offline, "missed_deadline": missed,
               "upload_bytes": rbytes, "cum_upload_mb": total_bytes / 1e6,
               "cum_wasted_mb": wasted_bytes / 1e6,
               "mean_bits": float(np.mean([p.bits for p in plans])) if plans else 0}
        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            row.update({f"val_{k}": v for k, v in metrics(params, *ev["val"], classes).items()})
            row.update({f"test_{k}": v for k, v in metrics(params, *ev["test"], classes).items()})
            log(row)
        history.append(row)
    if env:
        history[-1].update(env.summary())
    return history
