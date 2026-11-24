"""Resource-aware FL with heterogeneous training tiers and dynamic update quantization (simulation).

Each round, every selected client gets a plan (tier, bits, local epochs) from a policy:
    tier "skip" | "head" (6,405 params) | "block" (417,285 params, see rafl.fl.block)
    bits 2 | 4 | 8 | 32 for the uploaded update (rafl.fl.quant.encode, packed bytes are counted)
The server aggregates each parameter over the clients that trained it (sample-weighted), as in
HeteroFL, and applies the mean update with plain averaging (FedAvg) or a server optimizer (FedAdam,
Reddi et al.). FedProx adds a proximal term to local training. Uploads that miss the round deadline
of the simulated environment (rafl.fl.resources) are discarded after their bytes and energy are spent.

The 1x1-conv weight Wc is quantized in groups of `group` output channels (each group its own
lo/scale), because one range over 409,600 values is too coarse at 2 bits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rafl.fl import block
from rafl.fl.quant import decode, encode
from rafl.fl.resources import EnvConfig, ResourceEnv

# Seconds of on-device work per training sample per local epoch, on the board.
# head: MEASURED. The NPU recomputes the sample's features with the current global conv (15.3 ms per crop
#       including JPEG decode, Phase 8.2 run) plus the head SGD on the A53 (3.8e-5 s).
# block: ESTIMATED. The same NPU pass for the 7x7x320 input, plus FP32 forward and backward of the 1x1 conv
#        on the A53. The Mac needs 2.9e-4 s per sample for that step; the A53 is assumed ~35x slower.
#        Replace with a board measurement.
S_PER_SAMPLE = {"head": 1.53e-2, "block": 2.5e-2}


@dataclass
class Plan:
    tier: str = "head"
    bits: int = 4
    epochs: int = 1
    why: str = ""


@dataclass
class HeteroConfig:
    rounds: int = 100
    clients_per_round: int = 10
    min_client_samples: int = 20
    lr: float = 0.003
    momentum: float = 0.9
    batch: int = 32
    label_noise: float = 0.1
    mu: float = 0.0                # FedProx proximal weight
    server: str = "avg"            # "avg" or "adam"
    server_lr: float = 0.01
    group: int = 64
    eval_every: int = 10
    seed: int = 1337
    env: dict = field(default_factory=dict)


def payload_size(params: dict, tier: str, bits: int, group: int) -> int:
    rng = np.random.default_rng(0)
    return len(encode(split(params, tier, group), bits, rng)[0])


def split(p: dict, tier: str, group: int) -> list[np.ndarray]:
    out = []
    for k in block.TIERS[tier]:
        if k == "Wc":
            out += [p[k][:, s:s + group] for s in range(0, p[k].shape[1], group)]
        else:
            out.append(p[k])
    return out


def join(parts: list[np.ndarray], tier: str, like: dict) -> dict:
    d, i = {}, 0
    for k in block.TIERS[tier]:
        if k == "Wc":
            cols, width = [], 0
            while width < like[k].shape[1]:
                cols.append(parts[i])
                width += parts[i].shape[1]
                i += 1
            d[k] = np.concatenate(cols, 1)
        else:
            d[k] = parts[i].reshape(like[k].shape)
            i += 1
    return d


class Client:
    def __init__(self, cid, xq, y, deq):
        self.cid, self.xq, self.y, self.deq = cid, xq, y, deq
        self.residual: dict[str, list] = {}

    def fit(self, g: dict, plan: Plan, cfg: HeteroConfig, rng) -> tuple[bytes, dict]:
        x = self.deq(self.xq)
        new = block.local_train(g, x, self.y, plan.tier, plan.epochs, cfg.lr, cfg.momentum, cfg.batch, rng, mu=cfg.mu)
        delta = split({k: new[k] - g[k] for k in block.TIERS[plan.tier]}, plan.tier, cfg.group)
        res = self.residual.get(plan.tier)
        if res is not None:
            delta = [d + r for d, r in zip(delta, res)]
        payload, deq = encode(delta, plan.bits, rng)
        self.residual[plan.tier] = [d - q for d, q in zip(delta, deq)]
        return payload, {}


def run(xq_all, meta, classes, deq, init: dict, policy: Callable, cfg: HeteroConfig, log=print) -> list[dict]:
    idx = {c: i for i, c in enumerate(classes)}
    y_all = meta.label.map(idx).to_numpy()
    rng = np.random.default_rng(cfg.seed)
    tr = meta[meta.split == "train"]
    clients = {}
    for loc, g in sorted(tr.groupby("location_id"), key=lambda t: t[0]):
        if len(g) < cfg.min_client_samples:
            continue
        ii = g.index.to_numpy()
        y = y_all[ii]
        flip = rng.random(len(y)) < cfg.label_noise
        clients[loc] = Client(loc, xq_all[ii], np.where(flip, rng.integers(0, len(classes), len(y)), y), deq)
    ids = sorted(clients)
    env = ResourceEnv(ids, EnvConfig(seed=cfg.seed, **cfg.env))
    te = ((meta.split == "test") & (meta.verified != "n")).to_numpy()
    va = (meta.split == "val").to_numpy()
    ev = {"val": (np.flatnonzero(va), y_all[va]), "test": (np.flatnonzero(te), y_all[te])}

    def evaluate(p, which):
        ii, y = ev[which]
        z = np.concatenate([block.logits(p, deq(xq_all[ii[s:s + 1024]])) for s in range(0, len(ii), 1024)])
        pred = z.argmax(1)
        rec = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(classes)}
        return {"bacc": float(np.mean(list(rec.values()))), **{f"recall_{c}": v for c, v in rec.items()}}

    g = {k: v.astype(np.float32).copy() for k, v in init.items()}
    adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in g.items()}
    total, wasted, hist = 0, 0, []
    sizes = {(t, b): payload_size(g, t, b, cfg.group) for t in block.TIERS for b in (2, 4, 8, 32)}
    for rnd in range(1, cfg.rounds + 1):
        env.step(rnd)
        chosen = rng.choice(ids, size=min(cfg.clients_per_round, len(ids)), replace=False)
        sums = {k: np.zeros_like(v) for k, v in g.items()}
        wts = {k: 0.0 for k in g}
        row = {"round": rnd, "head": 0, "block": 0, "skip": 0, "offline": 0, "missed": 0, "busy_train": 0,
               "bytes": 0, "bits": []}
        for cid in chosen:
            c = clients[cid]
            if not env.online(cid):
                row["offline"] += 1
                continue
            plan = policy(rnd, cid, {**env.observe(cid), "n_samples": len(c.y)}, sizes)
            if plan.tier == "skip":
                row["skip"] += 1
                continue
            s_now = env.observe(cid)
            if s_now["cpu_load"] > 0.7 or s_now["battery_soc"] < 0.25 or s_now["temp_c"] >= 85.0:
                row["busy_train"] += 1
            payload, _ = c.fit(g, plan, cfg, rng)
            nbytes = len(payload)
            row["bytes"] += nbytes
            row["bits"].append(plan.bits)
            st = env.settle(cid, len(c.y), plan.epochs, nbytes, S_PER_SAMPLE[plan.tier])
            if not st.accepted:
                row["missed"] += 1
                wasted += nbytes
                continue
            row[plan.tier] += 1
            d = join(decode(payload)[0], plan.tier, g)
            for k, v in d.items():
                sums[k] += v * len(c.y)
                wts[k] += len(c.y)
        for k in g:
            if wts[k] == 0:
                continue
            upd = sums[k] / wts[k]
            if cfg.server == "adam":
                m, v = adam[k]
                m[:] = 0.9 * m + 0.1 * upd
                v[:] = 0.99 * v + 0.01 * upd ** 2
                g[k] = g[k] + cfg.server_lr * m / (np.sqrt(v) + 1e-3)
            else:
                g[k] = g[k] + upd
        total += row["bytes"]
        row.update({"cum_upload_mb": total / 1e6, "cum_wasted_mb": wasted / 1e6,
                    "mean_bits": float(np.mean(row["bits"])) if row["bits"] else 0.0})
        row.pop("bits")
        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            row.update({f"val_{k}": v for k, v in evaluate(g, "val").items()})
            row.update({f"test_{k}": v for k, v in evaluate(g, "test").items()})
            log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                 if k in ("round", "head", "block", "skip", "missed", "cum_upload_mb", "val_bacc", "test_bacc")})
        hist.append(row)
    hist[-1].update(env.summary())
    return hist


# ---------------- policies ----------------

def fixed(tier: str, bits: int):
    return lambda rnd, cid, s, sizes: Plan(tier, bits)


def static_capacity(threshold_kbps: float = 30.0):
    """HeteroFL-style: each client gets a fixed tier from its capacity, decided once from its first
    observed uplink (block if fast, head otherwise) and never revised. No skipping, fixed 4-bit."""
    assigned: dict[str, str] = {}

    def pol(rnd, cid, s, sizes):
        if cid not in assigned:
            assigned[cid] = "block" if s["uplink_kbps"] >= threshold_kbps else "head"
        return Plan(assigned[cid], 4)
    return pol


def resource_aware(margin=0.6, soc_reserve=0.25, soc_block=0.5, cpu_busy=0.7, temp_limit=85.0,
                   candidates=(("block", 4), ("block", 2), ("head", 4), ("head", 2)), schedule=None, rounds=100,
                   s_per_sample=None):
    """Ours: skip to protect detection; otherwise the most useful (tier, bits) whose predicted
    train + upload time fits `margin` of the deadline, using the bandwidth measured on the last upload.
    The block tier also needs battery >= soc_block. `schedule="coarse_first"` tries 2-bit before 4-bit
    for the block tier in the first half of training. `s_per_sample` (tier -> seconds per sample) replaces
    S_PER_SAMPLE, for example with times the board measured itself. Plan.why records the decision."""
    def pol(rnd, cid, s, sizes):
        if s["battery_soc"] < soc_reserve:
            return Plan("skip", 0, why=f"battery {s['battery_soc']:.2f} < {soc_reserve}")
        if s["cpu_load"] > cpu_busy:
            return Plan("skip", 0, why=f"cpu {s['cpu_load']:.2f} > {cpu_busy}")
        if s["temp_c"] >= temp_limit:
            return Plan("skip", 0, why=f"temp {s['temp_c']:.0f} C >= {temp_limit}")
        cands = list(candidates)
        if schedule == "coarse_first" and rnd <= rounds // 2:
            cands = [c for c in cands if c[0] == "block" and c[1] == 2] + [c for c in cands if not (c[0] == "block" and c[1] == 2)]
        kbps = max(s["uplink_kbps"], 1e-3)
        sps = s_per_sample if s_per_sample is not None else S_PER_SAMPLE
        budget = margin * s["deadline_s"]
        notes = []
        for tier, bits in cands:
            if tier == "block" and s["battery_soc"] < soc_block:
                notes.append(f"{tier}/{bits}: battery {s['battery_soc']:.2f} < {soc_block}")
                continue
            t = s["n_samples"] * sps[tier] + sizes[(tier, bits)] * 8 / (kbps * 1000)
            if t <= budget:
                return Plan(tier, bits, why="; ".join(notes + [f"{tier}/{bits}: {t:.0f} s <= {budget:.0f} s at {kbps:.0f} kbit/s"]))
            notes.append(f"{tier}/{bits}: {t:.0f} s > {budget:.0f} s")
        return Plan("skip", 0, why="; ".join(notes))
    return pol
