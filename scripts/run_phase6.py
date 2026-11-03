"""Phase 6: resource-aware controller vs fixed-width baselines under the
simulated camera resource environment.

Usage:
  PYTHONPATH=src python scripts/run_phase6.py --seeds 3 --scenario cellular
Writes runs/phase6/<arch>/<scenario>/<policy>_s<seed>.csv, summary.csv, summary.md, and
acc_vs_bytes.png (if matplotlib is installed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rafl.fl.controller import ControllerConfig, ResourceAwareController
from rafl.fl.resources import EnvConfig, ResourceEnv
from rafl.fl.sim import RoundPlan, SimConfig, fixed_policy, run

ROOT = Path(__file__).resolve().parents[1]
VAL_TARGET = 0.68  # ~95% of the Phase 5 FP32 final val bacc (0.716), below the plateau
BUDGETS_MB = (0.5, 1.0, 2.0)
SCENARIOS = {"cellular": {"bw_median_kbps": 48.0}, "constrained": {"bw_median_kbps": 8.0}}


def random_participation(bits, p, seed):
    rng = np.random.default_rng(seed)
    return lambda rnd, cid, s: RoundPlan(bool(rng.random() < p), bits, 1)


def policies(n_params, seed):
    C = lambda **kw: ResourceAwareController(ControllerConfig(**kw), n_params)
    return {
        "fp32": fixed_policy(32),
        "fixed8": fixed_policy(8),
        "fixed4": fixed_policy(4),
        "fixed2": fixed_policy(2),
        "fixed4_rand70": random_participation(4, 0.70, seed),
        "ctrl": C(),                                   # skip + bits in {2,4}
        "ctrl_bits_only": C(use_skip=False),
        "ctrl_skip_only": C(use_bits=False, fixed_bits=4),
        "ctrl_max8": C(max_bits=8),
        "ctrl_epochs": C(adapt_epochs=True),
    }


def bacc_at_budget(hist, mb):
    d = [r for r in hist if "val_bacc" in r]
    xs, ys = [r["cum_upload_mb"] for r in d], [r["val_bacc"] for r in d]
    return float(np.interp(mb, xs, ys)) if xs and xs[-1] >= mb else float("nan")


def mb_to_target(hist):
    for r in hist:
        if r.get("val_bacc", 0) >= VAL_TARGET:
            return r["cum_upload_mb"]
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--only", default="")
    ap.add_argument("--scenario", default="cellular", choices=sorted(SCENARIOS))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes, fl = cfg["models"]["classifier"]["classes"], cfg["fl"]
    x = np.load(ROOT / f"data/features/{args.arch}.npy")
    meta = pd.read_parquet(ROOT / f"data/features/{args.arch}_meta.parquet")
    keep = ~((meta.split == "test") & (meta.verified == "n"))
    x, meta = x[keep.to_numpy()], meta[keep].reset_index(drop=True)
    n_params = x.shape[1] * len(classes) + len(classes)

    out = ROOT / "runs/phase6" / args.arch / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    names = list(policies(n_params, 0))
    if args.only:
        names = [k for k in names if k in args.only.split(",")]
    rows = []
    for seed in range(args.seeds):
        pols = policies(n_params, seed)
        for name in names:
            pol = pols[name]
            sc = SimConfig(rounds=args.rounds or fl["rounds"], clients_per_round=fl["clients_per_round"],
                           lr=fl["local_optimizer"]["lr"], momentum=fl["local_optimizer"]["momentum"],
                           label_noise=fl["pseudo_label_noise"], seed=cfg["seed"] + seed)
            env_f = lambda ids, s=seed: ResourceEnv(ids, EnvConfig(seed=7 + s, **SCENARIOS[args.scenario]))
            hist = run(x, meta, classes, sc, pol, env_factory=env_f, log=lambda r: None)
            df = pd.DataFrame(hist)
            df.to_csv(out / f"{name}_s{seed}.csv", index=False)
            last = hist[-1]
            rows.append({"policy": name, "seed": seed, "mb_to_target": mb_to_target(hist),
                         **{f"val_at_{b}mb": bacc_at_budget(hist, b) for b in BUDGETS_MB},
                         "participants": int(df.participants.sum()), "missed": int(df.missed_deadline.sum()),
                         "offline_picks": int(df.offline.sum()),
                         **{k: last[k] for k in ("test_bacc", "test_recall_wolf", "test_recall_bear", "val_bacc",
                                                 "cum_upload_mb", "cum_wasted_mb", "camera_offline_pct",
                                                 "fl_energy_wh")}})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
                  flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    g = s.drop(columns="seed").groupby("policy", sort=False)
    m, sd = g.mean(), g.std()
    cols = ["test_bacc", "test_recall_wolf", "test_recall_bear", "cum_upload_mb", "cum_wasted_mb",
            "mb_to_target", *[f"val_at_{b}mb" for b in BUDGETS_MB], "missed", "camera_offline_pct", "fl_energy_wh"]
    m = m.reindex(names); sd = sd.reindex(names)
    tab = m[cols].round(3).astype(str) + " ± " + sd[cols].round(3).astype(str)
    (out / "summary.md").write_text(tab.to_markdown())
    print(tab.to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name in names:
            d = pd.concat([pd.read_csv(out / f"{name}_s{i}.csv") for i in range(args.seeds)])
            d = d.dropna(subset=["val_bacc"]).groupby("round").mean(numeric_only=True)
            ax[0].plot(d.cum_upload_mb, d.val_bacc, label=name)
            ax[1].plot(d.cum_upload_mb, d.test_recall_wolf, label=name)
        for a, t in zip(ax, ("val balanced accuracy", "test wolf recall")):
            a.set_xscale("log"); a.set_xlabel("cumulative upload (MB, log)"); a.set_ylabel(t); a.grid(alpha=.3)
        ax[0].axhline(VAL_TARGET, ls=":", c="k")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "acc_vs_bytes.png", dpi=120)
    except ImportError:
        print("matplotlib not installed; skipped plot")


if __name__ == "__main__":
    main()
