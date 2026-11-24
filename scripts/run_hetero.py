"""Compare FL policies on the two-tier model (rafl.fl.hetero): literature baselines vs the resource-aware
controller that picks the training tier and the update bit width per client per round.

  PYTHONPATH=src .venv/bin/python scripts/run_hetero.py --scenario cellular --seeds 1337
  PYTHONPATH=src .venv/bin/python scripts/run_hetero.py --policies fedavg_head_fp32,ours --rounds 20
Writes runs/fl_hetero/<scenario>/<policy>_s<seed>.csv and summary.csv / summary.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rafl.fl import hetero

ROOT = Path(__file__).resolve().parents[1]
# one round = one simulated hour with a 120 s upload window (a block update is ~200 KB at 4 bits)
SCENARIOS = {"cellular": {"bw_median_kbps": 48.0, "deadline_s": 120.0},
             "constrained": {"bw_median_kbps": 8.0, "deadline_s": 120.0}}


def policies():
    H = hetero
    return {
        # name: (policy, config overrides)
        "fedavg_head_fp32": (H.fixed("head", 32), {}),                       # McMahan et al., head only
        "fedpaq_head_4": (H.fixed("head", 4), {}),                           # head only, 4-bit (Phase 5 setting)
        "heterofl_static_4_fedadam": (H.static_capacity(), {"server": "adam"}),
        "fedavg_block_fp32": (H.fixed("block", 32), {}),                     # FedAvg, both layers, no compression
        "fedpaq_block_8": (H.fixed("block", 8), {}),                         # FedPAQ-style fixed quantization
        "fedpaq_block_4": (H.fixed("block", 4), {}),
        "fedprox_block_4": (H.fixed("block", 4), {"mu": 0.01}),              # FedProx
        "fedadam_block_4": (H.fixed("block", 4), {"server": "adam"}),        # Adaptive Federated Optimization
        "heterofl_static_4": (H.static_capacity(), {}),                      # HeteroFL-style fixed capacity tiers
        "ours": (H.resource_aware(), {}),
        "ours_coarse_first": (H.resource_aware(schedule="coarse_first"), {}),
        "ours_fedprox": (H.resource_aware(), {"mu": 0.01}),
        "ours_fedadam": (H.resource_aware(), {"server": "adam"}),
        "ours_coarse_first_fedadam": (H.resource_aware(schedule="coarse_first"), {"server": "adam"}),
        "fedprox_block_4_mu1": (H.fixed("block", 4), {"mu": 1.0}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="cellular")
    ap.add_argument("--policies", default="")
    ap.add_argument("--seeds", default="1337")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--init", default="runs/fl_hetero/init_imagenet.npz")
    ap.add_argument("--skip-done", action="store_true", help="skip (policy, seed) pairs already in summary_runs.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    q = json.load(open(ROOT / "runs/fl_hetero/block_quant.json"))
    s_x, z_x = q["x"]
    xq = np.load(ROOT / "data/features/qat_block_q.npy")
    meta = pd.read_parquet(ROOT / "data/features/qat_block_meta.parquet").reset_index(drop=True)
    deq = lambda a: (a.astype(np.float32) - z_x) * s_x  # noqa: E731
    init = dict(np.load(ROOT / args.init))

    out = ROOT / "runs/fl_hetero" / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    pols = policies()
    names = [n for n in (args.policies.split(",") if args.policies else pols)]
    rows = []
    done = set()
    if args.skip_done and (out / "summary_runs.csv").exists():
        d0 = pd.read_csv(out / "summary_runs.csv")
        done = set(zip(d0.policy, d0.seed))
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name in names:
            if (name, seed) in done:
                continue
            pol, over = pols[name] if name in pols else (None, None)
            pol, over = policies()[name]  # fresh state per run
            hc = hetero.HeteroConfig(rounds=args.rounds, seed=seed, env=SCENARIOS[args.scenario], **over)
            t0 = time.time()
            print(f"== {args.scenario} {name} seed {seed}", flush=True)
            hist = hetero.run(xq, meta, classes, deq, init, pol, hc)
            df = pd.DataFrame(hist)
            df.to_csv(out / f"{name}_s{seed}.csv", index=False)
            last = df.iloc[-1]
            rows.append({"policy": name, "seed": seed, "test_bacc": last["test_bacc"],
                         "test_recall_wolf": last["test_recall_wolf"], "test_recall_bear": last["test_recall_bear"],
                         "upload_mb": last["cum_upload_mb"], "wasted_mb": last["cum_wasted_mb"],
                         "block_updates": int(df["block"].sum()), "head_updates": int(df["head"].sum()),
                         "skips": int(df["skip"].sum()), "missed": int(df["missed"].sum()),
                         "busy_train": int(df["busy_train"].sum()),
                         "fl_energy_wh": last.get("fl_energy_wh", np.nan), "minutes": (time.time() - t0) / 60})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
            prev = pd.read_csv(out / "summary_runs.csv") if (out / "summary_runs.csv").exists() else pd.DataFrame()
            allr = pd.concat([prev, pd.DataFrame(rows[-1:])]).drop_duplicates(["policy", "seed"], keep="last")
            allr.to_csv(out / "summary_runs.csv", index=False)
    allr = pd.read_csv(out / "summary_runs.csv")
    s = allr.groupby("policy", sort=False).mean(numeric_only=True).drop(columns="seed")
    s.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(s.round(4).to_markdown())
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
