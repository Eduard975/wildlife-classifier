"""Run the Flower FL simulation baselines over cached features.

Usage:
  python scripts/run_fl_sim.py --arch efficientnet_lite0 --policies fp32,8,4,2,2noef
Writes runs/fl/<arch>/<policy>.csv and summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rafl.fl.sim import SimConfig, fixed_policy, run

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_lite0")
    ap.add_argument("--policies", default="fp32,16,8,4,2,2noef")
    ap.add_argument("--rounds", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    fl = cfg["fl"]
    x = np.load(ROOT / f"data/features/{args.arch}.npy")
    meta = pd.read_parquet(ROOT / f"data/features/{args.arch}_meta.parquet")
    keep = ~((meta.split == "test") & (meta.verified == "n"))
    x, meta = x[keep.to_numpy()], meta[keep].reset_index(drop=True)

    out = ROOT / "runs/fl" / args.arch
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.policies.split(","):
        bits = 32 if name == "fp32" else int(name.replace("noef", ""))
        sc = SimConfig(rounds=args.rounds or fl["rounds"], clients_per_round=fl["clients_per_round"],
                       lr=fl["local_optimizer"]["lr"], momentum=fl["local_optimizer"]["momentum"],
                       label_noise=fl["pseudo_label_noise"], error_feedback=not name.endswith("noef"),
                       seed=cfg["seed"])
        print(f"== {name}", flush=True)
        hist = run(x, meta, classes, sc, fixed_policy(bits),
                   log=lambda r: print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in r.items() if k.endswith(("bacc", "round", "mb", "clients"))})))
        pd.DataFrame(hist).to_csv(out / f"{name}.csv", index=False)
        last = hist[-1]
        summary[name] = {k: last[k] for k in last if k.startswith(("test_", "val_bacc", "cum_upload_mb"))}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(pd.DataFrame(summary).T[["val_bacc", "test_bacc", "test_recall_wolf", "test_recall_bear", "cum_upload_mb"]]
          .round(4).to_string())


if __name__ == "__main__":
    main()
