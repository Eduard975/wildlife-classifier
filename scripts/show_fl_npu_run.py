"""Print a run of scripts/run_fl_npu.py as tables: the board's per-round decisions, the global
accuracy per round, the simulated clients' decisions, traffic, and the loop checks.

  .venv/bin/python scripts/show_fl_npu_run.py [runs/fl_npu/run_<ts>]   (default: latest run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FP32_UPLOAD = 25637


def main():
    done = [p.parent for p in (ROOT / "runs/fl_npu").glob("run_*/clients.csv")]
    if len(sys.argv) < 2 and not done:
        raise SystemExit("no complete run in runs/fl_npu (a run is complete when it has clients.csv)")
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(done)
    d = pd.read_csv(run / "clients.csv")
    g = pd.read_csv(run / "global.csv")
    print(f"run: {run}\n")

    b = d[d.client == "board"].copy()
    b["decision"] = [("deploy only" if r.get("deploy_only") == 1 else
                      f"skip ({r.reason})" if r.skipped == 1 else f"train, {int(r.bits)}-bit")
                     for _, r in b.iterrows()]
    cols = {"round": "Round", "background_load": "Background load", "cpu_load": "CPU load", "temp_c": "Temp C",
            "decision": "Decision", "payload_bytes": "Payload B", "wire_bytes": "Wire B",
            "npu_vs_emulation_1lsb": "NPU = emulation", "npu_burst_acc": "NPU burst acc", "model_digest": "Model digest"}
    print("Board, per round")
    print(b[[c for c in cols if c in b]].rename(columns=cols).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nGlobal head after each round (int8, emulated on all test crops)")
    print(g[["round", "test_bacc_int8", "int8_recall_wolf", "int8_recall_bear", "head_digest"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    s = d[(d.client != "board") & (d.deploy_only != 1)]
    print("\nSimulated clients: skips by reason", s[s.skipped == 1].reason.value_counts().to_dict(),
          "| widths chosen", s[s.skipped == 0].bits.astype(int).value_counts().to_dict(),
          "| uploads rejected for the deadline", int((s.accepted == False).sum()))  # noqa: E712

    t = d[d.deploy_only != 1]
    n = int((t.skipped == 0).sum())
    print(f"Traffic: {n} uploads, {int(t.wire_bytes.sum()):,} B on the wire, {n * FP32_UPLOAD:,} B if FP32")

    lc = run / "loop_check.json"
    if lc.exists():
        c = json.load(open(lc))
        same = c["board_runs_server_head"]
        print(f"Loop check: board ran the server's exact file in {sum(same.values())} of {len(same)} rounds; "
              f"board features equal the Mac's on {100 * c['features_equal']:.1f}% "
              f"({100 * c['features_within_1lsb']:.1f}% within 1 LSB)")


if __name__ == "__main__":
    main()
