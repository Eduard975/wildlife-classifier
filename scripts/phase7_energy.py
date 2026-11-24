"""Phase 7: energy per operation from a board/run_phase7.sh run and USB power meter readings.

Fill <run>/meter_readings.csv with, per phase, either the meter's average power (avg_w) or its
energy counter at the START and END banners (start_mwh, end_mwh). The counter is more accurate.

  .venv/bin/python scripts/phase7_energy.py runs/board/phase7_<ts>
Writes <run>/energy.json and <run>/energy.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PAYLOAD_4BIT_BYTES = 3219  # head update per client per round (Phase 8)


def phase_power(row: dict, duration_s: float) -> float | None:
    if row.get("start_mwh") and row.get("end_mwh"):
        return (float(row["end_mwh"]) - float(row["start_mwh"])) * 3.6 / duration_s
    if row.get("avg_w"):
        return float(row["avg_w"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    phases = {p["phase"]: p for p in json.load(open(run / "phases.json"))["phases"]}
    readings = {r["phase"]: r for r in csv.DictReader(open(run / "meter_readings.csv"))}

    tel = defaultdict(list)
    for line in open(run / "telemetry.jsonl"):
        s = json.loads(line)
        tel[s["phase"]].append(s)

    power = {name: phase_power(readings.get(name, {}), p["duration_s"]) for name, p in phases.items()}
    if power.get("idle") is None:
        raise SystemExit("meter_readings.csv needs a reading for the idle phase")
    p_idle = power["idle"]

    rows = []
    for name, p in phases.items():
        w = power[name]
        ss = tel.get(name, [])
        r = {"phase": name, "unit": p["unit"], "count": p["count"], "duration_s": p["duration_s"],
             "ms_per_op": p.get("ms_per_op"), "board_w": w,
             "extra_w": None if w is None else w - p_idle,
             "cpu_util": round(sum(s["cpu_util"] for s in ss) / len(ss), 3) if ss else None,
             "temp_max_c": max((max(s["temp_c"].values()) for s in ss), default=None)}
        if w is not None and name != "idle" and p["count"]:
            r["board_mj_per_op"] = 1000 * w * p["duration_s"] / p["count"]
            r["extra_mj_per_op"] = 1000 * (w - p_idle) * p["duration_s"] / p["count"]
        rows.append(r)
    by = {r["phase"]: r for r in rows}

    derived = {"idle_w": p_idle}
    if "extra_mj_per_op" in by.get("fl_train", {}):
        derived["fl_train_extra_mj_per_round"] = by["fl_train"]["extra_mj_per_op"]
        derived["fl_train_s_per_round"] = by["fl_train"]["ms_per_op"] / 1000
        derived["train_w_extra"] = by["fl_train"]["extra_w"]
    if "extra_mj_per_op" in by.get("wifi_tx", {}):
        tx = phases["wifi_tx"]
        uj_per_byte = 1e6 * by["wifi_tx"]["extra_w"] * tx["duration_s"] / tx["bytes"]
        derived["wifi_extra_uj_per_byte"] = uj_per_byte
        derived["wifi_kbps"] = tx["throughput_kbps"]
        derived["upload_4bit_extra_mj"] = uj_per_byte * PAYLOAD_4BIT_BYTES / 1000
    if "fl_train_extra_mj_per_round" in derived and "upload_4bit_extra_mj" in derived:
        derived["fl_round_extra_mj"] = derived["fl_train_extra_mj_per_round"] + derived["upload_4bit_extra_mj"]
    for k in ("npu_pipeline", "cpu_pipeline", "npu_detector", "npu_classifier", "fl_features"):
        if "extra_mj_per_op" in by.get(k, {}):
            derived[f"{k}_extra_mj_per_op"] = by[k]["extra_mj_per_op"]
            derived[f"{k}_board_mj_per_op"] = by[k]["board_mj_per_op"]

    json.dump({"phases": rows, "derived": derived}, open(run / "energy.json", "w"), indent=1)

    def f(v, nd=1):
        return "" if v is None else f"{v:.{nd}f}"
    lines = ["| Phase | Unit | ms per op | Board W | Above idle W | Board mJ per op | Above-idle mJ per op | CPU util | Max temp C |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['unit']} | {f(r['ms_per_op'])} | {f(r['board_w'], 2)} | {f(r['extra_w'], 2)} | "
                     f"{f(r.get('board_mj_per_op'))} | {f(r.get('extra_mj_per_op'))} | {f(r['cpu_util'], 2)} | {f(r['temp_max_c'], 0)} |")
    lines += ["", "Derived:"] + [f"- {k}: {v:.4g}" for k, v in derived.items()]
    (run / "energy.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
