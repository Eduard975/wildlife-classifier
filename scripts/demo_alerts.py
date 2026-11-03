"""Local alerting demo: int8 classifier + burst voting on held-out camera bursts.

Runs the same int8 TFLite model that was benchmarked on the board, on bursts from
test-split camera locations the model never saw, and writes an HTML report.

Usage (tensorflow env):
  .venv-export/bin/python scripts/demo_alerts.py --per-class 4 --seed 7
  open runs/demo/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DANGEROUS = ("bear", "wolf", "lynx", "wild_boar")


def load_interpreter(path: Path):
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(str(path), num_threads=4)
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(str(path), num_threads=4)


class Classifier:
    def __init__(self, path: Path):
        self.it = load_interpreter(path)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def probs(self, img: np.ndarray) -> np.ndarray:
        x = ((img - MEAN) / STD)[None]
        if self.i["dtype"] == np.int8:
            s, z = self.i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"])[0].astype(np.float32)
        if self.o["dtype"] == np.int8:
            s, z = self.o["quantization"]
            y = (y - z) * s
        e = np.exp(y - y.max())
        return e / e.sum()


def pick_bursts(df: pd.DataFrame, classes, per_class: int, seed: int) -> pd.DataFrame:
    t = df[df.ok & (df.split == "test") & df.label.isin(classes) & (df.verified != "n")]
    size = t.groupby("seq_key").crop_id.transform("size")
    t = t[size >= 2]
    keys = []
    for c in classes:
        k = t[t.label == c].seq_key.drop_duplicates()
        keys += k.sample(min(per_class, len(k)), random_state=seed).tolist()
    return t[t.seq_key.isin(keys)].sort_values(["seq_key", "frame_num"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/export/efficientnet_lite0/efficientnet_lite0_int8_per_channel.tflite")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    classes = cfg["models"]["classifier"]["classes"]
    df = pd.read_parquet(ROOT / cfg["paths"]["manifest"])
    bursts = pick_bursts(df, classes, args.per_class, args.seed)
    clf = Classifier(ROOT / args.model)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    rows, times = [], []
    for key, g in bursts.groupby("seq_key", sort=False):
        frames = []
        for r in g.itertuples():
            img = np.asarray(Image.open(ROOT / r.crop_path).convert("RGB"), np.float32) / 255.0
            t0 = time.perf_counter()
            p = clf.probs(img)
            times.append(time.perf_counter() - t0)
            frames.append({"crop": r.crop_path, "pred": classes[int(p.argmax())], "conf": float(p.max())})
            g_probs = p if len(frames) == 1 else g_probs + p
        vote = g_probs / len(frames)
        pred = classes[int(vote.argmax())]
        alert = pred in DANGEROUS and float(vote.max()) >= args.threshold
        truth = g.label.iloc[0]
        rows.append({"seq_key": key, "dataset": g.dataset.iloc[0], "location": g.location_id.iloc[0],
                     "truth": truth, "pred": pred, "conf": float(vote.max()), "alert": alert,
                     "correct": pred == truth, "frames": frames})
        mark = "ALERT" if alert else "  -  "
        print(f"{mark} truth={truth:9s} vote={pred:9s} conf={vote.max():.2f} frames={len(frames)}  {key[:60]}")

    danger_true = [r for r in rows if r["truth"] in DANGEROUS]
    summary = {
        "model": args.model,
        "bursts": len(rows),
        "crops": len(times),
        "burst_accuracy": float(np.mean([r["correct"] for r in rows])),
        "dangerous_bursts": len(danger_true),
        "dangerous_alerted": sum(r["alert"] for r in danger_true),
        "false_alerts_on_other": sum(r["alert"] for r in rows if r["truth"] == "other"),
        "mean_ms_per_crop_cpu": 1000 * float(np.mean(times[1:] or times)),
        "threshold": args.threshold,
    }
    json.dump({"summary": summary, "bursts": rows}, open(out / "results.json", "w"), indent=2)
    (out / "index.html").write_text(render(summary, rows, out))
    print(json.dumps(summary, indent=2))
    print(f"report: {out / 'index.html'}")


def render(s: dict, rows: list, out: Path) -> str:
    def rel(p):
        return html.escape(str(Path("../" * len(out.relative_to(ROOT).parts)) / p))

    cards = []
    for r in rows:
        badge = '<span class="b alert">ALERT</span>' if r["alert"] else '<span class="b quiet">no alert</span>'
        ok = '<span class="b ok">correct</span>' if r["correct"] else '<span class="b bad">wrong</span>'
        imgs = "".join(
            f'<figure><img src="{rel(f["crop"])}"><figcaption>{html.escape(f["pred"])} {f["conf"]:.2f}</figcaption></figure>'
            for f in r["frames"])
        cards.append(
            f'<section class="card"><h3>truth: {html.escape(r["truth"])} | vote: {html.escape(r["pred"])} '
            f'({r["conf"]:.2f}) {badge} {ok}</h3><p class="meta">{html.escape(r["dataset"])} | '
            f'{len(r["frames"])} frames</p><div class="row">{imgs}</div></section>')
    return f"""<!doctype html><meta charset="utf-8"><title>Wildlife alert demo</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
.card{{background:#1c1c1c;border-radius:8px;padding:12px;margin:12px 0}}
.row{{display:flex;gap:8px;flex-wrap:wrap}} figure{{margin:0}} img{{width:150px;height:150px;border-radius:4px}}
figcaption{{font-size:12px;color:#aaa;text-align:center}} .meta{{color:#999;font-size:13px;margin:4px 0 8px}}
.b{{font-size:12px;padding:2px 8px;border-radius:10px;margin-left:6px}}
.alert{{background:#b3261e;color:#fff}} .quiet{{background:#444;color:#ddd}}
.ok{{background:#1e6b34;color:#fff}} .bad{{background:#8a6d00;color:#fff}}
td{{padding:2px 12px 2px 0}}
</style>
<h1>Wildlife alert demo (held-out camera locations)</h1>
<table>
<tr><td>Model</td><td>{html.escape(s["model"])}</td></tr>
<tr><td>Bursts / crops</td><td>{s["bursts"]} / {s["crops"]}</td></tr>
<tr><td>Burst accuracy</td><td>{s["burst_accuracy"]:.2f}</td></tr>
<tr><td>Dangerous bursts alerted</td><td>{s["dangerous_alerted"]} of {s["dangerous_bursts"]} (threshold {s["threshold"]})</td></tr>
<tr><td>False alerts on "other"</td><td>{s["false_alerts_on_other"]}</td></tr>
<tr><td>Mean latency per crop (Mac CPU)</td><td>{s["mean_ms_per_crop_cpu"]:.1f} ms</td></tr>
</table>
{"".join(cards)}
"""


if __name__ == "__main__":
    main()
