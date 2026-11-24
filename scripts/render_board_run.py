"""Render a board pipeline run (run_frames.py / run_node.sh) as an annotated video.

Each camera frame is drawn with the detector's boxes and the classifier's label and confidence,
plus a status bar with the burst, backend and on-board timings. After the last frame of a burst
a result card shows the burst vote and whether the board raised an alert. Everything comes
from the run's events.jsonl, so the video shows what the board decided, not a re-run.

  .venv/bin/python scripts/render_board_run.py runs/board/frames_ld_20261121_152830/npu
  .venv/bin/python scripts/render_board_run.py <run> --only b000,b017,b060 --name highlights
Writes <run>/video/frames_<name>/*.jpg and <run>/video/<name>.mp4 (needs ffmpeg). --only keeps the given order.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DANGEROUS = {"bear", "wolf", "lynx", "wild_boar"}
W = 1280
RED, GREEN, GREY, WHITE, BAR = (230, 40, 40), (40, 190, 80), (150, 150, 150), (255, 255, 255), (20, 20, 20)


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)
    except OSError:
        return ImageFont.load_default(size)


def label(d, xy, text, fg, bg, f):
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=bg)
    d.text(xy, text, fill=fg, font=f)


def fit(im):
    h = round(im.height * W / im.width) // 2 * 2
    return im.convert("RGB").resize((W, h), Image.BILINEAR)


def draw_frame(ev, stage, backend, idx, n, run_t):
    im = fit(Image.open(stage / ev["frame"]))
    d = ImageDraw.Draw(im)
    f_box, f_bar = font(26, True), font(22)
    for c in ev["crops"]:
        x0, y0, x1, y1 = (c["box"][0] * im.width, c["box"][1] * im.height, c["box"][2] * im.width, c["box"][3] * im.height)
        col = RED if c["pred"] in DANGEROUS else GREEN
        d.rectangle((x0, y0, x1, y1), outline=col, width=5)
        label(d, (x0 + 6, max(y0 - 34, 4)), f"{c['pred']} {c['conf']:.2f}  (det {c['det_conf']:.2f})", WHITE, col, f_box)
    if not ev["crops"]:
        label(d, (16, 16), "no animal detected", WHITE, GREY, f_box)
    bar = Image.new("RGB", (im.width, 44), BAR)
    b = ImageDraw.Draw(bar)
    b.text((12, 10), f"burst {ev['burst']}  frame {idx}/{n}   board {backend.upper()}   detector {ev['det_ms']:.0f} ms   "
                     f"classifier {ev['cls_ms']:.0f} ms   frame total {ev['total_ms']:.0f} ms   "
                     f"temp {max(ev['temp_c']):.0f} C   t={run_t:.1f} s", fill=WHITE, font=f_bar)
    out = Image.new("RGB", (im.width, im.height + 44))
    out.paste(im, (0, 0))
    out.paste(bar, (0, im.height))
    return out


def decision_banner(img, burst):
    """Clip mode: the board's latest burst decision, kept on screen until the next one."""
    d = ImageDraw.Draw(img)
    f, small = font(34, True), font(20)
    if burst is None:
        text, col = "board: deciding on first burst...", GREY
    elif burst["alert"]:
        text, col = f"ALERT: {burst['pred'].replace('_', ' ')} {burst['conf']:.2f}", RED
    else:
        text, col = f"no alert (vote: {burst['pred'].replace('_', ' ')})", GREY
    sub = f"last decision: burst {burst['burst']}, {burst['frames']} frames" if burst is not None else ""
    w = max(d.textbbox((0, 0), text, font=f)[2], d.textbbox((0, 0), sub, font=small)[2] if sub else 0)
    x = img.width - w - 24
    label(d, (x, 60), text, WHITE, col, f)
    if sub:
        label(d, (x, 110), sub, WHITE, BAR, small)
    return img


def title_card(size, lines):
    im = Image.new("RGB", size, BAR)
    d = ImageDraw.Draw(im)
    y = size[1] // 2 - 30 * len(lines)
    for i, t in enumerate(lines):
        f = font(48 if i == 0 else 30, i == 0)
        x0, _, x1, _ = d.textbbox((0, 0), t, font=f)
        d.text(((size[0] - (x1 - x0)) // 2, y), t, fill=WHITE, font=f)
        y += 70 if i == 0 else 45
    return im


def draw_result(burst, last_frame, alerts_dir):
    im = last_frame.copy()
    over = Image.new("RGBA", im.size, (0, 0, 0, 150))
    im = Image.alpha_composite(im.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(im)
    big, mid = font(64, True), font(32)
    if burst["alert"]:
        head, col = f"ALERT: {burst['pred'].replace('_', ' ')}  {burst['conf']:.2f}", RED
    elif burst["pred"] in DANGEROUS:
        head, col = f"no alert ({burst['pred'].replace('_', ' ')} {burst['conf']:.2f} < threshold)", GREY
    else:
        head, col = f"no alert (vote: {burst['pred'].replace('_', ' ')})", GREY
    label(d, (40, 60), head, WHITE, col, big)
    d.text((44, 170), f"burst {burst['burst']}: {burst['frames']} frames, {burst['boxes']} detections", fill=WHITE, font=mid)
    d.text((44, 215), f"ground truth (not known to the board): {burst['truth'].replace('_', ' ')}", fill=WHITE, font=mid)
    danger = burst["truth"] in DANGEROUS
    if danger and burst["alert"]:
        verdict, vcol = ("correct alert" if burst["pred"] == burst["truth"] else "alert raised, wrong species"), GREEN
    elif danger:
        verdict, vcol = "MISSED: this burst should have raised an alert", RED
    elif burst["alert"]:
        verdict, vcol = "FALSE ALERT", RED
    else:
        verdict, vcol = "correct: no alert", GREEN
    label(d, (44, im.height - 120), verdict, WHITE, vcol, font(36, True))
    if burst.get("alert_file") and alerts_dir and (alerts_dir / burst["alert_file"]).exists():
        th = Image.open(alerts_dir / burst["alert_file"]).convert("RGB")
        th.thumbnail((400, 400))
        x = im.width - th.width - 40
        d.text((x, 270), "alert file written by the board:", fill=WHITE, font=font(24))
        im.paste(th, (x, 305))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder with events.jsonl (e.g. runs/board/frames_ld_20261121_152830/npu)")
    ap.add_argument("--stage", default="runs/board_stage_frames")
    ap.add_argument("--frame-s", type=float, default=1.0, help="seconds each camera frame is shown")
    ap.add_argument("--result-s", type=float, default=2.0, help="seconds each burst result is shown")
    ap.add_argument("--only", default="",
                    help="comma-separated burst ids or patterns (e.g. 'bear_*'), or 'alerts' for alerted bursts only")
    ap.add_argument("--name", default="board_run", help="output name: <run>/video/<name>.mp4")
    ap.add_argument("--clip", action="store_true",
                    help="continuous clip: no result cards, the latest burst decision stays on screen")
    ap.add_argument("--title", default="", help="title card shown first; lines separated by '|'")
    args = ap.parse_args()
    run = Path(args.run)
    if args.stage == "runs/board_stage_frames" and (run / "stage.txt").exists():
        args.stage = (run / "stage.txt").read_text().strip()
    stage = ROOT / args.stage
    events = [json.loads(l) for l in open(run / "events.jsonl")]
    summary = json.load(open(run / "summary.json")) if (run / "summary.json").exists() else {}
    backend = summary.get("backend", run.name)
    bursts = {e["burst"]: e for e in events if e["type"] == "burst"}
    keep = set(bursts)
    if args.only == "alerts":
        keep = {k for k, v in bursts.items() if v["alert"]}
    elif args.only:
        pats = [p.strip() for p in args.only.split(",")]
        keep = {k for k in bursts if any(fnmatch.fnmatch(k, p) for p in pats)}

    out = run / "video"
    frames_dir = out / f"frames_{args.name}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    fps = 10
    k = 0

    def emit(img, seconds):
        nonlocal k
        for _ in range(max(1, round(seconds * fps))):
            img.save(frames_dir / f"{k:05d}.jpg", quality=88)
            k += 1

    per_burst = {}
    for e in events:
        if e["type"] == "frame":
            per_burst.setdefault(e["burst"], []).append(e)
    last = None
    order = [b.strip() for b in args.only.split(",")] if args.only and args.only != "alerts" else None
    if order and not any(ch in args.only for ch in "*?["):
        events = [e for b in order for e in events if e["burst"] == b]
    latest = None
    for e in events:
        if e["burst"] not in keep:
            continue
        if e["type"] == "frame":
            fr = per_burst[e["burst"]]
            last = draw_frame(e, stage, backend, fr.index(e) + 1, len(fr), e["t"])
            if args.title and k == 0:
                emit(title_card(last.size, args.title.split("|")), 3.0)
            emit(decision_banner(last.copy(), latest) if args.clip else last, args.frame_s)
        elif e["type"] == "burst" and last is not None:
            if args.clip:
                latest = e
            else:
                emit(draw_result(e, last, run / "alerts"), args.result_s)
    if args.clip and last is not None and latest is not None:
        emit(decision_banner(last.copy(), latest), args.result_s)

    mp4 = out / f"{args.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.jpg"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    sel = [bursts[b] for b in keep]
    print(f"{len(sel)} bursts, {sum(b['alert'] for b in sel)} alerts, {k / fps:.0f} s of video: {mp4}")


if __name__ == "__main__":
    main()
