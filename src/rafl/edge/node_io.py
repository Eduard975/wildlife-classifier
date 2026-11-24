"""Frame sources and alert artefacts for the node runtime (numpy-free apart from PIL)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from PIL import Image, ImageDraw


class Frame(NamedTuple):
    name: str
    load: Callable[[], Image.Image]


class Burst(NamedTuple):
    id: str
    truth: str
    frames: list[Frame]


class DirectoryBurstSource:
    """Replays bursts listed in <stage>/frames.json (scripts/stage_board_frames.py)."""

    def __init__(self, stage):
        self.stage = Path(stage).expanduser()
        self.meta = json.load(open(self.stage / "frames.json"))

    @property
    def classes(self):
        return self.meta["classes"]

    @property
    def animal_index(self):
        return self.meta["animal_index"]

    def first_image(self):
        return Image.open(self.stage / self.meta["bursts"][0]["frames"][0]).convert("RGB")

    def bursts(self) -> Iterator[Burst]:
        for b in self.meta["bursts"]:
            frames = [Frame(f, (lambda p=self.stage / f: Image.open(p).convert("RGB"))) for f in b["frames"]]
            yield Burst(b["id"], b["truth"], frames)


def split_jpegs(data: bytes) -> list[bytes]:
    out, pos = [], 0
    while True:
        a = data.find(b"\xff\xd8", pos)
        if a < 0:
            return out
        b = data.find(b"\xff\xd9", a)
        if b < 0:
            return out
        out.append(data[a:b + 2])
        pos = b + 2


class V4L2Source:
    """Captures MJPEG bursts from a V4L2 device through v4l2-ctl. Needs a camera; not
    exercised on the loaned board, which has none."""

    def __init__(self, device, classes, animal_index, burst_frames=5, n_bursts=1, interval_s=1.0,
                 size=(1280, 720)):
        self.device, self.classes, self.animal_index = device, classes, animal_index
        self.burst_frames, self.n_bursts, self.interval_s, self.size = burst_frames, n_bursts, interval_s, size

    @staticmethod
    def available(device="/dev/video0") -> bool:
        return os.path.exists(device)

    def _capture(self) -> list[bytes]:
        with tempfile.NamedTemporaryFile(suffix=".mjpg") as tmp:
            subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--set-fmt-video=width={self.size[0]},height={self.size[1]},pixelformat=MJPG",
                 "--stream-mmap", f"--stream-count={self.burst_frames}", f"--stream-to={tmp.name}"],
                check=True, timeout=30)
            return split_jpegs(Path(tmp.name).read_bytes())

    def first_image(self):
        import io
        return Image.open(io.BytesIO(self._capture()[0])).convert("RGB")

    def bursts(self) -> Iterator[Burst]:
        import io
        for i in range(self.n_bursts):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            frames = [Frame(f"v4l2_{stamp}_{k}.jpg", (lambda d=d: Image.open(io.BytesIO(d)).convert("RGB")))
                      for k, d in enumerate(self._capture())]
            yield Burst(f"cam{i:03d}", "unknown", frames)
            time.sleep(self.interval_s)


def pick_alert_crop(crops, species_idx):
    """Best (frame, crop) for the voted species. crops: dicts with frame, box, det_conf, probs."""
    scored = [(c["det_conf"] * c["probs"][species_idx], c) for c in crops]
    return max(scored, key=lambda t: t[0])[1] if scored else None


def write_alert(alerts_dir, burst_id, species, confidence, frame_name, box, image, wall_time=None,
                thumb_width=320, extra=None):
    """Writes <burst>.json and <burst>.jpg (downscaled frame with the box drawn). box is normalised xyxy."""
    alerts_dir = Path(alerts_dir)
    alerts_dir.mkdir(parents=True, exist_ok=True)
    wall_time = time.time() if wall_time is None else wall_time
    W, H = image.size
    thumb = image.convert("RGB").resize((thumb_width, max(1, round(H * thumb_width / W))), Image.BILINEAR)
    x0, y0, x1, y1 = (float(v) for v in box)
    rect = [x0 * thumb.width, y0 * thumb.height, x1 * thumb.width, y1 * thumb.height]
    d = ImageDraw.Draw(thumb)
    d.rectangle(rect, outline=(255, 0, 0), width=3)
    d.text((rect[0] + 4, max(rect[1] - 12, 0)), f"{species} {confidence:.2f}", fill=(255, 255, 0))
    jpg = alerts_dir / f"{burst_id}.jpg"
    thumb.save(jpg, quality=85)
    rec = {"time": wall_time, "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall_time)),
           "burst": burst_id, "species": species, "confidence": round(float(confidence), 3),
           "frame": frame_name, "box": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
           "thumbnail": jpg.name}
    rec.update(extra or {})
    (alerts_dir / f"{burst_id}.json").write_text(json.dumps(rec, indent=2))
    return rec
