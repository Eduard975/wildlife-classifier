"""Phase 2 step 4: contact sheets for hand-verifying the wolf/bear test crops.

Sequence-level labels plus MD boxes are still noisy (wrong animal in a mixed
frame, MD box on a rock). The honest evaluation set is a few hundred test-split
crops checked by a person. This writes numbered contact sheets and a CSV with a
blank `verified` column: fill in y / n / ? and rerun with --apply to merge the
answers into crops.parquet.

Output: data/review/<label>_sheet_NN.jpg, data/review/verify.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/crops.parquet"
REVIEW = ROOT / "data/review"
TARGET = {"wolf": 300, "bear": 300}
GRID, CELL = 6, 224


def make_sheets() -> None:
    df = pd.read_parquet(MANIFEST)
    df = df[df.ok & (df.split == "test")]
    REVIEW.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, n in TARGET.items():
        sub = df[df.label == label]
        # spread over locations: round-robin one crop per location
        sub = sub.sample(frac=1, random_state=1337)
        sub = sub.assign(r=sub.groupby("location_id").cumcount()).sort_values("r").head(n)
        per = GRID * GRID
        for s in range(0, len(sub), per):
            chunk = sub.iloc[s:s + per]
            sheet = Image.new("RGB", (GRID * CELL, GRID * CELL), "black")
            d = ImageDraw.Draw(sheet)
            for k, (_, r) in enumerate(chunk.iterrows()):
                idx = s + k
                x, y = (k % GRID) * CELL, (k // GRID) * CELL
                sheet.paste(Image.open(ROOT / r.crop_path), (x, y))
                d.rectangle([x, y, x + 44, y + 18], fill="black")
                d.text((x + 3, y + 3), f"{idx:03d}", fill="yellow")
                rows.append({"label": label, "idx": idx, "crop_id": r.crop_id,
                             "location_id": r.location_id, "sheet": s // per, "verified": ""})
            sheet.save(REVIEW / f"{label}_sheet_{s // per:02d}.jpg", quality=85)
    pd.DataFrame(rows).to_csv(REVIEW / "verify.csv", index=False)
    print(f"wrote {len(rows)} entries and sheets to {REVIEW}")


def apply() -> None:
    v = pd.read_csv(REVIEW / "verify.csv", dtype=str).fillna("")
    v = v[v.verified.str.lower().isin(["y", "n", "?"])]
    df = pd.read_parquet(MANIFEST)
    df["verified"] = df.crop_id.map(dict(zip(v.crop_id, v.verified.str.lower()))).fillna("")
    df.to_parquet(MANIFEST, index=False)
    print(v.groupby(["label", "verified"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    apply() if ap.parse_args().apply else make_sheets()
