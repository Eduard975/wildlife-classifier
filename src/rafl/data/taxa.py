"""Species grouping and dataset metadata shared by the Phase 2 scripts."""
from __future__ import annotations

import hashlib

# Datasets that contain the target species and for which we have MD results.
# value: (MD results file in data/lila/md, licence note)
DATASETS = {
    "Idaho Camera Traps": ("idaho-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_30_0.200.json.zip",
                           "CDLA-permissive-1.0; no resale; credit IDFG"),
    "WSU Lynx": ("wsu-lynx-2026-02-18-v5a.0.1_detections.filtered.json.zip", "CDLA-permissive-1.0"),
    "AMMonitor Camera Traps": ("ammonitor-camera-traps-2026-06-12-v5a.0.1_detections.filtered.zip",
                               "CDLA-permissive-1.0"),
    "NACTI": ("nacti_mdv5a.0.0_results.filtered_rde_0.150_0.850_50_0.200.json.zip", "CDLA-permissive-1.0"),
    "SWG Camera Traps": ("swg-camera-traps_public_mdv5a.0.0_results.filtered_rde_0.100_0.850_30_0.200.json.zip",
                         "CDLA-permissive-1.0"),
    "ENA24": ("ena24_mdv5a.0.0_results.json.zip", "CDLA-permissive-1.0"),
    "Oregon Critters": ("critters-appel-2025-03-02-v5a.0.1_detections.filtered.json.zip", "CDLA-permissive-1.0"),
    "Caltech Camera Traps": ("caltech-camera-traps_mdv5a.0.0_results.filtered_rde_0.150_0.850_10_0.300.json.zip",
                             "CDLA-permissive-1.0"),
    "Felidae Conservation Fund 2020-2025": ("wildepod-2026-02-16-v5a.0.1_detections.filtered.json.zip",
                                            "CDLA-permissive-1.0"),
}

CLASSES = ["bear", "wolf", "lynx", "wild_boar", "other"]
HARD_NEGATIVES = {"coyote", "dog"}


def group(family: str, genus: str, sci: str, cls: str, kingdom: str) -> str | None:
    """Map a LILA taxonomy row to a fine group; None means drop the row."""
    if family == "ursidae":
        return "bear"
    if sci.startswith("canis familiaris") or sci.startswith("canis lupus familiaris"):
        return "dog"
    if sci.startswith("canis lupus") and "dingo" not in sci:
        return "wolf"
    if genus == "lynx":
        return "lynx"
    if sci.startswith("sus scrofa"):
        return "wild_boar"
    if sci.startswith("canis latrans"):
        return "coyote"
    if sci.startswith("homo sapiens") or genus == "homo":
        return None  # people are the detector's job, not the species classifier's
    if kingdom == "animalia" and cls in ("mammalia", "aves"):
        return "other_animal"
    return None  # empty, vehicle, unidentified, reptiles, etc.


def to_class(fine_group: str) -> str:
    return fine_group if fine_group in CLASSES else "other"


def has_real_sequence(sequence_id: str) -> bool:
    """NACTI and Felidae use one placeholder '<dataset> : unknown' for every image."""
    return bool(sequence_id) and not sequence_id.endswith(" : unknown")


def unit_hash(key: str, salt: str = "rafl") -> float:
    """Deterministic uniform [0,1) from a string."""
    return int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:16], 16) / float(1 << 64)
