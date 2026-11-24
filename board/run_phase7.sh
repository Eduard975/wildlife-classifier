#!/usr/bin/env bash
# Phase 7 energy run: steady workload phases on the board, read with a USB power meter.
# Copies results to runs/board/phase7_<ts>/ and writes meter_readings.csv there to fill in.
#   board/run_phase7.sh [host]          (default $BOARD_HOST, else 192.168.1.12)
#   PHASE_S=90 GAP_S=15 PHASES=idle,npu_detector board/run_phase7.sh
set -euo pipefail
export COPYFILE_DISABLE=1
HOST=${1:-${BOARD_HOST:-192.168.1.12}}
IF=${BOARD_IF:-en0}
PHASE_S=${PHASE_S:-90}
GAP_S=${GAP_S:-15}
PHASES=${PHASES:-idle,npu_classifier,npu_detector,npu_pipeline,cpu_pipeline,fl_features,fl_train,wifi_tx}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
     -o "ProxyCommand=nc -b $IF %h %p" "root@$HOST")
TS=$(date +%Y%m%d_%H%M%S)
RUN=phase7_$TS
LOCAL=$ROOT/runs/board/$RUN
DET=runs/qat_detector/MDV6-yolov10-c_416/int8_per_tensor.tflite
CLS=runs/qat/efficientnet_lite0/efficientnet_lite0_int8_per_tensor.tflite
mkdir -p "$LOCAL"

"${SSH[@]}" "mkdir -p ~/rafl/models/det_ld ~/rafl/rafl/fl ~/rafl/runs"
tar -C "$ROOT/src/rafl/edge" -cf - run_frames.py node_io.py telemetry.py energy_phases.py | "${SSH[@]}" "tar -C ~/rafl -xf -"
tar -C "$ROOT/src/rafl/fl" -cf - __init__.py sim.py real.py quant.py | "${SSH[@]}" "tar -C ~/rafl/rafl/fl -xf -"
tar -C "$ROOT/$(dirname $DET)" -cf - "$(basename $DET)" | "${SSH[@]}" "tar -C ~/rafl/models/det_ld -xf -"
tar -C "$ROOT/$(dirname $CLS)" -cf - "$(basename $CLS)" | "${SSH[@]}" "tar -C ~/rafl/models -xf -"

# stdout carries the wifi_tx byte stream and is discarded here; the console arrives on stderr.
"${SSH[@]}" bash -s -- "$RUN" "$PHASE_S" "$GAP_S" "$PHASES" <<'REMOTE' 2>&1 >/dev/null | tee "$LOCAL/console.log"
set -e
RUN=$1; PHASE_S=$2; GAP_S=$3; PHASES=$4
OTHER=$(ps -eo pid,args | grep -E "[p]ython3|[b]enchmark_model" || true)
if [ -n "$OTHER" ]; then echo "board is busy, not starting:" >&2; echo "$OTHER" >&2; exit 3; fi
echo "board idle: load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
N=$(echo "$PHASES" | tr ',' '\n' | grep -c .)
LIMIT=$(( N * (PHASE_S + GAP_S) + GAP_S + 600 ))
D=$(find / -xdev -maxdepth 4 -name libvx_delegate.so 2>/dev/null | head -1)
cd ~/rafl
PYTHONPATH=$HOME/rafl:$HOME/rafl/pylib timeout "$LIMIT" python3 energy_phases.py --out runs/$RUN --delegate "$D" \
  --detector models/det_ld/int8_per_tensor.tflite --classifier models/efficientnet_lite0_int8_per_tensor.tflite \
  --phase-s "$PHASE_S" --gap-s "$GAP_S" --phases "$PHASES"
REMOTE

"${SSH[@]}" "tar -C ~/rafl/runs/$RUN -cf - ." | tar -C "$LOCAL" -xf -
"$ROOT/.venv/bin/python" - "$LOCAL" <<'PY'
import csv, json, sys
d = sys.argv[1]
ph = json.load(open(f"{d}/phases.json"))["phases"]
with open(f"{d}/meter_readings.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "start", "end", "avg_w", "start_mwh", "end_mwh", "note"])
    for p in ph:
        w.writerow([p["phase"], p["start_hms"], p["end_hms"], "", "", "", ""])
PY
echo "results: $LOCAL  (fill in $LOCAL/meter_readings.csv)"
