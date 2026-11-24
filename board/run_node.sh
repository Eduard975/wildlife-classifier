#!/usr/bin/env bash
# Phase 8 node runtime on the board (NPU): frame source -> detector -> classifier -> burst vote -> alert files.
# Copies results, including alerts/ (JSON + thumbnail), to runs/board/phase8_<ts>.
#   board/run_node.sh [host]        (default $BOARD_HOST, else 192.168.1.12)
#   STAGE=runs/board_stage_clips board/run_node.sh     (other staged frames, e.g. the real bear clip)
set -euo pipefail
export COPYFILE_DISABLE=1
HOST=${1:-${BOARD_HOST:-192.168.1.12}}
STAGE=${STAGE:-runs/board_stage_frames}
BSTAGE=$(basename "$STAGE")
IF=${BOARD_IF:-en0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o "ProxyCommand=nc -b $IF %h %p" "root@$HOST")
TS=$(date +%Y%m%d_%H%M%S)
RUN=phase8_$TS
DET=runs/qat_detector/MDV6-yolov10-c_416
CLS=runs/qat/efficientnet_lite0

"${SSH[@]}" "mkdir -p ~/rafl/$BSTAGE ~/rafl/runs ~/rafl/models"
tar -C "$ROOT/$STAGE" --exclude ./raw -cf - . | "${SSH[@]}" "tar -C ~/rafl/$BSTAGE -xf -"
tar -C "$ROOT/src/rafl/edge" -cf - run_frames.py node_io.py | "${SSH[@]}" "tar -C ~/rafl -xf -"
tar -C "$ROOT/$DET" -cf - int8_per_tensor.tflite | "${SSH[@]}" "mkdir -p ~/rafl/models/det && tar -C ~/rafl/models/det -xf -"
tar -C "$ROOT/$CLS" -cf - efficientnet_lite0_int8_per_tensor.tflite | "${SSH[@]}" "tar -C ~/rafl/models -xf -"

"${SSH[@]}" bash -s -- "$RUN" "$TS" "$BSTAGE" <<'REMOTE'
set -e
RUN=$1; TS=$2; BSTAGE=$3
D=$(find / -xdev -maxdepth 4 -name libvx_delegate.so 2>/dev/null | head -1)
cd ~/rafl
mkdir -p runs/$RUN
python3 run_frames.py --source dir --stage "$BSTAGE" --detector models/det/int8_per_tensor.tflite \
  --classifier models/efficientnet_lite0_int8_per_tensor.tflite --delegate "$D" \
  --cache-dir ~/rafl/vxcache_phase8_$TS --alerts-dir runs/$RUN/alerts --out runs/$RUN 2>&1 | tee runs/$RUN/console.log | tail -60
REMOTE

mkdir -p "$ROOT/runs/board/$RUN"
"${SSH[@]}" "tar -C ~/rafl/runs/$RUN -cf - ." | tar -C "$ROOT/runs/board/$RUN" -xf -
echo "$STAGE" > "$ROOT/runs/board/$RUN/stage.txt"
echo "results: $ROOT/runs/board/$RUN"
