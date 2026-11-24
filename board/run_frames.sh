#!/usr/bin/env bash
# Full-frame pipeline on the board: detector -> crops -> classifier -> burst vote.
# Stages frames + runtime, runs CPU then NPU (console also written to the board for live tailing),
# and copies results to runs/board/frames_<ts>/{cpu,npu}.
#   board/run_frames.sh [host]        (default $BOARD_HOST, else 192.168.1.12)
# Watch live from another terminal:  ssh root@<host> 'tail -f ~/rafl/runs/<run>/<cpu|npu>/console.log'
set -euo pipefail
HOST=${1:-${BOARD_HOST:-192.168.1.12}}
IF=${BOARD_IF:-en0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o "ProxyCommand=nc -b $IF %h %p" "root@$HOST")
TS=$(date +%Y%m%d_%H%M%S)
RUN=frames_$TS
DET=runs/qat_detector/MDV6-yolov10-c_416
CLS=runs/qat/efficientnet_lite0

"${SSH[@]}" "mkdir -p ~/rafl/stage_frames ~/rafl/runs ~/rafl/models"
tar -C "$ROOT/runs/board_stage_frames" -cf - . | "${SSH[@]}" "tar -C ~/rafl/stage_frames -xf -"
tar -C "$ROOT/src/rafl/edge" -cf - run_frames.py node_io.py | "${SSH[@]}" "tar -C ~/rafl -xf -"
tar -C "$ROOT/$DET" -cf - int8_per_tensor.tflite | "${SSH[@]}" "mkdir -p ~/rafl/models/det && tar -C ~/rafl/models/det -xf -"
tar -C "$ROOT/$CLS" -cf - efficientnet_lite0_int8_per_tensor.tflite | "${SSH[@]}" "tar -C ~/rafl/models -xf -"

"${SSH[@]}" bash -s -- "$RUN" "$TS" <<'EOF'
set -e
RUN=$1; TS=$2
D=$(find / -xdev -maxdepth 4 -name libvx_delegate.so 2>/dev/null | head -1)
cd ~/rafl
mkdir -p runs/$RUN/cpu runs/$RUN/npu
ARGS="--stage stage_frames --detector models/det/int8_per_tensor.tflite --classifier models/efficientnet_lite0_int8_per_tensor.tflite"
echo "=== CPU (4 threads)"
python3 run_frames.py $ARGS --out runs/$RUN/cpu 2>&1 | tee runs/$RUN/cpu/console.log | tail -30
echo "=== NPU ($D) cache vxcache_frames_$TS"
python3 run_frames.py $ARGS --delegate "$D" --cache-dir ~/rafl/vxcache_frames_$TS --out runs/$RUN/npu 2>&1 | tee runs/$RUN/npu/console.log | tail -50
EOF

mkdir -p "$ROOT/runs/board/$RUN"
"${SSH[@]}" "tar -C ~/rafl/runs/$RUN -cf - ." | tar -C "$ROOT/runs/board/$RUN" -xf -
echo "results: $ROOT/runs/board/$RUN"
