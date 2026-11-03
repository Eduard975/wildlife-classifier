#!/usr/bin/env bash
# Copy the staged bursts + runtime to the board, run them on CPU and NPU, fetch results.
#   board/run_poc.sh [host]        (default 192.168.1.15)
# BOARD_IF=en0 forces traffic out of the LAN interface when a VPN captures 192.168.1.x.
set -euo pipefail
HOST=${1:-192.168.1.15}
IF=${BOARD_IF:-en0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 -o "ProxyCommand=nc -b $IF %h %p" "root@$HOST")
RUN=poc_$(date +%Y%m%d_%H%M%S)
MODEL=efficientnet_lite0_int8_per_channel.tflite

# The board image has no rsync, so copy with tar over ssh.
"${SSH[@]}" "mkdir -p ~/rafl/stage ~/rafl/runs ~/rafl/models"
tar -C "$ROOT/runs/board_stage" -cf - . | "${SSH[@]}" "tar -C ~/rafl/stage -xf -"
tar -C "$ROOT/src/rafl/edge" -cf - run_bursts.py | "${SSH[@]}" "tar -C ~/rafl -xf -"
tar -C "$ROOT/runs/export/efficientnet_lite0" -cf - "$MODEL" | "${SSH[@]}" "tar -C ~/rafl/models -xf -"

"${SSH[@]}" bash -s -- "$RUN" "$MODEL" <<'EOF'
set -e
RUN=$1; MODEL=$2
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
cd ~/rafl
echo "=== CPU (4 threads)"
python3 run_bursts.py --stage stage --model models/$MODEL --out runs/$RUN/cpu | tail -16
echo "=== NPU ($D)"
python3 run_bursts.py --stage stage --model models/$MODEL --delegate "$D" --out runs/$RUN/npu
EOF

mkdir -p "$ROOT/runs/board/$RUN"
"${SSH[@]}" "tar -C ~/rafl/runs/$RUN -cf - ." | tar -C "$ROOT/runs/board/$RUN" -xf -
echo "results: $ROOT/runs/board/$RUN"
