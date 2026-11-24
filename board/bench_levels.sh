#!/bin/sh
# Precision levels on the board: int8 per-tensor / per-channel on the NPU, int8 and FP32 on the GPU,
# FP32 on the NPU's programmable unit, int8 and FP32 on the CPU.
# Usage: sh bench_levels.sh <models_dir> <out_dir>
# <models_dir> holds {cls,det}_{i8pt,i8pc,fp32}.tflite. Raw benchmark_model output per run: <out_dir>/<model>.<level>.txt
M=$1
O=$2
mkdir -p "$O"
B=$(find / -xdev -maxdepth 6 -name benchmark_model 2>/dev/null | head -1)
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
C=$O/vxcache
mkdir -p "$C"
{ echo "benchmark_model=$B delegate=$D"; uname -a; uptime; echo "temp_mC: $(cat /sys/class/thermal/thermal_zone0/temp)"; } > "$O/env.txt"
CACHE="VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C"

run() {  # name level env... -- args...
  n=$1; l=$2; shift 2
  f="$O/$n.$l.txt"
  { echo "### $n $l"; echo "start $(date +%s.%N) temp_mC $(cat /sys/class/thermal/thermal_zone0/temp)"; } > "$f"
  "$@" >> "$f" 2>&1
  echo "end $(date +%s.%N) temp_mC $(cat /sys/class/thermal/thermal_zone0/temp)" >> "$f"
  grep -h 'Inference timings' "$f" | sed "s/^/$n $l: /"
}

for p in cls det; do
  for q in i8pt i8pc; do
    m=$M/${p}_$q.tflite
    run ${p}_$q npu_build env $CACHE "$B" --graph=$m --external_delegate_path=$D --num_runs=3
    run ${p}_$q npu env $CACHE "$B" --graph=$m --external_delegate_path=$D --num_runs=100
  done
  run ${p}_i8pt gpu env USE_GPU_INFERENCE=1 "$B" --graph=$M/${p}_i8pt.tflite --external_delegate_path=$D --num_runs=30
  run ${p}_fp32 gpu env USE_GPU_INFERENCE=1 "$B" --graph=$M/${p}_fp32.tflite --external_delegate_path=$D --num_runs=30
  run ${p}_fp32 npu_ppu env USE_GPU_INFERENCE=0 "$B" --graph=$M/${p}_fp32.tflite --external_delegate_path=$D --num_runs=30
  run ${p}_i8pt cpu "$B" --graph=$M/${p}_i8pt.tflite --num_threads=4 --num_runs=50
  run ${p}_fp32 cpu "$B" --graph=$M/${p}_fp32.tflite --num_threads=4 --num_runs=50
done
echo done > "$O/DONE"
