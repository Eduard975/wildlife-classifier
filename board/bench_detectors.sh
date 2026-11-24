#!/bin/sh
# Benchmark int8 detectors on the board: CPU (4 threads), NPU, NPU with graph cache.
# Usage on the board: sh bench_detectors.sh <models_dir> <out_dir>
# Raw benchmark_model output for every run goes to <out_dir>/<model>.<mode>.txt
M=$1
O=$2
mkdir -p "$O"
B=$(find / -xdev -maxdepth 6 -name benchmark_model 2>/dev/null | head -1)
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
C=$HOME/rafl/vxcache_$(date +%s)
mkdir -p "$C"
echo "benchmark_model=$B delegate=$D cache=$C" > "$O/env.txt"
uname -a >> "$O/env.txt"
cd "$M" || exit 1
for m in *.tflite; do
  n=${m%.tflite}
  echo "### $n"
  echo "temp_mC_before: $(cat /sys/class/thermal/thermal_zone0/temp)" > "$O/$n.temp.txt"
  $B --graph=$m --num_threads=4 --num_runs=50 > "$O/$n.cpu4.txt" 2>&1
  $B --graph=$m --external_delegate_path=$D --num_runs=100 > "$O/$n.npu.txt" 2>&1
  VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C $B --graph=$m --external_delegate_path=$D --num_runs=5 > "$O/$n.npu_cachebuild.txt" 2>&1
  VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C $B --graph=$m --external_delegate_path=$D --num_runs=100 > "$O/$n.npu_cached.txt" 2>&1
  echo "temp_mC_after: $(cat /sys/class/thermal/thermal_zone0/temp)" >> "$O/$n.temp.txt"
  grep -h 'Inference timings' "$O/$n".*.txt
done
