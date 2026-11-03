#!/bin/sh
# Benchmark every int8 model in ~/rafl/models: CPU (4 threads), NPU, NPU with graph cache.
# Run on the board:  ssh root@<board> 'sh -s' < board/bench_models.sh
B=$(find / -xdev -maxdepth 6 -name benchmark_model 2>/dev/null | head -1)
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
C=$HOME/rafl/vxcache_$(date +%s)
mkdir -p "$C"
cd "$HOME/rafl/models" || exit 1
for m in *.tflite; do
  echo "### $m"
  echo "cpu4: $($B --graph=$m --num_threads=4 --num_runs=50 2>&1 | grep 'Inference timings')"
  echo "npu: $($B --graph=$m --external_delegate_path=$D --num_runs=100 2>&1 | grep -E 'Inference timings|Replacing' | tr '\n' ' ')"
  VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C $B --graph=$m --external_delegate_path=$D --num_runs=5 >/dev/null 2>&1
  echo "npu_cached: $(VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C $B --graph=$m --external_delegate_path=$D --num_runs=100 2>&1 | grep 'Inference timings')"
  echo "temp_mC: $(cat /sys/class/thermal/thermal_zone0/temp)"
done
