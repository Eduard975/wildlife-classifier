#!/bin/sh
# Phase 4 matrix on the board. Usage: sh bench_phase4.sh <models_dir> <out_dir> <rafl_dir>
# <models_dir> holds {cls,det}_{fp32,int8,int16}.tflite. Raw output per run goes to <out_dir>/<model>.<mode>.txt
M=$1
O=$2
R=$3
mkdir -p "$O"
B=$(find / -xdev -maxdepth 6 -name benchmark_model 2>/dev/null | head -1)
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
C=$R/vxcache_phase4_$(date +%s)
mkdir -p "$C"
{ echo "benchmark_model=$B delegate=$D cache=$C"; uname -a; uptime; echo "temp_mC: $(cat /sys/class/thermal/thermal_zone0/temp)"; } > "$O/env.txt"
cd "$M" || exit 1
CACHE="VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=$C"
for p in cls det; do
  for v in fp32 int8 int16; do
    m=${p}_$v.tflite
    n=${p}_$v
    echo "### $n"
    if [ "$v" != int16 ]; then
      $B --graph=$m --num_threads=4 --num_runs=50 > "$O/$n.cpu4.txt" 2>&1
      $B --graph=$m --num_threads=4 --num_runs=50 --report_peak_memory_footprint=true > "$O/$n.cpu4_mem.txt" 2>&1
    else
      $B --graph=$m --num_threads=4 --num_runs=20 > "$O/$n.cpu4.txt" 2>&1
    fi
    if [ "$v" != fp32 ]; then
      $B --graph=$m --external_delegate_path=$D --num_runs=100 > "$O/$n.npu.txt" 2>&1
      env $CACHE $B --graph=$m --external_delegate_path=$D --num_runs=5 > "$O/$n.npu_cachebuild.txt" 2>&1
      env $CACHE $B --graph=$m --external_delegate_path=$D --num_runs=100 > "$O/$n.npu_cached.txt" 2>&1
      env $CACHE $B --graph=$m --external_delegate_path=$D --num_runs=100 --report_peak_memory_footprint=true > "$O/$n.npu_cached_mem.txt" 2>&1
      env $CACHE $B --graph=$m --external_delegate_path=$D --num_runs=20 --enable_op_profiling=true > "$O/$n.npu_cached_ops.txt" 2>&1
    fi
    echo "temp_mC_after_$n: $(cat /sys/class/thermal/thermal_zone0/temp) load: $(cut -d' ' -f1 /proc/loadavg)" >> "$O/env.txt"
    grep -h 'Inference timings' "$O/$n".*.txt
  done
done
for p in cls det; do
  for v in fp32 int8 int16; do
    n=${p}_$v
    python3 "$R/rss_probe.py" $M/$n.tflite cpu > "$O/$n.rss_cpu.txt" 2>&1
    if [ "$v" != fp32 ]; then
      python3 "$R/rss_probe.py" $M/$n.tflite npu "$D" "$C" > "$O/$n.rss_npu.txt" 2>&1
    fi
  done
done
echo done > "$O/DONE"
