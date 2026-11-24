#!/bin/sh
# Usage: sh rss_int16.sh <phase4_dir>
cd "$1" || exit 1
M=$PWD/models
D=$(find / -xdev -maxdepth 3 -name libvx_delegate.so 2>/dev/null | head -1)
C=$(ls -d $PWD/vxcache_phase4_* | head -1)
export RSS_N=10
for p in det cls; do
  python3 rss_probe.py $M/${p}_int16.tflite cpu > out/${p}_int16.rss_cpu.txt 2>&1
  python3 rss_probe.py $M/${p}_int16.tflite npu "$D" "$C" > out/${p}_int16.rss_npu.txt 2>&1
done
echo done > out/DONE
