import resource
import subprocess
import sys
import time

p = subprocess.Popen(sys.argv[2:])
peak = 0
t0 = time.time()
while p.poll() is None:
    try:
        for line in open(f"/proc/{p.pid}/status"):
            if line.startswith("VmHWM"):
                peak = max(peak, int(line.split()[1]))
    except OSError:
        pass
    time.sleep(0.05)
ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
open(sys.argv[1], "w").write(f"exit={p.returncode} wall_s={time.time() - t0:.1f} polled_VmHWM_kB={peak} ru_maxrss_kB={ru}\n")
