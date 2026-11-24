import os
import sys
import time

import numpy as np


def status():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmHWM", "VmRSS")):
            k, v = line.split(":")
            d[k] = int(v.split()[0])
    return d


model, mode = sys.argv[1], sys.argv[2]
from tflite_runtime.interpreter import Interpreter, load_delegate

print("baseline", status())
if mode == "npu":
    os.environ["VIV_VX_ENABLE_CACHE_GRAPH_BINARY"] = "1"
    os.environ["VIV_VX_CACHE_BINARY_GRAPH_DIR"] = sys.argv[4]
    it = Interpreter(model, experimental_delegates=[load_delegate(sys.argv[3])])
else:
    it = Interpreter(model, num_threads=4)
print("loaded", status())
it.allocate_tensors()
print("allocated", status())
i = it.get_input_details()[0]
x = np.zeros(i["shape"], i["dtype"])
lat = []
for _ in range(int(os.environ.get("RSS_N", 100))):
    it.set_tensor(i["index"], x)
    t = time.perf_counter()
    it.invoke()
    lat.append((time.perf_counter() - t) * 1000)
print("done", status())
print("first_ms", round(lat[0], 2), "avg_ms", round(float(np.mean(lat[1:])), 2), "p95_ms", round(float(np.percentile(lat[1:], 95)), 2))
