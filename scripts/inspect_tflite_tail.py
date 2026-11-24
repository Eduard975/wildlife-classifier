"""Print the last ops of a TFLite graph with tensor shapes and quantization.

  .venv-export/bin/python scripts/inspect_tflite_tail.py <model.tflite> [n_ops]
"""
import sys

import tensorflow as tf

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
it = tf.lite.Interpreter(path)
td = {t["index"]: t for t in it.get_tensor_details()}
for o in it._get_ops_details()[-n:]:
    ins = [(i, td[i]["name"][:50], td[i]["shape"].tolist(), td[i]["dtype"].__name__, td[i]["quantization"])
           for i in o["inputs"] if i >= 0]
    outs = [(i, td[i]["name"][:50], td[i]["shape"].tolist(), td[i]["quantization"]) for i in o["outputs"]]
    print(o["index"], o["op_name"], ins, "->", outs)
print("inputs", [(d["index"], d["shape"].tolist(), d["dtype"].__name__, d["quantization"]) for d in it.get_input_details()])
print("outputs", [(d["index"], d["shape"].tolist(), d["dtype"].__name__, d["quantization"]) for d in it.get_output_details()])
