# Local demos

All commands run from `/Volumes/workplace/my-scripts/uni/EC`.

## 1. Alert demo (classifier + burst voting)

Runs the deployed int8 model (`efficientnet_lite0_int8_per_channel.tflite`, the same file benchmarked on the board's NPU) on bursts from test-split camera locations the model never saw during training. Each burst is classified frame by frame, the softmax is averaged over the burst, and an alert is raised when the vote is bear, wolf, lynx or wild boar with confidence >= 0.6.

```
.venv-export/bin/python scripts/demo_alerts.py --per-class 4 --seed 7
open runs/demo/index.html
```

The HTML page shows each burst's crops with the per-frame prediction, the burst vote, the ground truth, and whether an alert fired. `--seed` picks a different set of bursts; `--threshold` changes the alert threshold; `--model` takes any of the TFLite files in `runs/export/`, for example the per-tensor model to show its accuracy loss.

Result for seed 7 (20 bursts, 70 crops): burst accuracy 0.85, 13 of 16 dangerous bursts alerted, 1 false alert on an "other" burst, about 2.8 ms per crop on the Mac CPU.

Limitation: the demo starts from crops cut with MegaDetector boxes, not from full frames, because the on-board detector has not been chosen yet.

## 2. Federated learning controller

Re-runs the Phase 6 comparison (FP32, fixed 8-bit, fixed 4-bit, random participation, controller) over 3 seeds and writes a table and an accuracy-vs-upload plot.

```
PYTHONPATH=src .venv/bin/python scripts/run_phase6.py --seeds 3 --scenario constrained
open runs/phase6/efficientnet_lite0/constrained/acc_vs_bytes.png
cat runs/phase6/efficientnet_lite0/constrained/summary.md
```

Takes under a minute. Use `--scenario cellular` for the 48 kbps case.

## 3. NPU benchmark on the board

Shows NPU vs CPU latency for the exported models (needs SSH to the board, models already copied to `~/rafl/models`):

```
ssh root@192.168.1.17 'sh -s' < board/bench_models.sh
```
