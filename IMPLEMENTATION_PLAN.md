# Implementation Plan: Resource-Aware Federated Learning with Dynamic Quantization for Wildlife Alerting on i.MX 8M Plus

Plan version 2026-10-25. Companion to `research/datasets-and-approach.md`.

## 0. Scope and success criteria

Build and measure an end-to-end system that:
1. Detects animals on a remote camera node and classifies dangerous species (bear, wolf, lynx, wild boar) with an inference pipeline that runs int8 on the i.MX 8M Plus NPU.
2. Improves the species classifier over time through federated learning across camera nodes, training on the Cortex-A53 CPU in FP32, re-quantizing for the NPU.
3. Adapts per-client, per-round compute and communication cost (model-update bit width, local steps, trainable layers, participation) to live resource state (CPU load, temperature, memory, bandwidth, battery).

Success is defined by measurable targets, not by "it runs":
- Classifier: per-class recall for bear and wolf at a fixed system false-alarm rate, evaluated on held-out camera locations.
- Edge: end-to-end alert latency on the board, peak RAM, energy per inference and per FL round.
- Communication: model accuracy versus cumulative uploaded bytes; the dynamic controller must beat FP32 FedAvg on accuracy-per-byte without losing more than a stated recall margin.

The board is treated as production hardware on loan: no destructive operations, benchmarks are read/measure only, and all heavy data lives off-board on the laptop.

## 1. Workstreams and dependency order

```
W1 Environment + repo scaffold          (no deps)
W2 Data pipeline (labels -> crops)       (W1)  <-- running now, laptop-side
W3 Baseline models (detector + clf)      (W2)
W4 NPU export + on-board benchmark       (W3, needs SSH)   <-- gated on SSH
W5 Federated learning harness (sim)      (W3)
W6 Dynamic quantization controller       (W5)
W7 Resource telemetry on the board       (W4, SSH)
W8 End-to-end node runtime + alerting    (W4, W6)
W9 Evaluation, ablations, write-up       (W5..W8)
```

W2, W3, W5, W6 need no board and start immediately on the laptop. W4, W7, W8 need the SSH address. W5/W6 (the research contribution) are fully developable in Flower simulation before the board is touched.

## 2. Repository layout

```
EC/
  research/                      # existing: datasets-and-approach.md, counts
  data/
    lila/                        # downloading now: label CSV + MegaDetector results
    crops/                       # extracted species crops (gitignored)
    manifests/                   # train/val/test split by location, per-source licences
  src/
    rafl/
      data/      # LILA parsing, MD-box join, crop extraction, split-by-location
      models/    # detector wrapper, classifier head, quantization utils
      edge/      # tflite/VX delegate runner, telemetry readers, node runtime
      fl/        # Flower client/server, quantized transport, controller
      eval/      # metrics, plots, energy/latency logging
  scripts/       # one-shot CLIs (build_crops.py, export_npu.py, bench_npu.py, run_sim.py)
  configs/       # yaml: dataset paths, model variants, FL/controller hyperparams
  board/         # README + deploy scripts for the i.MX node (rsync, env setup)
  IMPLEMENTATION_PLAN.md
  pyproject.toml # uv-managed; Python 3.12 laptop side
```

## 3. Phase-by-phase plan

### Phase 1 (W1) Environment and scaffold
- `uv` project, Python 3.12 on the laptop. Core deps: `flwr[simulation]`, `torch`/`torchvision` (training + sim), `onnx`, `onnxruntime`, `ultralytics` (detector baseline + export), `tensorflow` CPU (int8 TFLite conversion), `numpy`, `pandas`, `pyarrow`, `pillow`, `matplotlib`, `pyyaml`.
- Board side kept minimal and separate (`board/requirements-board.txt`): `tflite-runtime` built against NXP's eIQ with the VX delegate, `flwr`, `numpy`, `pillow`. No torch on the board.
- Deterministic seeds; a `configs/default.yaml` single source of truth for paths and hyperparameters.
- Exit: `uv run python -c "import flwr, torch, tensorflow"` succeeds; empty pytest passes.

### Phase 2 (W2) Data pipeline  -- running now
Inputs already downloading into `data/lila/`:
- `lila_image_urls_and_labels.csv.zip` (unified labels, 765 MB zip).
- MegaDetector RDE results for the sources that contain the target species: Idaho, WSU Lynx, AMMonitor, NACTI, SWG, ENA24, Oregon Critters, Caltech, Felidae CF.

Steps:
1. Parse the unified CSV, keep only target taxa (reuse `research/lila_count_dangerous.py` grouping) plus coyote/dog as hard negatives for wolf and a sample of "empty"/other-animal.
2. Join each image to its MegaDetector detections (match on relative filename). Drop frames with no detection above a per-dataset confidence floor (~0.2) to remove empties and fix the sequence-label problem.
3. Extract square crops around each animal box, padded, resized to the classifier input (start 224x224, also emit 128x128 for the lite variant).
4. Write a manifest parquet: `crop_path, source, location_id, sequence_id, species, box_conf, split, licence`.
5. Split by `location_id` (whole locations held out; for NACTI, which lacks sequence ids, location is the only safe axis). Target ~70/15/15 by location with stratification so wolf locations appear in all splits given there are only ~200 wolf locations across 2 datasets.
6. Build a small hand-checked wolf/bear test set: sample a few hundred wolf and bear crops, manually confirm, mark `verified=true`. This is the honest evaluation set given sequence-level noise.

Only crops are materialised; full source images are streamed and discarded. Record per-source licence in the manifest (Idaho no-resale + IDFG credit, etc.).
Exit: `data/manifests/crops.parquet` with per-class counts matching the research note within the detection-filter loss, and zero location overlap between splits (asserted in a test).

### Phase 3 (W3) Baseline models
- Detector: start from a pretrained compact detector (YOLO11n or MegaDetector-v6 compact) with classes animal/person/vehicle. We do not train the detector from scratch; we fine-tune lightly if needed. This is the always-on NPU stage-1.
- Classifier: MobileNetV3-Small and EfficientNet-Lite0 crop classifiers over {bear, wolf, lynx, wild boar, other}. Train FP32 on the laptop (GPU/CPU). This is the model federated later.
- Temporal voting: per-sequence/burst aggregation with per-class thresholds biased to recall for dangerous classes.
- Exit: a centrally trained FP32 classifier with a reported per-class recall/precision table on held-out locations; this is the accuracy ceiling the quantized and federated variants are measured against.

### Phase 4 (W4) NPU export and on-board benchmark  -- gated on SSH
- Export the detector and classifier to int8 TFLite with per-tensor quantization (verify the per-tensor vs per-channel speed claim on this NPU; it is currently unverified, flagged in the research note). Use a calibration set drawn from the crop manifest.
- Guard against the two known NPU failure modes: strip int64 ops from Ultralytics exports; avoid `RESIZE_NEAREST_NEIGHBOR half_pixel_centers=true`. Run `benchmark_model` with the VX delegate (`libvx_delegate.so`) and record the delegated-op count: if ops fall back to CPU, fix the export, do not accept the fallback.
- Enable compiled-graph caching (`VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1`, `VIV_VX_CACHE_BINARY_GRAPH_DIR`) so warm-up cost is paid once.
- Benchmark matrix on the board: {detector, classifier} x {NPU int8, NPU int16, CPU FP32} for latency, peak RAM, delegated-op count, and first-vs-warm inference.
- Exit: a table of measured NPU vs CPU latency and memory for each model variant, with the delegated-op count proving the NPU actually runs the graph.

### Phase 5 (W5) Federated learning harness (simulation first)
- Flower simulation. Each LILA `location_id` is one simulated client, giving realistic non-IID partitions without synthetic Dirichlet splits. The board becomes one additional real client in Phase 8.
- Federate only the classifier head (or last block); server aggregates with sample-count weighting (FedAvg). Local training FP32 on CPU to mirror the board; re-quantize to int8 via PTQ on a local calibration set after each local round for the inference path.
- Label source assumption, stated explicitly: on-device labels come from ranger-confirmed/rejected alerts plus high-confidence temporal-voting pseudo-labels. In simulation this is modelled by exposing each client only its own location's labels plus a tunable pseudo-label noise rate.
- Exit: FedAvg FP32 run over N simulated clients reaching within a stated margin of the centralized ceiling; accuracy-vs-round and accuracy-vs-bytes curves logged.

### Phase 6 (W6) Dynamic quantization controller  -- the research contribution
- Quantized transport: stochastic uniform quantization of model updates at b in {2,4,8,16,32} bits with optional error feedback for low-bit clients; server dequantizes before aggregation.
- Controller inputs per client per round: CPU load, temperature (`/sys/class/thermal/thermal_zone*/temp`), free memory, measured uplink bandwidth, simulated battery SoC.
- Controller outputs: update bit width, number of local steps, trainable layer set (head vs last block), participate/skip.
- Baselines to compare against: FedAvg FP32; fixed 8-bit (FedPAQ-style); adaptive schemes from the literature (AdaQuantFL, EA-AQF, federated-supernet mixed precision). Implement at least FedAvg, fixed-8-bit, and our resource-aware policy; cite the others.
- Exit: accuracy-per-uploaded-byte curve for our controller strictly above fixed-8-bit and FP32 at matched recall on dangerous classes, on held-out locations, in simulation.

### Phase 7 (W7) Resource telemetry on the board  -- SSH
- Read-only sampler for CPU load, per-zone temperature, free memory, and a bandwidth probe. Expose the same interface the simulation controller consumes so the real client is a drop-in.
- Energy: measure with the board's power rails if exposed, else an external USB power meter; log per-inference and per-FL-round energy.
- Exit: a telemetry trace from the board feeding the controller, and an energy-per-round number.

### Phase 8 (W8) End-to-end node runtime and alerting  -- SSH
- Node runtime: trigger -> ISP capture -> NPU detector -> crop -> NPU classifier -> temporal vote -> alert with thumbnail. Alert transport stubbed (log/file) unless a real uplink is available.
- The board joins the Flower deployment as one real client alongside simulated laptop clients; one full federated round executes against the real device with the real controller reading real telemetry.
- Exit: a demonstrated alert on a dangerous-species burst, and one real+simulated mixed FL round completing.

### Phase 9 (W9) Evaluation, ablations, write-up
- Metrics as in section 0. Ablations: bit width fixed vs dynamic; head-only vs last-block; participation policy on vs off; pseudo-label noise sensitivity; brown-bear domain gap (train on LILA black bear, test on a small brown-bear set from COCO/Open Images/iNaturalist to quantify the gap the research note predicts).
- Plots: accuracy-vs-rounds, accuracy-vs-bytes, latency/energy bars NPU vs CPU, per-class recall at fixed FAR.
- Exit: results feeding the report; reproducible via `scripts/run_sim.py` + `scripts/bench_npu.py`.

## 4. What needs the SSH address (and what does not)

Needs the board: Phase 4 (NPU export validation + benchmark), Phase 7 (telemetry/energy), Phase 8 (node runtime + real FL client). Everything else (Phases 1,2,3,5,6 and all of the simulation-side evaluation) runs on the laptop and starts now. When you send the IP I will set up `board/` deploy (rsync + eIQ/VX env), run the benchmark matrix, and wire the real telemetry.

## 5. Open items carried from research

- Per-tensor vs per-channel int8 speed on this NPU: unverified, to be measured in Phase 4 (do not bake in the assumption).
- Dangerous-species set depends on deployment region. EEST/Carpathians suggests brown bear is the primary target; LILA has zero brown bear, so Phase 9 explicitly measures the black-to-brown bear domain gap and the design argues on-device FL as the mitigation.
- On-device label source is an assumption (ranger confirmations + pseudo-labels), modelled as pseudo-label noise in simulation.

## 6. Immediate next actions (executing now, laptop-side)

1. Finish the two background downloads (unified label CSV, MegaDetector results).
2. Scaffold the repo (Phase 1): `pyproject.toml`, `src/rafl/` packages, `configs/default.yaml`.
3. Begin Phase 2: parse labels, join MD boxes, extract a first crop batch, build the location-split manifest with the zero-overlap assertion.

Then: wait on the SSH address to start the board-gated phases (4, 7, 8).

## 7. Status log

### 2026-10-25: Phase 1 and Phase 2 done (except the manual verification step)

Pipeline (run in this order, from the project root with `PYTHONPATH=src .venv/bin/python`):
1. `scripts/select_candidates.py`: streams the 30.6M-row unified CSV in about 4.5 min and keeps 364,253 candidate images. Images with more than one label group are dropped (520).
2. `scripts/join_md.py`: streams the MDv5a RDE results with ijson, keeps animal boxes with conf >= 0.2 and area >= 0.15% of the frame, at most 3 per image. Result: 405,943 boxes on 338,369 images. Candidate match rate is above 99% for every dataset.
3. `scripts/build_crops.py`: keeps at most 3 frames per sequence (6 for wolf), applies per-class caps, splits by location, downloads, and saves 224x224 crops. Result: 29,895 of 29,896 crops in `data/crops` (482 MB), manifest at `data/manifests/crops.parquet`.
4. `scripts/review_sheets.py`: writes 600 test-split crops (300 wolf, 300 bear, spread round-robin over locations) to contact sheets plus `data/review/verify.csv`.

| Class | Train | Val | Test |
|---|---|---|---|
| bear | 4193 | 985 | 822 |
| lynx | 3255 | 984 | 761 |
| wild_boar | 3210 | 678 | 1112 |
| wolf | 2517 | 581 | 798 |
| other | 7200 | 1380 | 1419 |

Locations per split: 1473 train / 319 val / 300 test. Wolf locations: 141 / 23 / 37.

Findings:
- NACTI and Felidae CF use one placeholder sequence id (`<dataset> : unknown`) for every image. A naive per-sequence sample keeps all or none of each dataset. `taxa.has_real_sequence` handles this, and those datasets fall back to per-image sampling.
- LILA labels domestic dogs `canis familiaris`, not `canis lupus familiaris`. Before the fix, dogs were silently dropped from the hard negatives.
- The "lynx" class is 98% bobcat (Lynx rufus 4,908 vs Canada lynx 92). For a European deployment, treat it as a proxy for Eurasian lynx and report it that way.
- Wild boar comes from only 3 datasets, and wolf from only 2 (Idaho and WSU). Per-class results for these classes measure transfer across locations, not across regions.
- Spot check of the first wolf sheet and the first bear sheet: about 10-15% of crops are empty, unreadable at night, or possibly coyote. Treat the unverified test numbers as noisy. Report headline wolf/bear recall on the hand-verified subset.
- Only 224 px crops are stored. The 128 px variant is made by resizing at load time.

Open action (user): fill `verified` (y/n/?) in `data/review/verify.csv`, then run `scripts/review_sheets.py --apply`.

Update: verification done (293 y / 6 n / 1 ? wolf; 290 y / 9 n / 1 ? bear), merged into the manifest. Error rate 2.5%, lower than the spot-check estimate.

### 2026-10-25: Board reachable, NPU sanity benchmark

Board: imx8mpevk, Linux 6.12.34-lts, 4x A53, 5.6 GB RAM, 3.0 GB free disk, Python 3.13.4, tflite_runtime 2.19.0, numpy 2.2.3, libvx_delegate.so present, NXP benchmark_model under the TFLite 2.19.0 examples dir. Idle temps 38-41 C.

NXP's bundled mobilenet_v1_1.0_224_quant.tflite (int8 reference, not our model):

| Backend | Avg inference | First inference |
|---|---|---|
| CPU, 4 threads | 40.5 ms | 62 ms |
| NPU (VX delegate) | 3.5 ms | 2.98 s (graph compile) |
| NPU + graph cache (2nd run) | 3.0 ms | 0.27 s |

The cache is enabled with `VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1 VIV_VX_CACHE_BINARY_GRAPH_DIR=~/rafl/vxcache` (3.4 MB `.nb` per model). The node runtime must set both.

### 2026-10-25: Phase 4 first pass, classifier export and NPU benchmark

Export path: `scripts/export_onnx.py` (torch env) -> `scripts/quantize_tflite.py` (`.venv-export`, TF 2.19, onnx2tf 1.28.8) -> full-integer int8 TFLite with int8 I/O, 510 balanced train crops for calibration -> `scripts/eval_tflite.py` (accuracy on the test split minus 'n' crops) -> `board/bench_models.sh` (on-board latency). No int64 tensors in any graph. The FP32 TFLite models match the torch test bacc exactly (0.8577 and 0.8630), so the conversion is correct.

| Model | Test bacc | Verified wolf / bear recall | CPU 4 thr | NPU | NPU cached graph | NPU first inference (cached) |
|---|---|---|---|---|---|---|
| EffNet-Lite0 int8 per-channel | 0.857 | 0.778 / 0.903 | 26.8 ms | 8.7 ms | 7.8 ms | 0.28 s |
| EffNet-Lite0 int8 per-tensor | 0.539 | 0.392 / 0.745 | 26.8 ms | 4.7 ms | 3.8 ms | 0.29 s |
| MobileNetV3-S int8 per-channel | 0.227 | 0.000 / 0.007 | 7.5 ms | 21.8 ms | 28.8 ms | 2.84 s |
| MobileNetV3-S int8 per-tensor | 0.171 | 0.181 / 0.000 | 7.5 ms | 18.9 ms | 24.3 ms | 1.48 s |

Findings:
- Per-tensor int8 is about 1.85x faster than per-channel on this NPU, which confirms the earlier unverified claim. Plain post-training quantization at per-tensor costs 32 points of bacc, though. Per-tensor needs QAT before it can be used. If QAT closes the gap, the per-tensor/per-channel pair becomes a real inference-side switch for the controller (4.7 vs 8.7 ms).
- MobileNetV3-Small is not deployable as is. Post-training int8 destroys it (hard-swish + squeeze-excite, a known PTQ failure), and on the NPU it runs slower than on the CPU. The graph cache does not help either, which points to the graph being split between NPU and CPU.
- Carry forward EfficientNet-Lite0 int8 per-channel: lossless against FP32 (agrees on 96.9% of crops) and 3.1x faster than the CPU.

### 2026-10-25: Phase 3 done, central FP32 baselines

`scripts/train_classifier.py`, ImageNet-pretrained timm weights, 12 epochs AdamW + cosine, sqrt class weights, label smoothing 0.1, grayscale augmentation for IR. Best epoch chosen by val bacc. Test = test split minus 'n' crops (4,897).

| Class | EffNet-Lite0 R / P | MobileNetV3-S R / P |
|---|---|---|
| bear | 0.911 / 0.883 | 0.920 / 0.855 |
| wolf | 0.847 / 0.889 | 0.888 / 0.905 |
| lynx | 0.892 / 0.880 | 0.888 / 0.898 |
| wild_boar | 0.795 / 0.879 | 0.790 / 0.884 |
| other | 0.843 / 0.784 | 0.829 / 0.785 |
| bacc / macro F1 | 0.858 / 0.860 | 0.863 / 0.863 |
| verified wolf recall | 0.782 ± 0.047 (n=293) | 0.857 ± 0.040 |
| verified bear recall | 0.910 ± 0.033 (n=290) | 0.924 ± 0.030 |
| burst-vote bacc (3,043 bursts) | 0.884 | 0.882 |

Hard set ('?') is 2 crops, too small to report. Carry forward EfficientNet-Lite0: MobileNetV3-S is slightly better in FP32 (notably wolf) but does not survive int8 PTQ and is slower on the NPU than the CPU (see Phase 4). Wolf recall is the weakest headline number and the target for fine-tuning and FL.

### 2026-10-25: Phase 5 done, Flower FL simulation baselines

Setup: frozen ImageNet EfficientNet-Lite0 backbone (features cached by `scripts/extract_features.py`), softmax head federated with Flower FedAvg (sample-weighted). 184 clients (train-split camera locations with >= 20 crops), 10 per round, 100 rounds, 1 local epoch, SGD lr 0.01 momentum 0.9, 10% uniform label noise on client labels. Uploads are head deltas, stochastically quantized per tensor; error feedback on unless noted. `scripts/run_fl_sim.py`, outputs in `runs/fl/efficientnet_lite0/`.

| Upload | Test bacc | Wolf recall | Bear recall | Total upload (100 rounds) |
|---|---|---|---|---|
| fp32 | 0.740 | 0.741 | 0.791 | 25.6 MB |
| 16-bit | 0.740 | 0.741 | 0.791 | 12.8 MB |
| 8-bit | 0.741 | 0.746 | 0.796 | 6.4 MB |
| 4-bit | 0.740 | 0.742 | 0.793 | 3.2 MB |
| 2-bit | 0.715 | 0.665 | 0.786 | 1.6 MB |
| 2-bit, no error feedback | 0.703 | 0.681 | 0.750 | 1.6 MB |

Reference points: central head-only on the same frozen features, clean labels, 10 epochs: test bacc 0.787 (wolf 0.765, bear 0.845). Central full fine-tune: 0.858. Wolf/bear recall above is on all test crops of that class (minus 'n'), not the verified subset.

Findings:
- 4-bit uploads match fp32 accuracy at 1/8 of the traffic. 2-bit is the first width that costs accuracy (-2.5 points bacc, -7.6 wolf recall), and error feedback recovers about half of that at 2 bits. This puts the controller's useful range at 2-8 bits; 16 and 32 buy nothing for this head.
- FL ends 4.7 points below the central head-only ceiling. The gap mixes label noise, non-IID clients, and 100 rounds x 10 clients; curves are still rising slowly at round 100 (fp32: 0.701 at r5, 0.738 at r50, 0.740 at r100).
- The larger gap is head-only vs full fine-tune (0.787 vs 0.858). Federating the last block (config `fl.federate: last_block`) is the obvious next lever, at the cost of a bigger upload.

### 2026-10-26: Phase 6 done (head-only, simulated resources)

Code: `src/rafl/fl/resources.py` (per-camera environment), `src/rafl/fl/controller.py` (controller), `scripts/run_phase6.py`, `tests/test_controller.py` (18 tests pass in total). Run: `PYTHONPATH=src .venv/bin/python scripts/run_phase6.py --seeds 3 --scenario {cellular,constrained}`. Output: `runs/phase6/efficientnet_lite0/<scenario>/` (per-run CSVs, `summary.md`, `acc_vs_bytes.png`).

Environment: 1 round = 1 simulated hour. Per camera: solar-charged 20 Wh battery, 0.4 W base detection draw, lognormal site uplink (median 48 kbps "cellular", 8 kbps "constrained"), bursty CPU load at dawn and dusk, temperature from ambient + load (idle 39 C, measured on the board). Uploads that miss the 12 s round deadline are discarded after their bytes and energy are spent. Offline (empty-battery) cameras cannot train under any policy. Energy, bandwidth and training-time numbers are assumptions until Phase 7 measures them.

Controller: skip when battery < 25%, CPU > 0.7 or temperature >= 85 C; otherwise pick the widest width in {2, 4} whose predicted time fits 60% of the deadline, using the bandwidth measured on the client's last upload; skip if even 2-bit would miss.

Results, 3 seeds, mean (test bacc / wolf recall / bear recall / total upload / FL energy):

| Policy | cellular | constrained |
|---|---|---|
| fp32 | 0.742 / 0.713 / 0.807 / 22.3 MB / 6.7 Wh | 0.714 / 0.620 / 0.797 / 22.3 MB (15.7 wasted) / 22.8 Wh |
| fixed 8-bit | 0.749 / 0.731 / 0.809 / 5.59 MB / 4.3 Wh | 0.742 / 0.710 / 0.807 / 5.59 MB / 8.4 Wh |
| fixed 4-bit | 0.748 / 0.732 / 0.814 / 2.80 MB / 3.9 Wh | 0.742 / 0.721 / 0.802 / 2.80 MB / 5.9 Wh |
| fixed 4-bit, random 70% participation | 0.726 / 0.633 / 0.804 / 1.93 MB / 2.7 Wh | 0.720 / 0.636 / 0.789 / 1.93 MB / 4.1 Wh |
| controller | 0.751 / 0.785 / 0.817 / 1.99 MB / 2.7 Wh | 0.735 / 0.776 / 0.797 / 1.59 MB / 2.7 Wh |

Findings:
- Against FP32 and fixed 8-bit the exit criterion is met: the controller reaches val bacc 0.68 after 0.29 MB (fixed 8-bit 0.75 MB, FP32 4.7 MB) and ends at equal or better accuracy with 64-91% less upload.
- Against fixed 4-bit it is not strictly above on the accuracy-vs-bytes curve: MB to reach 0.68 is 0.29 vs 0.32 with seed spread of about 0.14, so equal within noise. The controller's gain over fixed 4-bit is that it stops spending: 29% less upload and 31% less FL energy (cellular), 43% and 55% (constrained), at final bacc within 0.7 points.
- Skipping on resources beats skipping at random. At the same upload (about 1.9 MB), random 70% participation loses 2.5 points bacc and 15 points wolf recall against the controller.
- Wolf recall is 4-6 points higher with the controller than fixed 4-bit in both scenarios, while bacc is unchanged, so other classes lose a little. This is not explained yet; it may come from which locations are skipped. Do not claim it until checked.
- Ablations: the skip rule gives most of the gain. Adapting bits matters only on the constrained links (deadline misses 136 -> 74 for bits-only). Allowing 8-bit doubles upload with no gain. Adaptive local epochs (2 epochs when battery high and CPU idle) costs 2.7-3.5 points bacc and is dropped.
- Camera downtime is 13.3% under every policy. For head-only training, FL energy (2.7-6.7 Wh across 184 cameras over 100 hours) is negligible against the base draw, so the battery rule protects nothing measurable here. It will matter with last-block training or real Phase 7 energy numbers.
- Discarded uploads in fixed widths silently bias training towards well-connected sites: FP32 on constrained links loses 71% of its bytes to missed deadlines.
