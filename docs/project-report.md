# Resource-Aware Federated Learning with Dynamic Quantization for Wildlife Alerting

Project report, state as of 2026-10-27. Detailed per-phase logs are in `IMPLEMENTATION_PLAN.md` section 7; the original literature and dataset research is in `research/datasets-and-approach.md`.

## 1. Goal

Camera traps in remote areas should raise an alert when a dangerous animal (bear, wolf, lynx, wild boar) is in view. Each camera is an NXP i.MX 8M Plus node that:

1. runs species recognition locally, int8 on the NPU, so no image has to leave the site to decide whether to alert;
2. improves its classifier over time through federated learning (FL) with other cameras, sending model updates instead of images;
3. adapts how much it trains and how many bits it uploads to its current resources (battery, CPU load, temperature, uplink bandwidth). This is the "resource-aware, dynamic quantization" part.

## 2. Hardware

Measured over SSH on the board (192.168.1.17):

| Item | Value |
|---|---|
| Board | imx8mpevk (i.MX 8M Plus EVK) |
| CPU | 4x Cortex-A53 |
| NPU | VeriSilicon NPU via the VX delegate (`libvx_delegate.so`), inference only, int8/int16 |
| RAM / free disk | 5.6 GB / 3.0 GB |
| OS / runtime | Linux 6.12.34-lts, Python 3.13.4, tflite_runtime 2.19.0, numpy 2.2.3 |
| Idle temperature | 38-41 C |

Constraint that shapes the whole design: the NPU cannot train. Inference runs int8 on the NPU; local FL training has to run in FP32 on the A53 cores. "Dynamic quantization" therefore means choosing the bit width of the uploaded model updates, which model variant to run, and whether and how much a camera trains in a round. It does not mean switching the NPU between arbitrary bit widths.

## 3. Datasets

### 3.1 Why LILA

LILA BC (lila.science) hosts the largest collection of public camera-trap datasets under one label taxonomy, with a single combined label file (30.6M rows, 21.7 GB uncompressed) and MegaDetector results for every image. Camera-trap images differ from web photos (night infrared, motion blur, partial animals, fixed backgrounds), so training on camera-trap data matters more than training on, for example, ImageNet or iNaturalist photos.

LILA also gives each image a `location_id` (camera site). This was used twice: to split the data without leakage (section 3.4) and as the natural unit for a simulated FL client (section 7).

### 3.2 Which datasets and why

The combined label file was scanned for the target taxa, and every LILA dataset that contributed images was kept:

| Dataset | bear | lynx | wild_boar | wolf | other | Total crops | Licence |
|---|---|---|---|---|---|---|---|
| NACTI | 3309 | 1753 | 3175 | 0 | 5473 | 13710 | CDLA-permissive-1.0 |
| Idaho Camera Traps | 172 | 64 | 0 | 3236 | 229 | 3701 | CDLA-permissive-1.0, no resale, credit IDFG |
| Felidae Conservation Fund | 3 | 2133 | 147 | 0 | 1058 | 3341 | CDLA-permissive-1.0 |
| AMMonitor | 1450 | 205 | 0 | 0 | 716 | 2371 | CDLA-permissive-1.0 |
| SWG Camera Traps | 38 | 0 | 1678 | 0 | 524 | 2240 | CDLA-permissive-1.0 |
| WSU Lynx | 623 | 235 | 0 | 660 | 666 | 2184 | CDLA-permissive-1.0 |
| Caltech Camera Traps | 0 | 501 | 0 | 0 | 1104 | 1605 | CDLA-permissive-1.0 |
| Oregon Critters | 298 | 88 | 0 | 0 | 128 | 514 | CDLA-permissive-1.0 |
| ENA24 | 107 | 21 | 0 | 0 | 101 | 229 | CDLA-permissive-1.0 |
| **Total** | 6000 | 5000 | 5000 | 3896 | 9999 | 29895 | |

Reasons for the class design:
- **Five classes: bear, wolf, lynx, wild_boar, other.** The four dangerous species are the alert targets. "other" holds everything that must not alert.
- **Hard negatives inside "other":** 2,500 coyote and 999 domestic dog crops, because they are the animals most often confused with wolves. The remaining 6,500 are a sample of other animals (deer, etc.).
- **Wolf is the limiting class.** LILA has only 5,537 wolf images in 856 sequences, all from Idaho and WSU Lynx. Wolf was allowed 6 frames per sequence (other classes 3) to use what exists.
- **Caps per class** (6,000 bear, 5,000 lynx and boar, 10,000 other) keep the classes within a factor of about 2.5 of each other and bound the download size.

### 3.3 Known dataset limits

- **No European species data in LILA.** Bear is 97% American black bear and there is no brown bear. The "lynx" class is 98% bobcat (4,908 bobcat vs 92 Canada lynx), so it is a stand-in for Eurasian lynx. Results describe transfer to new camera sites in North America, not to the Carpathians.
- **European sources considered, not used:** DeepFaune (France, includes bear, wolf, lynx; weights CC BY-NC-SA, images only on request), TrapperAI (YOLOv8-m covering wolf and lynx, no bear; GPL-3.0 weights, data not public), CzechLynx (about 40k Eurasian lynx images, lynx only). None gives openly downloadable bear and wolf images.
- **Wolf comes from 2 datasets, wild boar from 3.** Per-class results for these depend on few sources.

### 3.4 Pipeline from labels to crops

1. **Candidate selection** (`scripts/select_candidates.py`): streams the 30.6M-row label file in about 4.5 minutes and keeps 364,253 candidate images of the target taxa and negatives. Images with labels from more than one class group are dropped (520).
2. **MegaDetector join** (`scripts/join_md.py`): LILA labels are mostly per sequence (a whole burst gets one label), so many "wolf" frames contain no wolf. Each image was joined to LILA's precomputed MegaDetector v5a results, keeping animal boxes with confidence >= 0.2 and area >= 0.15% of the frame, at most 3 per image. Result: 405,943 boxes on 338,369 images; above 99% of candidates matched. MegaDetector v5a (a YOLOv5x6 detector, 1280 px input) was only used offline as a labelling tool; it is far too large for the board.
3. **Crop build** (`scripts/build_crops.py`): per-sequence and per-class caps, location split, then streams each source image from LILA's cloud storage, cuts a square crop around the box with 15% padding, resizes to 224x224 and discards the full image. 29,895 of 29,896 crops succeeded (482 MB). Manifest: `data/manifests/crops.parquet` (source, location, sequence, box, confidence, split, licence per crop).
4. **Split by camera location, not randomly** (`src/rafl/data/split.py`): a random split puts frames from the same camera, same background and often the same burst into both train and test, which inflates accuracy. Whole locations are held out instead: 1,473 train, 319 val, 300 test locations; wolf crops come from 141 / 23 / 37 locations. A test asserts zero location overlap.

| Class | Train | Val | Test |
|---|---|---|---|
| bear | 4193 | 985 | 822 |
| lynx | 3255 | 984 | 761 |
| wild_boar | 3210 | 678 | 1112 |
| wolf | 2517 | 581 | 798 |
| other | 7200 | 1380 | 1419 |

Two data bugs were found and fixed: NACTI and Felidae give every image the same placeholder sequence ID, which made per-sequence sampling keep all or none of those datasets; and LILA spells domestic dog `canis familiaris`, which first dropped all dogs from the hard negatives.

### 3.5 Hand-verified test set

The labels are automatic (sequence label plus MegaDetector box), so headline recall needs human-checked crops. 600 test crops (300 wolf, 300 bear, spread over test locations) were put on contact sheets (`scripts/review_sheets.py`) and marked y / n / ? by hand:

| Class | y (clean) | ? (hard) | n (removed) |
|---|---|---|---|
| wolf | 293 | 1 | 6 |
| bear | 290 | 1 | 9 |

The automatic labels were wrong for 2.5% of these crops. "n" crops are excluded from every test metric; headline wolf and bear recall is reported on the "y" crops.

## 4. Models

### 4.1 Classifier candidates and why

Two small ImageNet-pretrained classifiers from `timm`, chosen because they are standard mobile architectures with known TFLite int8 paths and fit the NPU's memory easily:

- **EfficientNet-Lite0** (`efficientnet_lite0.ra_in1k`): the "Lite" variant was designed for edge accelerators. It replaces swish with ReLU6 and removes squeeze-and-excitation blocks, both of which quantize poorly.
- **MobileNetV3-Small** (`mobilenetv3_small_100.lamb_in1k`): about half the size and the usual smallest baseline. It uses hard-swish and squeeze-and-excitation, so it was the test of whether those survive int8 on this NPU.

### 4.2 Specifications

| | EfficientNet-Lite0 | MobileNetV3-Small |
|---|---|---|
| Parameters (5 classes) | 3.38 M | 1.52 M |
| Classifier head (federated part) | 6,405 params (1280 x 5 + 5) | 5,125 params (576 x 5 + 5) |
| Input | 224 x 224 RGB, ImageNet normalisation | same |
| PyTorch checkpoint | 13.8 MB | |
| TFLite FP32 | 13.4 MB | 6.1 MB |
| TFLite int8 per-channel | 3.95 MB | 1.83 MB |
| TFLite int8 per-tensor | 3.45 MB | 1.59 MB |
| Board CPU (4 threads), int8 | 26.8 ms | 7.5 ms |
| Board NPU, int8 per-channel (cached graph) | 8.7 ms (7.8 ms) | 21.8 ms (28.8 ms) |
| Board NPU, int8 per-tensor (cached graph) | 4.7 ms (3.8 ms) | 18.9 ms (24.3 ms) |

### 4.3 Detector

Not done yet. The planned stage 1 is a compact int8 detector on the NPU (YOLO11n or a compact MegaDetector variant) that finds animal, person and vehicle boxes in the full frame. So far the classifier has only been run on crops cut from MegaDetector boxes computed offline. Choosing and benchmarking the on-board detector is open work (section 10).

## 5. Training (yes, fine-tuning)

Both classifiers were fine-tuned end to end from ImageNet weights on the 20,375 train crops (`scripts/train_classifier.py`, Mac GPU via MPS):

- 12 epochs, AdamW, lr 5e-4, weight decay 0.05, cosine schedule, batch 128;
- class-weighted cross-entropy (square-root inverse frequency) with label smoothing 0.1;
- augmentation: random resized crop (scale 0.6-1.0), flip, colour jitter, random grayscale with p = 0.25 so daytime colour crops also look like night infrared;
- the epoch with the best validation balanced accuracy is kept (epoch 10 for EfficientNet-Lite0).

Results on the test split (held-out locations, "n" crops removed, 4,897 crops):

| Class | EfficientNet-Lite0 recall / precision | MobileNetV3-Small recall / precision |
|---|---|---|
| bear | 0.911 / 0.883 | 0.920 / 0.855 |
| wolf | 0.847 / 0.889 | 0.888 / 0.905 |
| lynx | 0.892 / 0.880 | 0.888 / 0.898 |
| wild_boar | 0.795 / 0.879 | 0.790 / 0.884 |
| other | 0.843 / 0.784 | 0.829 / 0.785 |
| Balanced accuracy | 0.858 | 0.863 |
| Verified wolf recall (n=293) | 0.782 +/- 0.047 | 0.857 +/- 0.040 |
| Verified bear recall (n=290) | 0.910 +/- 0.033 | 0.924 +/- 0.030 |
| Balanced accuracy after burst voting | 0.884 | 0.882 |

Burst voting averages the softmax over all crops of a sequence; this is how the camera decides, since a bear often appears clearly in only some frames. The most common errors are wild boar predicted as "other" (172 crops) and wolf predicted as "other" (73), the latter mostly coyote-like crops.

## 6. Quantization and NPU export

Path: PyTorch -> ONNX (`scripts/export_onnx.py`) -> TFLite full-integer int8 with int8 input and output, calibrated on 510 class-balanced train crops (`scripts/quantize_tflite.py`, TensorFlow 2.19 and onnx2tf in `.venv-export`) -> accuracy check (`scripts/eval_tflite.py`) -> board benchmark (`board/bench_models.sh`). The FP32 TFLite models reproduce the PyTorch accuracy exactly, and no graph contains int64 ops (a known NPU problem with some exports).

| Model | Test balanced accuracy | Verified wolf / bear recall |
|---|---|---|
| EfficientNet-Lite0 FP32 | 0.858 | 0.782 / 0.910 |
| EfficientNet-Lite0 int8 per-channel | 0.857 | 0.778 / 0.903 |
| EfficientNet-Lite0 int8 per-tensor | 0.539 | 0.392 / 0.745 |
| MobileNetV3-Small int8 per-channel | 0.227 | 0.000 / 0.007 |
| MobileNetV3-Small int8 per-tensor | 0.171 | 0.181 / 0.000 |

Decisions and findings:
- **Deployed model: EfficientNet-Lite0 int8 per-channel.** It loses 0.1 points against FP32, agrees with FP32 on 96.9% of crops, and runs 3.1x faster on the NPU than on the CPU.
- **MobileNetV3-Small was dropped.** Post-training int8 destroys its accuracy, and it runs slower on the NPU than on the CPU, which suggests the graph is split between NPU and CPU.
- **Per-tensor is 1.85x faster than per-channel on this NPU** but costs 32 points without quantization-aware training (QAT). If QAT closes the gap, the two variants become an inference-side speed/accuracy switch for the controller.
- **NPU graph cache:** the first NPU inference compiles the graph (about 3 s). Setting `VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1` and `VIV_VX_CACHE_BINARY_GRAPH_DIR` cuts it to about 0.28 s. The node runtime must set both.

## 7. Federated learning simulation

Framework: Flower, run in a single process on the Mac (`src/rafl/fl/sim.py`, `scripts/run_fl_sim.py`).

- **Clients:** each train-split camera location with at least 20 crops is one client (184 clients). This gives realistic uneven, non-IID data without synthetic partitioning.
- **What is federated:** only the classifier head (6,405 parameters), on features from the frozen ImageNet EfficientNet-Lite0 backbone, cached once (`scripts/extract_features.py`). Head-only keeps local training cheap enough for the A53 cores and the uploads small.
- **Labels on the device:** the realistic source is ranger confirmations of alerts plus confident burst-vote pseudo-labels. This is modelled as 10% random label noise on every client.
- **Rounds:** 100 rounds, 10 clients per round, 1 local epoch, SGD lr 0.01 momentum 0.9, FedAvg weighted by sample count.
- **Update quantization** (`src/rafl/fl/quant.py`): stochastic uniform rounding of the head update to 2, 4, 8, 16 or 32 bits per tensor, with error feedback (the rounding error is added to the next round's update).

| Upload | Test balanced accuracy | Wolf recall | Bear recall | Total upload |
|---|---|---|---|---|
| 32-bit | 0.740 | 0.741 | 0.791 | 25.6 MB |
| 16-bit | 0.740 | 0.741 | 0.791 | 12.8 MB |
| 8-bit | 0.741 | 0.746 | 0.796 | 6.4 MB |
| 4-bit | 0.740 | 0.742 | 0.793 | 3.2 MB |
| 2-bit | 0.715 | 0.665 | 0.786 | 1.6 MB |
| 2-bit, no error feedback | 0.703 | 0.681 | 0.750 | 1.6 MB |

Reference points: the same head trained centrally on the same frozen features reaches 0.787; full central fine-tuning reaches 0.858. So 4-bit uploads are free, 2-bit is the first width that costs accuracy, FL loses 4.7 points against central head-only training, and training only the head costs more (7.1 points) than federation does.

## 8. Resource-aware controller

Code: `src/rafl/fl/resources.py` (simulated camera resources), `src/rafl/fl/controller.py` (controller), `scripts/run_phase6.py`.

**Simulated environment.** One round is one hour. Each camera has a solar-charged 20 Wh battery, a 0.4 W base draw for detection, a site-specific lognormal uplink (median 48 kbps "cellular" or 8 kbps "constrained"), bursty CPU load at dawn and dusk, and a temperature from ambient plus load (idle 39 C as measured on the board). An upload that misses the 12 s round deadline is discarded after its bytes and energy are spent.

**Controller rule, per camera per round:**
1. Skip if battery < 25%, CPU load > 0.7 or temperature >= 85 C (detection has priority).
2. Otherwise choose the widest of {4, 2} bits whose predicted training plus upload time fits 60% of the deadline, using the bandwidth measured on that camera's last upload.
3. Skip if even 2 bits would miss the deadline.

Results, mean over 3 seeds, cellular scenario:

| Policy | Test balanced accuracy | Wolf recall | Bear recall | Upload | FL energy |
|---|---|---|---|---|---|
| FP32 | 0.742 | 0.713 | 0.807 | 22.3 MB | 6.7 Wh |
| Fixed 8-bit | 0.749 | 0.731 | 0.809 | 5.59 MB | 4.3 Wh |
| Fixed 4-bit | 0.748 | 0.732 | 0.814 | 2.80 MB | 3.9 Wh |
| Fixed 4-bit, random 70% of clients | 0.726 | 0.633 | 0.804 | 1.93 MB | 2.7 Wh |
| Controller | 0.751 | 0.785 | 0.817 | 1.99 MB | 2.7 Wh |

On constrained links the controller uses 1.59 MB against 2.80 MB for fixed 4-bit and 55% less FL energy, at 0.735 vs 0.742 balanced accuracy; FP32 loses 71% of its uploads to missed deadlines.

What the results support:
- The controller matches or beats FP32 and fixed 8-bit with 64-91% less upload.
- Against fixed 4-bit, accuracy per byte is equal within seed noise; the gain is 29-43% less upload and 31-55% less energy.
- Skipping based on resources is the main effect: random skipping at the same traffic loses 15 points of wolf recall.
- Not yet explained: the controller's wolf recall is 4-6 points above fixed 4-bit while balanced accuracy is unchanged. It is not claimed as a result.
- For head-only training the battery rule has no measurable effect, because FL energy is small next to the detection draw.

## 9. Measured vs assumed

| Quantity | Status |
|---|---|
| Classifier accuracy, int8 accuracy, NPU and CPU latency, graph-cache effect, board idle temperature | Measured |
| FL accuracy vs bit width | Measured in simulation on real data |
| Per-camera energy, uplink bandwidth, training time per sample on the A53, solar input, CPU load pattern | Assumed; Phase 7 replaces these with board measurements |
| On-device label noise (10%) | Assumed |

## 10. Remaining work

| Item | Where it runs | Why it matters |
|---|---|---|
| Choose, export and benchmark the on-board detector (YOLO11n or compact MegaDetector), check NPU op coverage | Mac + board | The camera cannot alert from full frames without it |
| Phase 7: telemetry sampler and energy per inference and per FL round | Board (+ USB power meter if the rails are not exposed) | Replaces the assumed numbers in section 9 |
| Phase 8: node runtime (capture, detect, crop, classify, vote, alert) and the board as a live Flower client | Board | End-to-end demonstration on real hardware |
| QAT for per-tensor EfficientNet-Lite0 | RTX 4090 | Could make the 1.85x faster variant usable |
| Federate the last block, not only the head | RTX 4090 | Head-only limits accuracy most; larger updates also give the controller more to save |
| Explain the controller's wolf recall gain; brown-bear domain-gap test; recall at a fixed false-alarm rate; final plots | Mac | Needed before claiming results in the write-up |

## 11. Reproducing

From `/Volumes/workplace/my-scripts/uni/EC`, with `PYTHONPATH=src`:

```
.venv/bin/python scripts/select_candidates.py
.venv/bin/python scripts/join_md.py
.venv/bin/python scripts/build_crops.py
.venv/bin/python scripts/review_sheets.py            # then --apply after filling verify.csv
.venv/bin/python scripts/train_classifier.py --arch efficientnet_lite0
.venv/bin/python scripts/export_onnx.py --arch efficientnet_lite0
.venv-export/bin/python scripts/quantize_tflite.py --arch efficientnet_lite0
.venv-export/bin/python scripts/eval_tflite.py --arch efficientnet_lite0
.venv/bin/python scripts/extract_features.py
.venv/bin/python scripts/run_fl_sim.py
.venv/bin/python scripts/run_phase6.py --seeds 3 --scenario cellular
.venv/bin/python -m pytest tests                      # 18 tests
```

Data download (LILA label file and MegaDetector results) is described in `IMPLEMENTATION_PLAN.md` Phase 2. Data preparation takes about 30 minutes, training about 40 minutes per model on the Mac, and the FL experiments under a minute each once features are cached.
