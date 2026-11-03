# Resource-Aware Federated Learning with Dynamic Quantization: Wildlife Alerting on i.MX 8M Plus

Research notes, 2026-10-25.

## 1. Hardware constraints (i.MX 8M Plus EVK)

| Unit | Capability | Implication |
|---|---|---|
| NPU (VeriSilicon VIP8000) | 2.3 TOPS, inference only, int8 (also int16) | Runs the quantized detector/classifier. Cannot train. |
| CPU | 4x Cortex-A53 @ 1.8 GHz | All on-device training (FL local steps) runs here, in FP32. |
| GPU | GC7000UL (OpenCL/Vulkan) | Fallback for some ops; slower than NPU for CNNs. |
| ISP | 375 MPixel/s | Camera pipeline without CPU cost. |

Known NPU issues reported on the NXP community and TensorFlow tracker:
- YOLOv5 int8 TFLite partially falls back to CPU because `RESIZE_NEAREST_NEIGHBOR` with `half_pixel_centers=true` is refused (NNAPI path). Use the VX delegate (`libvx_delegate.so`) rather than NNAPI and check the delegated-op count in `benchmark_model` output.
- Ultralytics int8 export can leave int64 ops in the graph, which the NPU does not run.
- The first inference has a long warm-up (graph compilation); see NXP AN12964. Cache the compiled graph (`VIV_VX_ENABLE_CACHE_GRAPH_BINARY=1`, `VIV_VX_CACHE_BINARY_GRAPH_DIR`).
- Per-tensor int8 quantization is generally faster than per-channel on this NPU (from memory; verify in NXP UG10166, i.MX Machine Learning User's Guide).

Consequence for the design: "dynamic quantization" cannot mean switching the NPU between arbitrary bit widths. The NPU runs int8 (or int16). The adjustable knobs are:
1. bit width of the federated model updates sent upstream (2/4/8/16/32-bit),
2. which model variant runs for inference (e.g. int8 nano on NPU vs. int16 or larger model), selected from resource state,
3. which layers are trained locally (head only vs. last blocks), selected from CPU/thermal/energy budget.

## 2. Public datasets with dangerous species

Counts below come from streaming LILA's combined label file (`lila_image_urls_and_labels.csv`, 21.7 GB uncompressed, release dated 2026-09-02). Script: `lila_count_dangerous.py`; full output: `lila_counts.txt`. Only rows with >=200 images per dataset are listed.

| Species group | Images | Main sources (images / sequences / locations) |
|---|---|---|
| Bear | 92,724 | AMMonitor 46,180 / 4,776 / 439; NACTI 28,770 / n.a. / 277; WSU Lynx 8,281 / 1,908 / 481; Idaho 3,180 / 566 / 106; Oregon Critters 2,573; SWG 1,624; WCS 1,118; ENA24 893 |
| Lynx / bobcat | 77,799 | Felidae CF 31,278; NACTI 25,434; Caltech 7,984; WSU Lynx 6,184; AMMonitor 4,082 |
| Puma | 33,790 | NACTI 14,756; WCS 6,823; Felidae CF 6,705; WSU Lynx 3,244; Idaho 1,797 |
| Wolf | 5,537 | Idaho 4,697 / 640 / 104; WSU Lynx 840 / 216 / 97 |
| Wild boar | 396,766 | SWG 237,353; NACTI 139,439 |
| Big cats (lion, leopard, jaguar, tiger) | 43,715 | Snapshot Serengeti 22,816; WCS 10,655; Snapshot Safari 2024 7,587 |
| Spotted hyena | 36,985 | Snapshot Serengeti 21,998; Snapshot Safari 2024 9,602 |
| Coyote (useful hard negative for wolf) | 102,584 | WSU Lynx, Felidae CF, NACTI, Caltech, AMMonitor |

Species breakdown inside the predator groups: American black bear 89,982; bobcat 76,074; grey wolf 5,537; Canada lynx 1,725; Asiatic black bear 1,232; sun bear 941; spectacled bear 462. **There are zero brown bear (Ursus arctos), zero Eurasian lynx and zero wolverine images on LILA.**

European sources (not on LILA):
- **DeepFaune** (France, 26 taxa incl. bear, wolf, lynx). Model weights public under CC BY-NC-SA 4.0; images only "available upon request to each partner".
- **TrapperAI v02.2024** (OSCF; Poland, Germany, Sweden, Austria, Switzerland; 401,458 images). YOLOv8-m detector for 18 species incl. grey wolf and Eurasian lynx, no bear class. Weights on Hugging Face (GPL-3.0); training data not public.
- **CzechLynx** (Scientific Data, 2026): 39,760 images of Eurasian lynx with masks and identities. Useful for lynx only.
- Swedish four-carnivore set (bear, lynx, wolf, wolverine) used in arXiv:2310.08316; no public download found.
- Brown bear supplements: COCO `bear` and Open Images `Brown bear` classes have human boxes but are mostly zoo/web photos, not camera-trap imagery (domain gap: viewpoint, IR night shots, motion blur).

## 3. Quality assessment

| Issue | Where | Effect | Mitigation |
|---|---|---|---|
| Sequence-level labels | Idaho, WSU Lynx, AMMonitor, SWG, WCS | A frame labeled "wolf" may contain no wolf | Use LILA's published MegaDetector results to get boxes and drop frames without a detection above ~0.2 |
| Few human-drawn boxes for target species | Only ENA24 (bear 893), Caltech (no bear/wolf), Missouri, Nkhotakota | No clean detection ground truth for wolf | MegaDetector pseudo-boxes + manual check of a few hundred wolf frames as the test set |
| Wolf is rare | 5,537 images but only 856 sequences, 2 datasets, both in the US Pacific Northwest | Effective sample size is the sequence count; strong location bias | Count sequences, not images; augment; treat wolf vs. coyote vs. dog as the hard case |
| Brown bear absent | All of LILA | A North-American-trained model will see a different bear in Europe | Fine-tune on brown bear images from COCO/Open Images/iNaturalist or partner data; this is a real use case for on-device FL |
| Missing sequence ids / unknown annotation level | NACTI | Cannot do sequence-aware splits | Split by `location_id` |
| High empty rate | Idaho ~70.5% empty | Class imbalance, many false triggers | Detector-first pipeline filters empties before classification |
| Location leakage | All camera-trap data | Random splits overstate accuracy (Beery et al., "Recognition in Terra Incognita") | Hold out whole locations |
| Licences | Idaho: no resale, cite IDFG. DeepFaune weights: non-commercial. TrapperAI: GPL-3.0 | Fine for coursework; check before redistribution | Record licence per source in the data manifest |

A useful property: `location_id` gives a natural federated partition. Each camera location is one client, with its own species mix, lighting and background. This produces realistic non-IID data without synthetic Dirichlet splits.

## 4. Proposed system design

Pipeline per camera node:
1. Motion/PIR trigger, then ISP capture.
2. Stage 1 on NPU: small int8 detector (YOLOv8n/YOLO11n or MegaDetector v6 compact variant) with classes animal / person / vehicle at 320-640 px.
3. Stage 2 on NPU: int8 crop classifier (MobileNetV3 or EfficientNet-Lite0) with classes e.g. bear, wolf, lynx, wild boar, other-animal.
4. Temporal voting across the burst/sequence before alerting (the class-tracking idea in arXiv:2310.08316), with per-class thresholds biased towards recall for dangerous classes.
5. Alert over the uplink (LoRa/LTE) with a crop thumbnail.

Federated learning:
- Framework: Flower. It has an embedded-devices example and runs on aarch64 Linux.
- Federate only the classifier head (or the last block) of stage 2; train in FP32 on the A53 cores, then re-quantize to int8 for the NPU with post-training quantization on a small local calibration set. Training the detector on the A53 is not practical.
- Labels on device: confirmed or rejected alerts from rangers, plus high-confidence pseudo-labels from temporal voting. Without a label source FL has nothing to learn from; state this assumption explicitly.
- Testbed: the EVK is one real client; the other clients are simulated on a laptop with Flower simulation, partitioned by LILA `location_id`.

Resource-aware dynamic quantization controller (per round, per client):
- Inputs: CPU load and temperature (`/sys/class/thermal/thermal_zone*/temp`), free memory, measured uplink bandwidth, (simulated) battery state of charge.
- Outputs: update bit width b in {2, 4, 8, 16, 32} with stochastic quantization, number of local steps, trainable layer set, participate or skip.
- Server: aggregate with dequantized updates; weight by sample count; optionally error feedback for low-bit clients.
- Baselines: FedAvg FP32; fixed 8-bit FedPAQ-style; adaptive schemes from the literature (AdaQuantFL, Jhunjhunwala et al. 2021; EA-AQF, Cluster Computing 2026; resource-aware mixed-precision via federated supernets, PMLR v306).

Metrics:
- Detection/classification: per-class recall for dangerous species at a fixed false-alarm rate, on held-out locations.
- Edge: inference latency (NPU vs. CPU), energy per inference and per FL round (measure with the EVK's power rails or an external USB meter), peak RAM.
- Communication: bytes uploaded per round and accuracy vs. cumulative bytes.

## 5. Sources

- LILA BC datasets and taxonomy mapping: https://lila.science/datasets, https://lila.science/taxonomy-mapping-for-camera-trap-data-sets/
- Idaho Camera Traps: https://lila.science/datasets/idaho-camera-traps/
- ENA24-detection: https://lila.science/datasets/ena24detection/
- DeepFaune: https://link.springer.com/article/10.1007/s10344-023-01742-7
- TrapperAI: https://huggingface.co/OSCF/TrapperAI-v02.2024
- CzechLynx: https://www.nature.com/articles/s41597-026-06853-9
- Swedish carnivores tracking: https://arxiv.org/abs/2310.08316
- Flower embedded devices: https://flower.ai/docs/examples/embedded-devices.html
- NXP i.MX ML User's Guide (UG10166): https://www.nxp.com/docs/en/user-guide/UG10166.pdf
- NPU warm-up (AN12964): https://www.mouser.com/pdfDocs/AN12964.pdf
- YOLOv5 int8 NPU fallback: https://github.com/tensorflow/tensorflow/issues/53056
- Resource-aware mixed-precision FL: https://proceedings.mlr.press/v306/ma26ae.html
- EA-AQF: https://link.springer.com/article/10.1007/s10586-026-06503-0
