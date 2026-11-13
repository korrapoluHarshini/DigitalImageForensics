# Evaluation Report (current state)

This report consolidates every measured result in the repository. It is written
to be defensible: numbers come from committed/generated artifacts, unavailable
evidence is labelled unavailable, and small or synthetic samples are not
presented as real-world performance.

Generated artifacts live in `results/evaluation/` and `results/manifests/`.
Re-running an experiment overwrites only its own artifact; nothing in this
report was tuned on the test split.

---

## 1. Summary

| Module | Evidence | Headline result | Status |
| --- | --- | --- | --- |
| 1 - Manipulation localization | CASIA2.0 test split (768 tampered images, cached maps) | Global IoU **0.7466**, mean image IoU **0.5891**, micro F1 **0.855** | Measured, with leakage caveat |
| 1 - Image-level | Validation (767 tampered + 398 authentic) | AUC **0.9880**, recall **0.901** at threshold 0.42, FPR **0.045** | Measured |
| 1 - Leakage-free retraining | Group-aware split, harness only | **No metrics yet** | Smoke-verified CPU pipeline; GPU run pending (Colab runbook) |
| 2 - AI-generated detection | None | **No metrics** | Harness implemented; dataset approval pending |
| 3 - Chat screening (no reference) | 8 synthetic conversations | Localization IoU **0.0**, authentic FPR **1.0** | Measured; out-of-domain failure |
| 3 - Chat reference comparison | 8 synthetic conversations / 10 edits | Text-change precision/recall **1.0 / 1.0**, authentic-pair FPR **0.0** | Measured on controlled synthetic data |
| 3 - Degradation stress test | Same 8 pairs, JPEG/rescale | Text recall **1.0 -> 0.7** as degradation grows; authentic FPR **0.0** | Measured on synthetic data |

---

## 2. Module 1 — CASIA2.0 localization

Artifact: `results/evaluation/module1_test_metrics.json` (threshold 0.40,
validation-selected).

| Metric | Value |
| --- | --- |
| Test tampered images | 768 |
| Global (micro) IoU | 0.7466 |
| Mean image IoU | 0.5891 |
| Micro precision | 0.8784 |
| Micro recall | 0.8326 |
| Micro F1 | 0.8549 |
| Mean image F1 | 0.6594 |

Image-level estimate (historical, 12 authentic test images, old 0.40 operating
point): tampered recall 0.9232, authentic FPR 0.0833, ROC AUC 0.9773. This is a
post-hoc sample estimate and was **not** used to select the deployed 0.42
threshold.

### 2.1 Cross-split leakage

Artifact: `results/evaluation/casia2_leakage_impact.json`.

The Phase 0 duplicate audit links CASIA2.0 samples across the historical
train/validation/test splits using exact, decoded, pHash and dHash evidence.
Unioning those groups gives 1,733 duplicate components; one very large component
(568 samples) and many small ones span splits.

| Quantity | Value |
| --- | --- |
| Test images in a cross-split duplicate component | **687 / 768 (89.45%)** |
| Clean test images (component confined to test) | **81 (10.55%)** |
| Duplicate components linked across splits | 1,141 |
| Full micro IoU | 0.7492 |
| Clean-subset micro IoU | 0.7498 |
| Full mean image IoU | 0.5891 |
| Clean-subset mean image IoU | **0.7796** |
| Tampered recall @0.42, full | 0.9141 |
| Tampered recall @0.42, clean | 0.9877 |

**Interpretation.** Micro IoU is essentially unchanged because pixel counts are
dominated by large masks; the per-image mean jumps from 0.589 to 0.780 because
the clean subset excludes 142 images whose predictions were nearly empty. The
clean subset is small (81 images) and biased toward images that happen to have
no detected duplicate, so it is **not** a leakage-free benchmark. The honest
conclusion is: the historical test metrics are inflated by cross-split
duplicates and must be reported as such until the model is retrained.

Per-image counts were recovered algebraically from the stored IoU/precision/
recall and measured ground-truth mask areas (`TP = recall × GT`,
`FN = GT − TP`, `FP = TP/IoU − TP − FN`). Rounding causes small differences
(recovered micro precision 0.8820 vs reported 0.8784). 142 images have TP = 0,
so their false positives cannot be separated from true negatives.

### 2.2 Leakage-free experimental split

Artifact: `results/manifests/casia2_groupaware_split.csv`
(summary: `casia2_groupaware_split_summary.json`).

* 8,831 train / 1,694 validation / 2,089 test rows
* 7,441 duplicate components
* **0 components span experimental splits**
* Assignment is deterministic (seeded hash of component ID) and class-balanced
  approximately; mixed-class components are assigned as a whole.
* The original CASIA split files are untouched. Retraining on this split is
  required before quoting leakage-free test metrics; **no such metrics exist
  yet**.

### 2.3 Retraining harness (implemented, not yet run to completion)

`training/train_mvss_groupaware.py` is ready for a GPU run:

* Loads the manifest and group-aware split, resolves paths against
  `--data-root`, and excludes the 17 tampered rows without a readable mask
  (counted and listed in `run.json`).
* Reuses the existing MVSS-Net architecture and `EdgeGenerator` target parity.
* Initialises read-only from `ckpt/mvssnet_casia.pt`, never writes to `ckpt/`.
* Loss weights `0.16 seg + 0.80 edge + 0.04 cls`; AdamW, cosine schedule, AMP,
  gradient clipping.
* Preprocessing is BGR + ImageNet statistics, exactly matching
  `modules.image_manipulation.preprocessing.preprocess_bgr` and the application.
* Model selection uses validation mean image IoU at a fixed 0.5 threshold. The
  test split is never read during training.
* Writes resumable `last.pt`/`best.pt`, a plain `mvssnet_groupaware.pt` for the
  app, `training_log.csv`, `run.json` and `split_used.csv`.

A CPU smoke run (4 train / 3 validation images, 256 px) completed every epoch
and wrote all six artefacts. Its metrics are meaningless by design and are not
reported here.

`evaluation/evaluate_groupaware_checkpoint.py` evaluates a trained checkpoint on
the group-aware test split using the fixed 0.40 pixel and 0.42 image-level
thresholds, and records checkpoint/split hashes and threshold provenance in its
JSON. It also refuses to hide sampling: a limited run is marked `sampled` and
must not be quoted as a full-split result. Neither script has been run on a GPU;
see `COLAB_RUNBOOK.md` for the commands and the resumable workflow.

**No group-aware test metrics are claimed in this report yet.**

---

## 3. Module 2 — AI-generated image detection

Artifact: none. No detector weights are bundled and no labelled AI-image
dataset was available.

Implemented: `training/train_ai_detector.py`, a dataset-agnostic harness that
trains a small CNN or torchvision backbone (EfficientNet-B0, ResNet-18/50,
ConvNeXt-Tiny), performs a deterministic split, supports held-out generators,
resumes from checkpoints, and exports TorchScript plus an app-compatible
`manifest.json`.

Validated by `tests/test_module2_training.py` on a tiny synthetic dataset:
split determinism, held-out-generator exclusion, checkpoint/export, manifest
schema and `load_backend` TorchScript prediction all pass. The exported model is
verified to load through `modules.ai_generated_detection.model.load_backend`.

**No accuracy, AUC or cross-generator claim is made.** Real training is blocked
by an explicit approval gate, not by missing information:

* CPU-only machine (no CUDA); full training is infeasible locally.
* The dataset catalogue was re-checked on 2026-11-13. GenImage (official) is
  CC BY-NC-SA 4.0, academic/non-commercial, over one million images; its
  unofficial HuggingFace mirror `jzousz/GenImage` accounts for ~678 GB. The
  official release is distributed through Google Drive/Baidu.
  Synthbuster (Zenodo `10.5281/zenodo.10066460`) is 12.37 GB, CC BY-NC-SA 4.0,
  and contains 9 generators x 1000 images, but **not** the real RAISE-1k images.
* No dataset has been downloaded. The project rules require approval before a
  large or licence-restricted download, and RAISE-1k needs its own licence
  check. Until then the application keeps its honest `unavailable` placeholder.

Recommended Module 2 experiment once approved: train on Synthbuster with two
generators held out entirely (`--holdout-generators`), then report
in-distribution and held-out-generator metrics separately. See
`COLAB_RUNBOOK.md` §4 for the exact commands and expected folder layout.

When an approved dataset exists:

```powershell
.\mvss_venv\Scripts\python.exe training\train_ai_detector.py `
  --real-dir D:\ImageForensics_Datasets\module2\real `
  --ai-dir D:\ImageForensics_Datasets\module2\ai `
  --output-dir experiments\module2\run_v1 `
  --backbone efficientnet_b0 --epochs 20 `
  --holdout-generators gen_holdout_a,gen_holdout_b

.\mvss_venv\Scripts\python.exe evaluation\evaluate_module2.py `
  --real-dir D:\ImageForensics_Datasets\module2\real `
  --ai-dir D:\ImageForensics_Datasets\module2\ai `
  --manifest experiments\module2\run_v1\manifest.json
```

Report per-generator metrics separately; a generator present in training is not
a held-out-generator result.

---

## 4. Module 3 — Chat screenshot tampering

Dataset: `D:\ImageForensics_Datasets\chat_synthetic_v1`, generated locally by
`evaluation/build_synthetic_chat_dataset.py`.

* 8 fictional conversations, three themes (iOS light, Android dark, WhatsApp
  light), 16 manifest rows (8 authentic, 8 edited), all in the `test` split.
* 10 semantic edits across 5 categories (header name, amount, message text,
  date, time), with pixel-level masks marking the edited text boxes and expected
  transcripts.
* Locally rendered with Pillow; no real conversation was collected or
  downloaded. Licence label: CC0-1.0 (generated synthetic data).

### 4.1 Without a reference screenshot

Artifact: `results/evaluation/module3_metrics.json`.

| Metric | Value |
| --- | --- |
| Localization global IoU | **0.000** |
| Mean image IoU | 0.000 |
| Tampered recall @0.40 | 1.000 |
| Authentic false-positive rate (8 images) | **1.000 (8/8)** |
| Image-level accuracy | 0.500 |
| OCR mean character similarity | 0.7722 |

**Interpretation.** MVSS-Net is out of domain on rendered UI screenshots. It
scores almost every screenshot as manipulated, and its masks do not overlap the
edited text boxes at all. The high tampered recall is therefore meaningless: a
detector that always says "tampered" achieves the same recall. The application
labels this visual signal as a triage aid, not evidence, and reports the measured
false-positive rate.

### 4.2 With a trusted reference screenshot

Artifact: `results/evaluation/module3_reference_metrics.json`
(per-conversation: `..._per_conversation.csv`).

Protocol: align the authentic render to its edited copy with ECC, threshold
per-pixel differences at 25/255 before morphology, and diff OCR transcripts.
Thresholds were fixed before evaluation.

| Metric | Value |
| --- | --- |
| Conversations | 8 |
| Expected text edits | 10 |
| Detected text edits / matched | 10 / **10** |
| Text-change precision | **1.000** |
| Text-change recall | **1.000** |
| Region recall (IoU ≥ 0.10 or centre-in-box) | **1.000** |
| Region precision | 0.556 |
| Pixel IoU | 0.1803 |
| Pixel precision | 0.760 |
| Pixel recall | 0.191 |
| Authentic identical pairs with any difference | **0 / 8** (FPR 0.0) |
| Alignment mismatch rate | 0.0 |

Per category: amount 4/4, date 1/1, message text 2/2, header name 2/2, time 1/1.

**Interpretation.** With a genuine reference, the pipeline recovered every
injected change in this controlled set and produced no differences on identical
pairs. Pixel IoU is low *by construction*: ground-truth masks mark whole edited
text boxes, while the predicted mask marks only pixels that actually changed.
Region recall, not pixel IoU, is the meaningful localization measure here.
Region precision is 0.56 because one edit can produce several adjacent change
components (e.g. glyph strokes); this is a presentation issue, not a missed
edit.

These are synthetic, controlled results. They do **not** establish performance
on real messaging-app screenshots, where re-rendering, compression, theme
changes and OCR errors are much more varied. A reference comparison also cannot
establish which screenshot is the original.

### 4.3 Degradation stress test

Artifact: `results/evaluation/module3_degradation_metrics.json`
(per-condition: `..._per_condition.csv`). The same 8 authentic/edited pairs are
re-compared after controlled degradation. Only the suspect, or both images, are
degraded; ground-truth masks and expected text changes are unchanged.

| Condition | Text recall | Text precision | Region recall | Pixel IoU | Authentic FPR |
| --- | --- | --- | --- | --- | --- |
| Clean | 1.000 | 1.000 | 1.000 | 0.1803 | 0.000 |
| Suspect JPEG q85 | 1.000 | 0.3448 | 1.000 | 0.1607 | 0.000 |
| Suspect JPEG q60 | 1.000 | 0.2778 | 1.000 | 0.1156 | 0.000 |
| Suspect JPEG q40 | 0.900 | 0.2727 | 1.000 | 0.1009 | 0.000 |
| Suspect rescale 50% | 0.700 | 0.1346 | 1.000 | 0.1276 | 0.000 |
| Both JPEG q60 | 1.000 | 1.000 | 1.000 | 0.1826 | 0.000 |

**Interpretation.** Text-change *recall* is fairly robust: JPEG q85/q60 keep all
10 edits and q40 loses one. Halving the suspect resolution loses three of ten
edits. Text-change *precision* collapses under one-sided JPEG because compression
noise creates many spurious OCR-difference fragments; the clean and both-JPEG
conditions stay at 1.0 because the noise is shared by both images. Region recall
stays 1.0 throughout, and identical pairs still produce no differences. These
are synthetic renders; real app re-rendering is not represented.

---

## 5. Reproduction commands

```powershell
# Module 1 leakage impact + leakage-free experimental split
.\mvss_venv\Scripts\python.exe evaluation\analyze_casia_leakage.py

# Module 1 group-aware retraining (GPU recommended) and independent evaluation
.\mvss_venv\Scripts\python.exe training\train_mvss_groupaware.py `
  --output-dir experiments\module1\groupaware_v1 --device cuda --epochs 40 --batch-size 4
.\mvss_venv\Scripts\python.exe evaluation\evaluate_groupaware_checkpoint.py `
  --checkpoint experiments\module1\groupaware_v1\mvssnet_groupaware.pt `
  --split test --output results\evaluation\module1_groupaware_test_metrics.json

# Module 2 harness (requires a real dataset; refuses placeholder data)
.\mvss_venv\Scripts\python.exe training\train_ai_detector.py --help

# Module 3 synthetic dataset + evaluations
.\mvss_venv\Scripts\python.exe evaluation\build_synthetic_chat_dataset.py `
  --root D:\ImageForensics_Datasets\chat_synthetic_v1 --conversations 8 --force
.\mvss_venv\Scripts\python.exe evaluation\evaluate_chat_reference.py `
  --dataset D:\ImageForensics_Datasets\chat_synthetic_v1
.\mvss_venv\Scripts\python.exe evaluation\evaluate_chat_degradation.py `
  --dataset D:\ImageForensics_Datasets\chat_synthetic_v1
.\mvss_venv\Scripts\python.exe evaluation\evaluate_module3.py `
  --manipulated-dir D:\ImageForensics_Datasets\chat_synthetic_v1\manipulated `
  --masks-dir D:\ImageForensics_Datasets\chat_synthetic_v1\masks `
  --authentic-dir D:\ImageForensics_Datasets\chat_synthetic_v1\authentic `
  --ocr-references D:\ImageForensics_Datasets\chat_synthetic_v1\ocr_refs
```

---

## 6. Threats to validity and honest limitations

1. **Leakage.** 89.45% of historical test tampered images belong to duplicate
   components that cross splits. The reported test metrics are therefore
   optimistic; only retraining on the group-aware split can quantify the true
   gap.
2. **Small clean subset.** 81 images, selected by absence of detected
   duplicates, not randomly. Not a benchmark.
3. **Round-trip metric recovery.** Leakage analysis uses stored rounded
   per-image metrics, not re-inference. Differences up to ~0.004 in precision
   are visible and documented.
4. **Module 2 has no evidence.** The harness is tested structurally only. No
   real detector performance is claimed.
5. **Module 3 data is synthetic.** Rendered screenshots lack real capture,
   recompression and app-version artefacts. 100% text recall is not a
   real-world expectation.
6. **OCR dependence.** Text-change results depend on the installed EasyOCR
   version and its errors; one edit produced two adjacent OCR fragments, which
   the pipeline merges.
7. **Threshold provenance.** Module 1 thresholds (0.40 localization, 0.42 image
   level, ±0.13 band) were selected on validation only. Module 3 uses a fixed
   pixel-difference threshold (25/255) declared before evaluation.
8. **No device-level evaluation.** No labelled DSLR/mirrorless/Android/iPhone
   data was available, so per-device claims are unavailable.
9. **Out-of-domain visual screening.** The no-reference Module 3 path fails on
   authentic synthetic screenshots (FPR 1.0). The application discloses this.
10. **Group-aware split is not the historical benchmark.** The experimental
    split moves whole duplicate components, so its test rows differ from the
    original CASIA test split. Retrained metrics may only be compared with each
    other, not with the old leakage-affected numbers as if the benchmark were
    unchanged. No retrained metrics exist yet.
11. **Degradation test is synthetic.** JPEG/rescale stress testing of rendered
    screenshots does not reproduce messaging-app re-rendering, colour
    management, or unknown resampling. The text-precision collapse under
    one-sided JPEG is a warning, not a measured real-world rate.
12. **Module 2 dataset gate.** GenImage and Synthbuster are academic,
    non-commercial datasets and GenImage is hundreds of gigabytes. Neither has
    been downloaded; no Module 2 metric exists.
