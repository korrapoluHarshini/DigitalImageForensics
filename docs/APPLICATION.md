# Digital Image Authentication and Tampering Detection System

A single Streamlit application that brings three independent forensic modules
into one dashboard:

| Module | Purpose | Status |
| --- | --- | --- |
| 1. Image Manipulation | Pixel-level forgery localization with MVSS-Net (ResNet-50, CASIA2.0 checkpoint) | Ready (cross-split leakage quantified; leakage-free retraining harness ready, GPU run pending) |
| 2. AI-Generated Image | Real / AI-generated / uncertain classification through a pluggable detector | Integration placeholder — no weights bundled; training harness ready |
| 3. Chat Screenshot | Visual anomaly screening, OCR transcription and reference comparison | Ready (screening only; reference comparison for message-level differences) |

The application is an academic decision-support demo. It does **not** prove that
an image is authentic or fake, and it does not verify chat messages against
original records unless a trusted reference screenshot is supplied.

Current measured results, including negative findings, are consolidated in
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md). GPU training and dataset
acquisition steps are in [`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md).

---

## 1. Quick start (Windows PowerShell)

```powershell
cd D:\projects\MVSS-Net\MVSS-Net-master

# The project venv is already present. If starting from scratch:
# python -m venv mvss_venv

.\mvss_venv\Scripts\python.exe -m pip install -r requirements-app.txt

.\mvss_venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL printed by Streamlit (normally <http://localhost:8501>).

The MVSS-Net checkpoint must be at `ckpt/mvssnet_casia.pt`. It is loaded lazily
on the first Module 1 analysis, so the dashboard starts without waiting for the
560 MB checkpoint. CPU inference takes roughly 7–10 s per image on the reference
machine; a CUDA build of PyTorch is used automatically when available.

### Dataset location

Module 1 does not need the dataset at runtime. The evaluation scripts and the
About page read the dataset through `DIGITAL_FORENSICS_DATA_ROOT`:

```powershell
$env:DIGITAL_FORENSICS_DATA_ROOT = "D:\ImageForensics_Datasets"
```

The preferred root is `D:\ImageForensics_Datasets`; the previous
`C:\Users\harsh\Documents\DigitalImageForensics` folder is used automatically
when the new variable is unset but the legacy folder exists. The CASIA layout is
`datasets/CASIA2.0_revised/{Au,Tp}` and `datasets/CASIA2.0_Groundtruth`, with
`experiments/dataset_split.csv` and `experiments/localization_pairs.csv`.
Generated evaluation-only data (for example `chat_synthetic_v1`) lives under the
same configurable root and is never committed.

### Optional environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DIGITAL_FORENSICS_DATA_ROOT` | legacy dataset folder | Dataset/experiment root |
| `MVSS_CHECKPOINT` | `ckpt/mvssnet_casia.pt` | MVSS-Net checkpoint |
| `MVSS_THRESHOLD` | `0.4` | Pixel-level localization threshold (validation-selected) |
| `MVSS_IMAGE_THRESHOLD` | `0.42` | Image-level Authentic/Tampered threshold (validation-selected) |
| `MVSS_IMAGE_UNCERTAINTY_MARGIN` | `0.13` | Half-width of the abstention band around the image threshold |
| `MVSS_DEVICE` | `auto` | `auto`, `cpu` or `cuda` |
| `FORENSICS_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `FORENSICS_MAX_IMAGE_PIXELS` | `60000000` | Decompression-bomb guard |
| `AI_MIN_CONFIDENCE` | `0.60` | Minimum top confidence before a class is reported |
| `AI_UNCERTAINTY_MARGIN` | `0.20` | Minimum class-probability margin |

---

## 2. What each page shows

### Home Dashboard
Project description, three module cards with live status, workflow summary and
a statement of the system's limits.

### Image Manipulation Detection
* A prominent **image-level prediction**: *Authentic (Predicted)*, *Tampered
  (Predicted)* or *Uncertain (Insufficient Evidence)*
* A separate **pixel-level badge**: *Pixel-level signal: Possible manipulation*
  or *Pixel-level signal: No strong evidence*
* Original image
* Probability heatmap (JET colour map)
* Binary mask at the pixel localization threshold
* Highlighted/annotated suspicious regions with numbered boxes
* Maximum pixel probability, image decision threshold, uncertainty margin,
  localization threshold, raw thresholded area, retained highlighted area and a
  region table
* A plain-language **"Why this prediction?"** explanation for every image
* Downloadable heatmap, mask, annotated image and text report
* Optional "Detailed patch mode" that blends overlapping 512×512 patches

**Image-level decision vs pixel-level localization.** These are two different
outputs and must not be read as one label:

| Output | Meaning | Controlled by |
| --- | --- | --- |
| Image-level prediction | Authentic / Tampered / Uncertain for the whole image | Image threshold 0.42 ± 0.13 |
| Pixel-level signal | Whether any pixel crossed the localization threshold | Localization threshold 0.40 |

A `Pixel-level signal: Possible manipulation` badge does **not** mean the image
was classified as Tampered. It can appear while the image-level result is still
Uncertain or Authentic. Conversely, an image can be predicted Tampered while the
highlighted mask is empty. The application states this next to both results.

**How the image-level decision works.** The score is the **maximum pixel
probability** of the existing MVSS-Net map. There is no separate authenticity
classifier and no change to the pretrained weights. The image is predicted
Tampered when `score >= image_threshold`, Authentic when
`score <= image_threshold - uncertainty_margin`, and Uncertain in between. The
uncertainty band is an abstention rule, not a calibrated confidence.

**Raw vs retained suspicious area.** The binary mask is thresholded first, then
connected components smaller than
`max(16, round(total_pixels * 0.0002))` are removed. On a 12 MP phone image that
minimum is about 2,400 pixels (a 900×1600 image needs 288 connected pixels). A
handful of isolated above-threshold pixels can therefore be dropped entirely.
The page shows both:

* **Raw thresholded area** — every pixel at/above the localization threshold;
* **Highlighted area (retained)** — only pixels inside components that survived
  the minimum-area filter; this is what the mask and overlay display.

Areas are shown with up to eight decimal places (plus pixel counts) so very small
signals are not rounded to `0.00%`. No region is drawn when no component survives;
the raw pixel count is still reported for transparency.

**Two different thresholds.** The *image decision threshold* classifies the whole
image and is selected on validation authentic + tampered images. The *pixel
localization threshold* (default 0.40) only builds the binary mask and
highlighted overlay; it was selected on validation tampered images for
localization IoU. Changing one does not silently change the other.

### AI-Generated Image Detection
Until a detector is supplied the page shows an explicit integration placeholder
and produces **no scores**. To integrate a model, see
[`models/ai_detector/README.md`](../models/ai_detector/README.md).

Supported backend kinds:

* `torchscript` — exported `*.ts` model
* `onnx` — ONNX model (requires `onnxruntime`)
* `python_adapter` — callable `module:function` or `module:Class`

The wrapper returns `real`, `ai_generated` or `uncertain`. `uncertain` is an
abstention when the top confidence or class margin is too low; it is not a third
learned class.

### Chat Screenshot Analysis
* Visual screening with the same MVSS-Net localization model
* Suspicious-region highlighting and heatmap
* OCR lines and visual blocks (EasyOCR, Tesseract fallback)
* A region↔text-overlap table (co-location cue only)
* **Optional reference comparison**: upload the genuine, unedited screenshot;
  the app aligns the two images, shows a change heatmap, side-by-side view,
  binary change mask and line-level OCR differences
* Downloadable report, highlighted image, heatmap, mask, comparison image,
  change mask and transcript

OCR is used **for readability only**. Without a reference, the module never
claims to recover or verify the original conversation. With a reference it
reports *visible* pixel and text differences relative to that reference, and
still cannot establish which screenshot is the original.

Measured on eight controlled synthetic conversations, this out-of-domain visual
screening flagged every authentic screenshot (false-positive rate 1.0). The
reference comparison recovered all ten injected text edits with no differences
on identical pairs. The page states both facts; see
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) for protocol and caveats.

### Analysis History
Metadata-only log (UTC time, module, image-level result, pixel-level signal,
score, localization threshold, raw/highlighted area, dimensions, content
fingerprint, timing). Screenshots, OCR text, chat content and filesystem paths
are never stored. The legacy finding key is retained for compatibility but is
displayed as "Legacy finding key", separate from the image-level and pixel-level
labels.

### About the Project
Architecture, dataset status, reproducible Module 1 metrics loaded from
`results/evaluation/module1_test_metrics.json`, limitations and citation.

---

## 3. Architecture

```
app.py                              Streamlit UI and page router
modules/
  image_manipulation/               Module 1: MVSS-Net localization
    model.py                        lazy checkpoint loading, single/tiled inference
    preprocessing.py                official BGR / 512 / ImageNet pipeline
    inference.py                    structured result, regions, warnings
    visualization.py                heatmap, mask, overlay, annotation
  ai_generated_detection/           Module 2: pluggable detector
    model.py                        manifest, TorchScript/ONNX/adapter backends
    inference.py                    structured result and abstention policy
    preprocessing.py                configurable normalisation
  chat_tampering/                   Module 3: screenshot pipeline
    inference.py                    visual screening + OCR association
    ocr.py                          EasyOCR/Tesseract abstraction, line grouping
    reference.py                    reference alignment, pixel/text comparison
    preprocessing.py                screenshot sizing
    visualization.py                OCR boxes and combined view
utils/
  config.py                         paths, thresholds, environment overrides
  validation.py                     upload/format/pixel validation
  image_utils.py                    decode, convert, encode, fingerprint
  metrics.py                        pixel/image/classification metrics
  reporting.py                      metadata history and text reports
training/
  train_ai_detector.py              Module 2 training/export harness (TorchScript)
  train_mvss_groupaware.py          Module 1 leakage-free retraining (resumable)
evaluation/
  select_threshold_module1.py       validation-only localization threshold selection
  select_image_threshold_module1.py validation-only image-level threshold + band
  compare_image_level_scores.py     validation-only comparison of image-level scores
  score_authentic_validation.py     resumable authentic validation scoring for calibration
  external_image_test.py            separate external/generalization image runner
  evaluate_module1.py               localization + authentic FPR
  evaluate_module2.py               AI detector metrics
  evaluate_module3.py               chat localization, FPR, OCR quality
  build_synthetic_chat_dataset.py   reproducible synthetic chat-screenshot data
  evaluate_chat_reference.py        reference-comparison metrics
  evaluate_chat_degradation.py      JPEG/rescale stress test for reference comparison
  evaluate_groupaware_checkpoint.py independent leakage-free test evaluation
  analyze_casia_leakage.py          leakage impact + group-aware experimental split
models/
  ai_detector/                      manifest + integration contract
  chat_detector/                    notes for a future dedicated model
tests/                              pytest suite (150 tests)
notebooks/colab_workflow.ipynb      Colab GPU workflow (setup, train, evaluate)
results/evaluation/                 generated metrics artifacts
requirements-app.txt                runtime dependencies
docs/APPLICATION.md                 this document
docs/EVALUATION_REPORT.md           consolidated measured results and caveats
docs/COLAB_RUNBOOK.md               GPU training, resume and dataset-licence steps
```

Every module returns a structured dataclass and catches its own failures. A
missing checkpoint, OCR backend or AI model degrades only the affected page.

---

## 4. Evaluation

All numbers below come from artifacts in `results/evaluation/`. No metrics are
reported for the placeholder Module 2, and thresholds were never tuned on test
data.

### Threshold selection (validation only)

```powershell
.\mvss_venv\Scripts\python.exe evaluation\select_threshold_module1.py
```

* 767 validation tampered images with ground-truth masks
* Rule: maximum mean image IoU, tie-break by mean image F1
* Selected threshold: **0.40**
* Validation metrics at 0.40: mean image IoU 0.5895, mean image F1 0.6615,
  global IoU 0.7085

Artifact: `results/evaluation/validation_threshold_selection.json`.

### Image-level Authentic/Tampered threshold (validation only)

```powershell
.\mvss_venv\Scripts\python.exe evaluation\score_authentic_validation.py --limit 400
.\mvss_venv\Scripts\python.exe evaluation\select_image_threshold_module1.py
```

* 767 cached validation tampered scores + 398 newly scored authentic validation
  images (a deterministic 400-image sample with seed 42; two `.bmp` files were
  rejected by the upload validator)
* Rule: **maximise tampered recall subject to validation FPR ≤ 5%**, then choose
  the uncertainty band that minimises confident errors while keeping at least
  90% coverage
* Selected image-level threshold: **0.42**
* Validation tampered recall: **0.9009** (691/767)
* Validation authentic false-positive rate: **0.0452** (18/398)
* Precision: **0.9746**
* F1: **0.9363**
* Balanced accuracy: **0.9278**
* ROC AUC: **0.9880**
* Uncertainty band: **±0.13**, covering 90.13% of validation images and leaving
  41 confident errors (versus 94 without abstention)

Artifacts: `results/evaluation/module1_image_level_threshold.json`,
`module1_image_level_threshold.csv` and
`module1_image_level_uncertainty.csv`.

The test split was **not** used to choose the threshold or the band. The earlier
12-image test estimate below was produced at the old 0.40 operating point and is
retained unchanged; it is not a validation of the new threshold.

### Image-level score comparison (validation only)

```powershell
.\mvss_venv\Scripts\python.exe evaluation\compare_image_level_scores.py
```

The maximum-pixel rule was compared with mean-top-k and suspicious-area rules on
the same **validation** maps (767 tampered + 398 authentic). AUC is the primary
metric because it does not depend on one cutoff; each method also gets an
operating point selected under the same FPR ≤ 5% rule.

| Score | Validation AUC | Best-FPR≤5% recall | Operating BA |
| --- | --- | --- | --- |
| `max_pixel` (current) | **0.987994** | 0.9022 | **0.9272** |
| mean of top 5 pixels | 0.987984 | 0.9022 | 0.9272 |
| mean of top 10 pixels | 0.987847 | 0.8957 | 0.9252 |
| mean of top 25 pixels | 0.987676 | 0.8957 | 0.9252 |
| raw area at 0.40 | 0.944009 | 0.8892 | 0.9232 |
| retained area at 0.40 | 0.928274 | 0.8670 | 0.9209 |

Averaging top-k pixels does not improve ranking on validation; larger k values
lose recall. Mean-of-top-5 is effectively tied (AUC 0.987984 versus 0.987994),
but switching would require a new threshold calibration for no measurable gain.
Area-based scores rank substantially worse. At the deployed threshold **0.42**,
max-pixel gives recall 0.900913, FPR 0.045226, precision 0.974612, F1 0.936314
and balanced accuracy 0.927843, matching the preserved calibration artifact.

**Decision: keep maximum pixel probability.** The comparison is diagnostic only;
the threshold and uncertainty margin are unchanged. Artifacts:
`results/evaluation/module1_image_level_score_comparison.json` and
`module1_image_level_score_comparison.csv`.

### External / generalization test (not used for tuning)

```powershell
# manifest.csv columns: path,group,label,note
# label is tampered / authentic / unknown; phone photos should normally be unknown
.\mvss_venv\Scripts\python.exe evaluation\external_image_test.py --manifest manifest.csv
```

This runner never changes the thresholds and never feeds these results back into
calibration. It writes per-image decisions, raw/highlighted area, explanations
and a group summary to `results/evaluation/external_generalization_test.json`
and `.csv`.

Run on 2026-11-10 against images available on this machine:

| Group | n | Authentic | Tampered | Uncertain |
| --- | --- | --- | --- | --- |
| Known AI-generated (ChatGPT images) | 2 | 0 | 2 | 0 |
| External UI screenshots | 3 | 2 | 1 | 0 |

The known AI-generated images were both predicted Tampered (max probabilities
0.7717 and 0.9439), but AI-generated imagery is not a validated MVSS-Net class;
this is descriptive only. One external screenshot was predicted Tampered at
0.8685, which shows that the image-level score can react to out-of-distribution
screen content. No genuine phone photographs or independently labelled
manipulated photographs were available on this machine, so that part of the
external test is **unavailable**, not passed.

**Do not treat a phone photograph as authentic just because it came from a
phone.** To run a genuine external test, put the files in a manifest with
`label=unknown` (or `authentic`/`tampered` only when independently known) and run
the command above. Report the result as external/generalization evidence, not as
a replacement for the test-split metrics.

### Module 1 test localization

```powershell
.\mvss_venv\Scripts\python.exe evaluation\evaluate_module1.py --split test --authentic-limit 0
```

* 768 test tampered images (cached prediction maps)
* Global IoU 0.7466
* Mean image IoU 0.5891
* Micro precision 0.8784
* Micro recall 0.8326
* Micro F1 0.8549
* Mean image F1 0.6594

Artifact: `results/evaluation/module1_test_metrics.json`.

**Leakage caveat.** The Phase 0 duplicate audit shows that 687 of 768 test
tampered images (89.45%) belong to duplicate components linked across the
historical splits. A clean subset of only 81 images gives micro IoU 0.7498 but
mean image IoU 0.7796 and tampered recall 0.9877 at 0.42. The clean subset is
small and biased, so these numbers are diagnostic, not a leakage-free benchmark.

```powershell
.\mvss_venv\Scripts\python.exe evaluation\analyze_casia_leakage.py
```

This writes `results/evaluation/casia2_leakage_impact.json` and a separate
leakage-free experimental split `results/manifests/casia2_groupaware_split.csv`
(8,831 train / 1,694 validation / 2,089 test, 0 components spanning splits). The
original CASIA split files are untouched; retraining on the group-aware split is
required before quoting leakage-free test metrics. The retraining harness
(`training/train_mvss_groupaware.py`) and independent evaluator
(`evaluation/evaluate_groupaware_checkpoint.py`) are implemented and
smoke-verified on CPU; no group-aware test metrics exist yet. Do not quote the
old test metrics as leakage-free.

### Module 1 image-level estimate

```powershell
.\mvss_venv\Scripts\python.exe evaluation\evaluate_module1.py --split test --authentic-limit 12
```

* 768 tampered + 12 authentic images, at the old 0.40 operating point
* Tampered recall 0.9232
* Authentic false-positive rate 0.0833 (1 of 12)
* ROC AUC 0.9773

This artifact is preserved as a historical post-hoc estimate and was **not**
used to select the 0.42 image-level threshold.

The authentic FPR is an **estimate from a 12-image sample**, not the full
authentic split. CPU inference is slow; increase `--authentic-limit` for a
tighter estimate and report the sample size.

### Module 2

There is no labelled AI-image dataset in this project and no bundled detector, so
no metrics are reported. The training/export harness is implemented and tested
structurally:

```powershell
.\mvss_venv\Scripts\python.exe training\train_ai_detector.py --help
```

It supports `small_cnn`, `efficientnet_b0`, `resnet18`, `resnet50` and
`convnext_tiny`, deterministic splits, held-out generators, resume and early
stopping, and exports TorchScript plus an app-compatible `manifest.json`.
`tests/test_module2_training.py` verifies the split, export and
`load_backend` contract on a tiny synthetic dataset.

Real training is blocked: this machine is CPU-only and no dataset has been
downloaded. GenImage (CC BY-NC-SA 4.0, academic/non-commercial, >1M images) and
Synthbuster (12.37 GB, CC BY-NC-SA 4.0, 9 generators) were verified as available
on 2026-11-13, but both require explicit approval because of size/licence, and
Synthbuster excludes the real RAISE-1k images. No accuracy or cross-generator
claim is made. Once an approved dataset is available (see
[`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md) §4):

```powershell
.\mvss_venv\Scripts\python.exe training\train_ai_detector.py `
  --real-dir path\to\real --ai-dir path\to\ai `
  --output-dir experiments\module2\run_v1 `
  --backbone efficientnet_b0 --epochs 20 `
  --holdout-generators gen_holdout_a,gen_holdout_b

.\mvss_venv\Scripts\python.exe evaluation\evaluate_module2.py `
  --real-dir path\to\real --ai-dir path\to\ai `
  --manifest experiments\module2\run_v1\manifest.json
```

The evaluator reports accuracy, macro precision/recall/F1, the confusion matrix,
uncertain rate and coverage, grouped by generator. Report held-out generators
separately from generators seen in training.

### Module 3

Evaluation uses a locally generated, privacy-safe synthetic dataset
(`chat_synthetic_v1`): 8 fictional conversations in three themes, 8 authentic
and 8 edited screenshots, 10 injected edits across 5 categories, exact masks and
reference transcripts. All rows are in the `test` split.

```powershell
.\mvss_venv\Scripts\python.exe evaluation\build_synthetic_chat_dataset.py `
  --root D:\ImageForensics_Datasets\chat_synthetic_v1 --conversations 8
```

**Without a reference** (`evaluation/evaluate_module3.py`): localization global
IoU **0.0**, tampered recall 1.0 but authentic FPR **1.0 (8/8)**, OCR mean
character similarity 0.7722. MVSS-Net is out of domain on rendered screenshots
and the page discloses this.

**With a trusted reference** (`evaluation/evaluate_chat_reference.py`): text
change precision/recall **1.0 / 1.0** (10/10), region recall **1.0**, pixel IoU
0.1803, identical authentic pairs with any difference **0/8** (FPR 0.0),
alignment mismatch rate 0.0.

Low pixel IoU is expected: ground-truth masks mark whole edited text boxes,
whereas predicted pixels mark only glyphs that changed. Region recall is the
meaningful measure. These are synthetic results and do not establish real
messaging-app performance; a reference comparison also cannot establish which
screenshot is the original. Full protocol and numbers:
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

**Degradation stress test** (`evaluation/evaluate_chat_degradation.py`): text
recall stays 1.0 at suspect JPEG q85/q60, falls to 0.9 at q40 and 0.7 after a 50%
rescale; region recall stays 1.0 and authentic-pair FPR stays 0.0. Text precision
collapses under one-sided JPEG (0.34 at q85, 0.28 at q60) because compression
noise creates spurious OCR-difference fragments. These are synthetic renders and
must not be read as real-app robustness.

---

## 5. Tests

```powershell
.\mvss_venv\Scripts\python.exe -m pytest tests -q
```

150 tests cover upload validation, preprocessing parity with
`common.transforms.direct_val`, region extraction, image-level decision and
uncertainty logic, raw-vs-retained area diagnostics, explanation text, label
consistency, metrics, module failure isolation, manifest handling, a real
TorchScript backend, the Module 2 training/export harness, the Module 1
group-aware training/evaluation pipeline, OCR line grouping, chat reference
comparison, chat degradation helpers, the synthetic chat dataset and all six
Streamlit pages via `AppTest`.

---

## 6. Security and privacy

* Uploads limited by size, extension, real decoded format and pixel count.
* Pillow verifies each file; corrupted or mislabelled files are rejected.
* Uploaded content is never executed and never written to disk by the app.
* Display filenames are sanitised; absolute paths are not shown.
* No screenshots or OCR text are persisted or logged.
* History stores aggregate metadata and a truncated content hash only.
* Truncated file hashes and model scores must not be presented as proof.

---

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Checkpoint not found" in sidebar | Place `mvssnet_casia.pt` in `ckpt/` or set `MVSS_CHECKPOINT` |
| Module 1 is slow | CPU inference is ~7–10 s/image on the reference machine; install a CUDA PyTorch build or leave detailed patch mode off |
| AI page shows placeholder | No `models/ai_detector/manifest.json`; see the integration contract |
| OCR unavailable | `pip install easyocr` (first run downloads model files) or install Tesseract + `pytesseract` |
| `onnxruntime` error | `pip install onnxruntime` only if an ONNX detector is used |
| Out-of-memory on tall screenshots | Leave detailed patch mode off; tile count is capped |
