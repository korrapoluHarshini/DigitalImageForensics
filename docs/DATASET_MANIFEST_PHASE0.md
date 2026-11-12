# Phase 0 — Unified Dataset Manifest and Evaluation Tooling

**Status:** Phase 0 tooling only. No dataset was downloaded, no model was trained or
fine-tuned, no checkpoint was replaced, no threshold was changed, and the deployed
preprocessing was not modified.

This document is additive. It describes the manifest, duplicate audit and per-source
evaluation tools under `dataset_manifest/` and the command-line wrappers under
`evaluation/`. It does not replace the existing CASIA loader, evaluator or Streamlit
application.

---

## 1. Purpose and non-goals

The long-term objective is to measure and eventually improve Module 1 on DSLR,
mirrorless, Android and iPhone photographs without losing CASIA 2.0 performance.
Phase 0 prepares the data and evaluation plumbing needed for that work:

* one schema for heterogeneous datasets;
* explicit grouping to prevent train/test leakage;
* read-only duplicate and near-duplicate audits;
* per-source image- and pixel-level evaluation from supplied predictions;
* a manifest-compatible representation of the preserved CASIA split.

Phase 0 does **not** claim that real-world accuracy has improved. It does not
establish a new threshold, retrain MVSS-Net, resolve the known BGR/RGB
preprocessing difference, or evaluate on devices for which labelled data is not
available.

---

## 2. Preserved artefacts

The following were treated as read-only:

| Artefact | Location |
| --- | --- |
| MVSS-Net checkpoint | `ckpt/mvssnet_casia.pt` |
| CASIA split | `<data_root>/experiments/dataset_split.csv` |
| Tampered/mask pairing | `<data_root>/experiments/localization_pairs.csv` |
| CASIA images | `<data_root>/datasets/CASIA2.0_revised/{Au,Tp}` |
| CASIA masks | `<data_root>/datasets/CASIA2.0_Groundtruth` |
| Existing test metrics | `results/evaluation/module1_test_metrics.json` |
| Thresholds | localization `0.40`, image `0.42`, uncertainty `+/-0.13` |
| Application | `app.py` and all three modules |

The default data root is
`C:\Users\harsh\Documents\DigitalImageForensics`, matching
`utils/config.py`. It can be overridden with `DIGITAL_FORENSICS_DATA_ROOT`.

---

## 3. Unified manifest schema

The canonical format is CSV (JSON Lines and JSON arrays are also supported). The
writer always emits this field order:

```
sample_id, dataset, source_group, split, class_name, image_path, mask_path,
pair_id, group_id, manipulation_type, device_make, device_model, capture_type,
width, height, format, mask_available, mask_type, mask_quality, license,
provenance_url, sha256, phash, notes,
dhash, decoded_sha256, file_size, validation_status, validation_errors,
pixel_training_eligible
```

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Globally unique row identifier. Use a dataset prefix and a stable digest. |
| `dataset` | string | Dataset name, e.g. `CASIA2.0`, `IMD2020`, `custom_phone_2026`. |
| `source_group` | enum | `dslr_mirrorless`, `android`, `iphone`, `other_mobile` or `unknown`. |
| `split` | enum | `train`, `validation`, `test` or `unassigned`. |
| `class_name` | enum | `Authentic`, `Tampered` or `Unknown`. |
| `image_path` | string | Path relative to the data root, or an absolute path. |
| `mask_path` | string | Ground-truth mask path; empty for authentic images. |
| `pair_id` | string | Optional image/mask or original/manipulated pairing key. |
| `group_id` | string | Shared by all originals, manipulated versions, masks and derived variants. |
| `manipulation_type` | string | Dataset-specific type; `unspecified` is allowed. Do not invent a type. |
| `device_make` | string | Manufacturer when verified. Empty if unknown. |
| `device_model` | string | Model when verified. Empty if unknown. |
| `capture_type` | string | `dslr`, `mirrorless`, `smartphone`, etc.; `unknown` is allowed. |
| `width`, `height` | integer | Decoded image dimensions. |
| `format` | string | Lower-case file extension/format, e.g. `jpg`, `png`, `tif`. |
| `mask_available` | boolean | Whether a valid mask is present. |
| `mask_type` | string | e.g. `binary_png`; empty when no mask. |
| `mask_quality` | string | e.g. `groundtruth`, `manual`, `auto`, `synthetic`; do not overstate. |
| `license` | string | Dataset/licence label. Re-verify before redistribution. |
| `provenance_url` | string | Official source page when known; empty for local-only provenance. |
| `sha256` | string | SHA-256 of the file bytes. |
| `phash` | string | 64-bit DCT perceptual hash, 16 hex characters. |
| `notes` | string | Provenance, limitations, filename family or other context. |
| `dhash` | string | 64-bit difference hash, 16 hex characters. |
| `decoded_sha256` | string | SHA-256 over decoded EXIF-oriented RGB pixels and dimensions. |
| `file_size` | integer | File size in bytes. |
| `validation_status` | string | `ok`, `warning` or `invalid`. |
| `validation_errors` | string | Semicolon-separated issues; empty for clean rows. |
| `pixel_training_eligible` | boolean | True only for a Tampered training row with a valid mask. |

A JSON Schema is provided at `docs/manifest_schema.json`. The illustrative
records in `docs/manifest.example.csv` use fake paths; they are **not** dataset
evidence.

### Identity and grouping rules

* `sample_id` is globally unique. It is not the filename.
* Original, manipulated version, mask and derived variant rows share `group_id`.
  A mask is normally referenced by `mask_path`, but a separate mask row is also
  permitted if its `group_id` matches.
* `pair_id` links an image to its mask or source pair; it does not replace
  `group_id`.
* A single `group_id` must not appear in more than one split. The validator
  reports `group_split_leakage` as an error.
* Tampered training rows without valid masks are retained but marked invalid and
  `pixel_training_eligible=false`. They are never silently dropped.
* Phone photographs without independently established authenticity are
  `class_name=Unknown` and are not counted as authentic ground truth.

### Path handling

Image and mask paths are stored relative to the data root whenever they are
inside it. Relative paths are resolved read-only against `--data-root`.
Missing, unreadable or ambiguous references are reported. The CASIA adapter
retains every split row even when a mask is missing.

---

## 4. CASIA 2.0 compatibility

`evaluation/build_casia_manifest.py` reads the preserved split and pair CSVs and
writes a manifest. It does not rewrite either CSV or any image/mask.

Verified from the preserved local files on 2026-11-12:

* 12,614 split rows: 8,829 train, 1,892 validation, 1,893 test.
* 7,491 authentic and 5,123 tampered rows.
* 17 tampered rows have no mask mapping in `localization_pairs.csv`:
  15 train, 1 validation and 1 test. All 17 are retained and flagged.
* 5,106 rows have a valid mask; 3,571 train tampered rows are pixel-training
  eligible.
* All 12,614 image paths exist; all 5,106 mapped masks are readable.
* No duplicate split filenames or ambiguous image-to-mask mappings were found.
* CASIA metadata does not expose a verified camera make/model, so every row is
  `source_group=unknown`.

The existing CASIA loader still skips maskless tampered rows. That legacy
behaviour is preserved; the manifest makes the omission visible instead of
silent. Filter on `pixel_training_eligible=true` when building future pixel
training sets.

Known but intentionally unresolved: `casia_dataset.py` converts BGR to RGB
before ImageNet normalisation, while the deployed `preprocess_bgr` preserves the
official BGR order. This is reserved for a later approved phase.

---

## 5. Dataset provenance, licensing and access

The following is the working record from the earlier dataset investigation. It
was **not** re-downloaded in Phase 0. Items marked "verify" must be checked
against the official source before acquisition, use or redistribution.

| Source | Working facts | Status / verification needed |
| --- | --- | --- |
| CASIA 2.0 | 7,491 authentic and 5,123 tampered rows in the preserved split; masks available for 5,106 tampered rows. | Local copy verified. Licence/access terms must be re-checked with the provider before redistribution. |
| IMD2020 real-life | 2,010 manipulated images, originals and manually created masks. | High-value public addition from prior investigation. Licence/download terms: verify. |
| IMD2020 large-scale | 35,000 real images from 2,322 camera models plus 35,000 synthetic manipulations. | Prior investigation fact. Device labels and licence: verify. |
| MIML | 123,150 forged images with automatically generated masks; CC BY-NC 4.0. | Non-commercial licence restricts use. Auto-mask quality: verify per subset. |
| RAISE | 8,156 RAW authentic images. No manipulation masks. | Useful for authentic FPR / camera-source evaluation, not pixel supervision. Licence: verify. |
| SIDD | Authentic smartphone photographs. | Authentic-only; no manipulation masks. Device labels/licence: verify. |
| VISION | Authentic smartphone photographs. | Authentic-only; no manipulation masks. Device labels/licence: verify. |
| FloreView | Authentic smartphone/device photographs. | Authentic-only; no manipulation masks. Device labels/licence: verify. |
| FIDD-6000 | Social-media manipulation benchmark. | Device labels are not independently verified. Licence: verify. |
| SMIFD-1000 | Social-media manipulation benchmark. | Device labels are not independently verified. Licence: verify. |
| MFC / OpenMFC | Previously considered manipulation dataset. | No longer publicly available from the reviewed source at investigation time. |
| Custom phone collection | Would provide verified Android/iPhone labels and capture metadata. | Not yet collected. Consent, privacy and provenance rules required. |

**Conclusion carried forward:** no public dataset with verified per-device
smartphone labels, authentic originals and pixel masks was identified. A
controlled collection is needed for a real Android/iPhone evaluation. Synthetic
fixtures validate tooling only; they are not evidence of real-world accuracy.

---

## 6. Duplicate and leakage audit

`evaluation/audit_manifest_duplicates.py` is read-only. It detects:

1. **Exact file duplicates** by SHA-256.
2. **Decoded-image duplicates** by a canonical hash of EXIF-oriented RGB pixels
   and dimensions. Re-encoding the same pixels does not hide this duplicate.
3. **Perceptual near-duplicates**:
   * pHash: 32x32 grayscale, 8x8 DCT, median threshold, 64 bits;
   * dHash: 9x8 grayscale adjacent-pixel differences, 64 bits;
   * Hamming thresholds default to 6 (pHash) and 8 (dHash).
4. **Geometric near-duplicates** when `--geometric` is passed: ORB keypoints,
   Lowe ratio test, RANSAC homography and an inlier floor. Horizontal flips are
   tried explicitly. The number of compared pairs is bounded by
   `--max-geometric-pairs`; this bound is reported in the audit parameters and
   can be raised for smaller manifests.
5. **Explicit original/manipulated relationships** through shared `group_id`
   and `pair_id`, plus cross-class duplicate or geometric matches.
6. **Cross-dataset duplicates** whenever a duplicate group contains more than
   one dataset value.

The audit groups rows and writes JSON only. It never renames, rewrites, moves or
deletes a source image. A test verifies that file hashes and modification times
are unchanged after an audit.

### Audit output

The JSON contains `exact_groups`, `decoded_groups`, `phash_groups`,
`dhash_groups`, `geometric_groups`, `cross_dataset_groups`, `relationships`,
`issues` and `summary`. Each group lists sample IDs, dataset names, paths,
splits, classes and group IDs.

### Limits

* pHash/dHash are not proof of provenance; they detect near-identical visual
  content.
* Geometric matching can produce false positives on flat, synthetic or
  low-texture images and is therefore opt-in.
* The pairwise geometric scan is deliberately bounded; run it on a reduced
  candidate set for CASIA-scale manifests.
* Similar scene series may be grouped transitively. Review large groups before
  making split decisions.

---

## 7. Data-leakage prevention

Recommended workflow for every new dataset:

1. Ingest all rows into one manifest; never discard missing/invalid rows.
2. Set `group_id` before splitting. All versions and variants inherit it.
3. Validate that each group appears in exactly one split.
4. Run exact and decoded duplicate audits before accepting the split.
5. Run perceptual and, for selected candidates, geometric audits.
6. Review cross-dataset duplicate groups, especially when mixing CASIA,
   IMD2020, MIML and phone collections.
7. Put related camera/scene series in the same split.
8. Keep `Unknown` phone photographs out of authentic and tampered training
   counts until authenticity is independently established.
9. Preserve the split CSV and record its SHA-256 in the experiment notes.

The validator reports group/split conflicts as errors. It does not auto-fix or
re-split data.

---

## 8. Per-source evaluation

`evaluation/evaluate_module1_by_source.py` evaluates supplied predictions. It
does not run the model or change thresholds. Predictions CSV:

```csv
sample_id,image_score,probability_map_path,raw_mask_path,filtered_mask_path,score_definition
android_0001,0.87,,masks/raw/android_0001.png,masks/filtered/android_0001.png,max_pixel_probability
```

* `image_score` must be the maximum pixel probability in `[0, 1]`. If a
  probability map is supplied, the evaluator recomputes the maximum and rejects
  a mismatching score.
* `raw_mask_path` is the raw mask at the localization threshold.
* `filtered_mask_path` is the post-processed highlighted mask. If it is absent
  and a probability map is supplied, the deployed connected-component filter can
  be reproduced (`--no-derive-filtered` disables this). Raw and filtered metrics
  remain separate.
* `score_definition`, when present, must be `max_pixel_probability`.

### Source groups

| Report key | Members |
| --- | --- |
| `dslr_mirrorless` | DSLR and mirrorless cameras |
| `android` | Android smartphones |
| `iphone` | iPhones |
| `other_mobile` | Other mobile devices |
| `combined_camera` | Derived camera group |
| `combined_mobile` | Android + iPhone + other mobile |
| `all_sources` | Every canonical group; image metrics exclude `Unknown` class |

### Image-level metrics

Computed per group from binary predictions at the image threshold (default
`0.42`):

* accuracy, precision, recall, F1-score;
* false-positive rate on authentic images;
* ROC AUC (only when both classes are present);
* balanced accuracy;
* uncertain rate: fraction of scores within `threshold +/- 0.13`.

Undefined values are `null`; they are not replaced with zero. A group with no
labelled predictions is `not_available`. A group missing some predictions is
`partial`, and the missing sample IDs are listed.

### Pixel-level metrics

Computed only on tampered samples with a valid ground-truth mask and a supplied
or derived prediction mask:

* global IoU (micro TP / (TP+FP+FN));
* mean image IoU (macro mean of per-image IoU);
* pixel precision, recall and F1 (micro from accumulated counts);
* mean predicted area fraction and mean ground-truth area fraction.

`raw_thresholded` and `post_filtered_highlighted` have separate accumulators and
separate metrics. Raw metrics measure the localization threshold; filtered
metrics measure what the application would highlight after connected-component
filtering.

The evaluator fails loudly if a prediction supplies a mask for a tampered sample
whose ground-truth mask is missing. It never silently treats that sample as a
negative or perfect result.

### What is not yet available

No real labelled Android, iPhone, DSLR or mirrorless evaluation set is present.
The tool will therefore report those groups as `not_available` unless real
predictions and labels are supplied. Synthetic tests validate arithmetic and
grouping only.

---

## 9. Commands

```powershell
# Build the CASIA manifest (reads the preserved dataset only)
.\mvss_venv\Scripts\python.exe evaluation/build_casia_manifest.py

# Fast schema/path pass without hashes
.\mvss_venv\Scripts\python.exe evaluation/build_casia_manifest.py --no-hashes

# Duplicate audit (add --geometric for ORB/RANSAC)
.\mvss_venv\Scripts\python.exe evaluation/audit_manifest_duplicates.py `
  --manifest results/manifests/casia2_manifest.csv

# Per-source evaluation from supplied predictions
.\mvss_venv\Scripts\python.exe evaluation/evaluate_module1_by_source.py `
  --manifest results/manifests/casia2_manifest.csv `
  --predictions results/manifests/casia2_predictions.csv
```

Generated manifests, audit reports and evaluation JSONs under `results/` are
ignored by Git and are not dataset or checkpoint replacements.

---

## 10. Future dataset integration procedure

1. Verify licence, access and redistribution terms.
2. Download only after explicit approval; never commit the dataset.
3. Record official URL, licence, collection date and checksum.
4. Create rows with stable `sample_id` and `group_id`; preserve originals.
5. Populate device make/model/capture type only from verified metadata. Use
   `unknown` otherwise.
6. Add masks with `mask_available`, `mask_type` and honest `mask_quality`.
7. Run `validate_manifest`; fix every missing/invalid path. Do not remove rows to
   hide failures.
8. Run exact and decoded duplicate audits across all datasets, then perceptual
   and geometric review.
9. Assign splits at group level. Validation is for model/threshold comparison;
   external test data stays untouched until final evaluation.
10. For smartphones, collect authentic and manipulated images under documented
    consent and metadata. Unlabelled photographs stay `Unknown`.
11. Generate predictions without changing deployed preprocessing or thresholds.
12. Report each source group separately, and report unavailable groups as
    unavailable.
13. Only after Phase 0 evidence is reviewed should any training, threshold or
    preprocessing change be proposed.

---

## 11. Tests

New tests:

* `tests/test_dataset_manifest.py` — round trip, unique IDs, path checks,
  missing masks, group/split leakage, source aliases, hashes, CASIA adapter and
  append-only preservation.
* `tests/test_duplicate_audit.py` — exact, decoded, perceptual, geometric,
  cross-dataset, original/manipulated relationships and read-only behaviour.
* `tests/test_source_evaluation.py` — per-source/combined grouping, image
  metrics, FPR, AUC, uncertain rate, raw vs filtered pixel metrics, missing
  predictions, missing masks and max-pixel score enforcement.

Synthetic fixtures are test data only. They must not be quoted as real-world
model performance.
