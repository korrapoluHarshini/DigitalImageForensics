# Module 2 dataset — GenImage-processed subset

This page records exactly which dataset the Module 2 detector was trained on, how it was
verified, and what may and may not be claimed. It is evidence, not a marketing page.

---

## 1. Source and provenance

| Item | Verified value |
| --- | --- |
| Dataset | `Qwerty0193/genimage-processed` on the Hugging Face Hub |
| Metadata file | `metadata.parquet`, 570,313 bytes |
| Metadata SHA-256 | `85968EC2FB845879B69327F83685E1456854114A4A4D30C77A31F935ED716ED9` |
| Image archive | `processed_images.zip`, 701,877,525 bytes (669.4 MiB transferred), reported as 702 MB |
| Archive SHA-256 | `A2242593FD3A9038D25FD3A54A45198667D72CDA41C5E5C2B3531F3070B0AA40` |
| Upstream | Derived from the Kaggle *Tiny GenImage* mirror of **GenImage** (Zhu et al., 2023) |
| Download date | 14 November 2026, with owner approval |

Both hashes match the LFS object identifiers published in the Hugging Face repository
listing, so the downloaded files are the files the host serves.

## 2. Composition

Preprocessing performed by the dataset author: resize to **224×224** (bilinear), re-encode
as **JPEG quality 96** to remove PNG-vs-JPEG bias, no normalisation applied to the stored
pixels. The application applies the declared ImageNet mean/std at inference time.

| Property | Verified value |
| --- | --- |
| Images | 30,000 unique files (30,000 metadata rows, 0 duplicate paths) |
| Real class | 15,000 images, label `nature`, unique basenames |
| AI class | 15,000 images, six generators × 2,500 |
| Generators | ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion V1.5, VQDM |

| Generator | AI images | Real images in metadata |
| --- | --- | --- |
| ADM | 2,500 | 2,500 |
| BigGAN | 2,500 | 2,500 |
| GLIDE | 2,500 | 2,500 |
| Midjourney | 2,500 | 2,500 |
| Stable Diffusion V1.5 | 2,500 | 2,500 |
| VQDM | 2,500 | 2,500 |

The real images are ImageNet "nature" photographs, replicated in the metadata for each
generator row but unique on disk (15,000 files).

The dataset's own columns give an in-distribution split of `train 21,202 / val 4,427 /
test 4,371`, and a cross-generator view with **BigGAN and Midjourney in `holdout_test`**.
The training harness does **not** read these columns: it re-splits the remaining images
70/15/15 with seed 42 and sends every held-out generator image to the test set. This keeps
the protocol explicit and reproducible from `split.csv`.

## 3. Cross-generator protocol used in this project

* **Held out entirely:** BigGAN and Midjourney (5,000 AI images).
* **Training generators:** ADM, GLIDE, Stable Diffusion V1.5, VQDM.
* **Real images:** no basename overlap between the held-out rows and the training rows
  (verified: 5,000 holdout basenames, 7,152 train basenames, overlap 0), so the holdout is
  not contaminated through duplicated photographs.
* Metrics are reported per generator in `metrics.json → per_generator_test`. The mixed
  `test` row contains both held-out and in-distribution generators and is not, by itself,
  a cross-generator result.

## 4. Licence and redistribution

The Hugging Face dataset card declares **no licence tag**. The images are a derivative of
GenImage, whose terms are **CC BY-NC-SA 4.0** (attribution, non-commercial,
share-alike). This project therefore treats the dataset as:

* usable for academic coursework and non-commercial research;
* **not** usable for a commercial product;
* requiring attribution to GenImage (Zhu et al., 2023) and to the uploader's
  preprocessing;
* not redistributed by this repository. `datasets/` is git-ignored, and no dataset image
  is committed. Only the derived detector weights and measured metrics are kept.

The real class is ImageNet "nature" photographs, and all images went through one uniform
224×224 JPEG Q96 pipeline. Metrics therefore describe this preprocessing and class, not
"all real photographs" and not "all AI generators".

## 5. Local layout (not committed)

```text
datasets/module2_genimage/
├── metadata.parquet          # authoritative composition metadata
├── processed_images.zip      # original download
├── images/                   # extracted: train|val|test / ai|nature / <generator>
├── prepared/                 # hardlinked layout for training
│   ├── real/                 # 15,000 real photographs
│   └── ai/<generator>/       # 2,500 images each
└── smoke/                    # 200 real + 600 AI pipeline check
```

`training/prepare_module2_dataset.py` builds `prepared/` (and `smoke/` with `--smoke`)
from `images/` using hardlinks, falling back to copies. It never modifies the extracted
files.

## 6. What must not be claimed

* Do not call the detector accurate on "AI images" in general: two generators were held
  out and only six are represented.
* Do not present the mixed `test` AUC as a cross-generator number; quote the
  BigGAN/Midjourney rows.
* Do not claim robustness to resizing, recompression, screenshots, or social-media
  pipelines — none were part of this dataset.
* Do not describe the dataset as MIT or public domain: the card is silent and the
  GenImage CC BY-NC-SA 4.0 terms are inherited.
