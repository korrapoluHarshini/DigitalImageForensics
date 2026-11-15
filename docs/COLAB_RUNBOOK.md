# Google Colab / GPU Runbook

> **Start with the automated notebook:**
> [`notebooks/colab_end_to_end.ipynb`](../notebooks/colab_end_to_end.ipynb) runs the whole
> Module 1 GPU workflow — environment and input checks, consistency gate, pytest, smoke
> test, resume-aware training, checkpoint export/download, independent test evaluation and
> artifact packaging — **without Google Drive**. The manual commands below remain useful as
> a reference and for reproducing the Module 2 run. If Drive has little
> free space, skip the Drive step and keep outputs under `/content`, then download the
> exported checkpoint at the end.

This repository is CPU-only on the development machine. The two training
workloads that benefit from a GPU are:

| Task | Script | Local status | GPU status |
| --- | --- | --- | --- |
| Module 1 retraining on the leakage-free split | `training/train_mvss_groupaware.py` | smoke-verified on CPU | ready to run |
| Module 2 AI-image detector | `training/train_ai_detector.py` | trained and evaluated locally on CPU (cross-generator holdout) | ready to reproduce (see §4) |

All commands below are written for a Colab notebook cell (`!` prefix optional)
or a Colab terminal. Nothing in this runbook has been executed on a GPU in this
repository; the local evidence is CPU smoke runs only.

---

## 1. Prepare Colab

1. Runtime -> Change runtime type -> **GPU** (T4 is enough for Module 1 at
   512 px, batch 4; A100/L4 is faster).
2. Mount Drive and keep every large artefact there, not in `/content`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Put the repository and the data under one Drive folder, for example:

```text
/content/drive/MyDrive/mvss/
├── repo/                       # this repository (code only)
├── data/                       # CASIA2.0_revised/, CASIA2.0_Groundtruth/
└── experiments/                # training outputs (never inside the repo)
```

4. Install runtime dependencies (Colab already ships a CUDA PyTorch build):

```bash
cd /content/drive/MyDrive/mvss/repo
pip install -r requirements-app.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The legacy `requirements.txt` pins PyTorch 1.5/CUDA 10.1 and is **not** used.

5. Verify the preserved inputs before training:

```bash
python - <<'PY'
import hashlib, pathlib
expected = {
    "ckpt/mvssnet_casia.pt":
        "080BC6C3AAE59F748B547DBF090786FE9D31A6E50749DAAA40871E298D6A7E50",
    "results/manifests/casia2_groupaware_split.csv":
        "BE5ACAB205E294A563397C8E6756A6CA429DAD01E3F4FEE5EEBDC84E2593E49C",
}
for name, digest in expected.items():
    path = pathlib.Path(name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    print(name, actual, "OK" if actual == digest else "MISMATCH")
PY
```

> The split hash is a guard computed from the committed artefact. If it differs,
> run `Get-FileHash results/manifests/casia2_groupaware_split.csv -Algorithm SHA256`
> and investigate before training. **Never** modify the original
> `ckpt/mvssnet_casia.pt` or the historical CASIA split files.

---

## 2. Module 1 - retrain on the group-aware split

### 2.1 Smoke check (1 epoch, 4 images)

```bash
python training/train_mvss_groupaware.py \
  --data-root /content/drive/MyDrive/mvss/data \
  --manifest results/manifests/casia2_manifest.csv \
  --split-csv results/manifests/casia2_groupaware_split.csv \
  --output-dir /content/drive/MyDrive/mvss/experiments/module1/groupaware_smoke \
  --device cuda --smoke --batch-size 2
```

Expected artefacts: `last.pt`, `best.pt`, `mvssnet_groupaware.pt`,
`training_log.csv`, `run.json`, `split_used.csv`. The smoke model is not a
result; `run.json` is marked `"mode": "smoke"`.

### 2.2 Full run (resumable)

The script writes `last.pt` every epoch and `best.pt` whenever validation mean
image IoU improves. Model selection uses **validation only**, at the fixed 0.5
operating threshold. The test split is never read during training.

```bash
python training/train_mvss_groupaware.py \
  --data-root /content/drive/MyDrive/mvss/data \
  --output-dir /content/drive/MyDrive/mvss/experiments/module1/groupaware_v1 \
  --device cuda --epochs 40 --batch-size 4 --num-workers 2 --image-size 512
```

If Colab disconnects, rerun the same command with:

```bash
  --resume /content/drive/MyDrive/mvss/experiments/module1/groupaware_v1/last.pt
```

Do not pass `--overwrite` to a run you want to keep. Use a new versioned
directory (`groupaware_v2`) for a different configuration.

The run reuses the existing MVSS-Net architecture, is initialised read-only from
`ckpt/mvssnet_casia.pt`, and uses loss weights `0.16 seg + 0.80 edge + 0.04 cls`.
Preprocessing is BGR + ImageNet statistics, matching the Streamlit application
and the official evaluation pipeline.

### 2.3 Independent test evaluation

Run **after** training finishes, against the trained checkpoint, not the
original:

```bash
python evaluation/evaluate_groupaware_checkpoint.py \
  --checkpoint /content/drive/MyDrive/mvss/experiments/module1/groupaware_v1/mvssnet_groupaware.pt \
  --split test \
  --data-root /content/drive/MyDrive/mvss/data \
  --device cuda --batch-size 4 \
  --output results/evaluation/module1_groupaware_test_metrics.json \
  --per-image-csv results/evaluation/module1_groupaware_test_per_image.csv
```

The script reports pixel metrics at the fixed 0.40 threshold and image-level
metrics at the fixed 0.42 threshold; both thresholds were selected on the
historical validation split and are not tuned on test. Copy the JSON and CSV
back into the repository (they are small) and quote them with the caveat that
the group-aware test split is experimental, not the historical benchmark.

To keep raw probability maps, add
`--save-probabilities /content/drive/MyDrive/mvss/experiments/module1/prob_test`.
They are float16 and can be large; don't commit them.

---

## 3. Module 1 - what to report

Report, in this order:

1. Validation curve (`training_log.csv`): best epoch, mean image IoU, AUC.
2. Group-aware test metrics from step 2.3 (pixel and image level).
3. The earlier leakage analysis (`results/evaluation/casia2_leakage_impact.json`)
   as the reason the retraining exists.
4. The original historical metrics, clearly labelled as leakage-affected.

Do not compare the new test numbers to the historical test numbers as if they
were the same benchmark: the splits differ by construction.

---

## 4. Module 2 - GenImage-processed detector (approved dataset)

The Module 2 dataset was approved and used: `Qwerty0193/genimage-processed` on the
Hugging Face Hub.

| Property | Value |
| --- | --- |
| Download | `metadata.parquet` 570 KB + `processed_images.zip` 702 MB (669.4 MB transferred) |
| Images | 30,000: 15,000 real + 15,000 AI, six generators x 2,500 |
| Format | 224x224 JPEG Q96 (uniform re-encoding removes PNG-vs-JPEG bias) |
| Real class | ImageNet "nature" photographs (the dataset card's label) |
| Generators | ADM, BigGAN, GLIDE, Midjourney, Stable Diffusion V1.5, VQDM |
| Licence | The card declares no tag. It is a derivative of GenImage, so **CC BY-NC-SA 4.0** (academic, non-commercial) is inherited. Preprocessing is third-party. |
| Protocol | BigGAN and Midjourney held out entirely; report per-generator and cross-generator metrics |

The local copy lives in `datasets/`, which is git-ignored. To reproduce on Colab:

```bash
mkdir -p /content/mvss_data/genimage_processed
curl -L -o /content/mvss_data/genimage_processed/metadata.parquet \
  https://huggingface.co/datasets/Qwerty0193/genimage-processed/resolve/main/metadata.parquet
curl -L -o /content/mvss_data/genimage_processed/processed_images.zip \
  https://huggingface.co/datasets/Qwerty0193/genimage-processed/resolve/main/processed_images.zip
cd /content/mvss_data/genimage_processed && unzip -q processed_images.zip -d images
cd /content/DigitalImageForensics
python training/prepare_module2_dataset.py \
  --source /content/mvss_data/genimage_processed/images \
  --output /content/mvss_data/genimage_processed/prepared
```

`prepare_module2_dataset.py` flattens real images and groups AI images by generator; it
hardlinks when the filesystem allows and never modifies the extracted files. Add
`--smoke` for a 200-real / 100-per-generator pipeline check.

### 4.1 Train with two generators held out

```bash
python training/train_ai_detector.py \
  --real-dir /content/mvss_data/genimage_processed/prepared/real \
  --ai-dir /content/mvss_data/genimage_processed/prepared/ai \
  --output-dir /content/drive/MyDrive/mvss/experiments/module2/genimage_resnet18 \
  --holdout-generators "BigGAN,Midjourney" \
  --backbone resnet18 --pretrained \
  --input-size 192 --epochs 4 --batch-size 32 --learning-rate 3e-4 \
  --device cuda \
  --dataset-name "GenImage-processed (Tiny-GenImage derivative)" \
  --dataset-license "CC BY-NC-SA 4.0 (inherited from GenImage; card does not declare a tag)" \
  --provenance "Hugging Face Qwerty0193/genimage-processed; 224x224 JPEG Q96; real class = ImageNet nature photographs; BigGAN and Midjourney held out"
```

`--holdout-generators` removes entire generators from train/validation and puts them in
the test set, which is the only way to make a cross-generator claim. Without it, test
metrics are in-distribution only. The harness re-splits the remaining images 70/15/15
with seed 42; the dataset's own `split` column is not used.

Resume with `--resume .../last.pt`.

Outputs: `best.pt`, `last.pt`, `detector.ts`, `manifest.json`, `metrics.json`, `split.csv`,
`training_log.csv`. `metrics.json` contains `validation`, `test`, `per_generator_test` and
`history`; quote the held-out-generator rows, not only the mixed `test` row.

### 4.2 Independent evaluation and integration

```bash
python evaluation/evaluate_module2.py \
  --real-dir /content/mvss_data/genimage_processed/prepared/real \
  --ai-dir /content/mvss_data/genimage_processed/prepared/ai \
  --manifest /content/drive/MyDrive/mvss/experiments/module2/genimage_resnet18/manifest.json \
  --output results/evaluation/module2_genimage_metrics.json
```

Copy `detector.ts` and `manifest.json` into `models/ai_detector/` to switch the app from
`Integration placeholder` to `Detector integrated`. Never hand-edit metrics into a
manifest; they are written by the training run.

> **Licence caveat for reports.** CC BY-NC-SA 4.0 is academic/non-commercial. The real
> class is ImageNet "nature" photographs, so the numbers describe this preprocessing and
> class, not every photograph or every generator.

---

## 5. Module 3 - synthetic stress test (CPU is fine)

Module 3 does not need a GPU. The degradation stress test runs locally:

```bash
python evaluation/evaluate_chat_degradation.py \
  --dataset /content/drive/MyDrive/mvss/data/chat_synthetic_v1
```

It reports clean, JPEG 85/60/40 (suspect only), 50% rescale (suspect only) and
JPEG 60 on both images. Results are synthetic and must be labelled as such.

---

## 6. Housekeeping

- Keep checkpoints in Drive `experiments/`, never in the repository
  (`*.pt`, `*.pth` and `results/` are git-ignored anyway).
- Never write to `ckpt/` on the development machine; `--output-dir` must point
  outside it.
- Do not commit probability caches, checkpoints, or downloaded datasets.
- After a run, copy back only: `run.json`, `training_log.csv`, `metrics.json`,
  `per_generator` entries, and the final evaluation JSON/CSV.
- Record the Colab GPU type and runtime in the report; do not claim hardware you
  did not use.
