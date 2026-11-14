# 5-minute teacher demonstration guide

This guide is the short version you can follow in front of your teacher. It explains
what to click, what to say, and which claims are honest. The long reference is
[`APPLICATION.md`](APPLICATION.md); measured numbers are in
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

---

## 0. One-paragraph project statement

> The system is a Streamlit application that integrates three forensic modules.
> Module 1 localises image manipulation with MVSS-Net and classifies the whole image as
> Authentic, Tampered or Uncertain. Module 2 is an AI-generated-image detector whose
> training/evaluation harness is finished, but which honestly reports **unavailable**
> because no licensed dataset and trained checkpoint exist yet. Module 3 screens chat
> screenshots with OCR and, when a trusted original screenshot is supplied, highlights
> visible pixel and text differences. Every displayed number comes from a recorded run;
> unavailable evidence is labelled unavailable rather than invented.

---

## 1. Before the demo (2 minutes, once)

### 1.1 Start the application

```powershell
cd D:\projects\MVSS-Net\MVSS-Net-master
.\mvss_venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL Streamlit prints (normally <http://localhost:8501>).

The first Module 1 analysis loads the 560 MB checkpoint and takes roughly 7–10 s on
CPU. Analyse one image **before** the teacher arrives so the model is warm.

### 1.2 Choose which Module 1 checkpoint to show

* **Default (no extra step):** the preserved original `ckpt/mvssnet_casia.pt`.
  Its test metrics are optimistic because 89.45% of the historical test images share
  duplicate components with training data. Say that out loud.
* **After the Colab notebook has produced `mvssnet_groupaware.pt`:** keep the original
  file untouched and point the app at the new one with an environment variable:

```powershell
$env:MVSS_CHECKPOINT = "D:\projects\MVSS-Net\MVSS-Net-master\experiments\module1\groupaware_t4_v1\mvssnet_groupaware.pt"
.\mvss_venv\Scripts\python.exe -m streamlit run app.py
```

Use the actual path where you saved the downloaded checkpoint. The group-aware test
numbers then come from `module1_groupaware_test_metrics.json` / `demo_summary.json`
exported by the notebook, **not** from the old report.

### 1.3 Files used in the Module 3 demo

The synthetic demo dataset is already on this laptop:

```text
D:\ImageForensics_Datasets\chat_synthetic_v1\authentic\conv_001.png          <- reference
D:\ImageForensics_Datasets\chat_synthetic_v1\manipulated\conv_001_edited.png <- suspect
```

If that folder is missing, rebuild it (takes under a minute):

```powershell
.\mvss_venv\Scripts\python.exe evaluation\build_synthetic_chat_dataset.py --root D:\ImageForensics_Datasets\chat_synthetic_v1 --conversations 8
```

For Module 1, have any ordinary photo ready. A second, visually edited copy makes the
localisation demo clearer; create one with:

```powershell
.\mvss_venv\Scripts\python.exe -c "from PIL import Image, ImageDraw; im=Image.open('my_photo.jpg').convert('RGB'); d=ImageDraw.Draw(im); w,h=im.size; d.rectangle([int(w*0.45),int(h*0.35),int(w*0.75),int(h*0.65)], fill=(35,35,35)); im.save('demo_tampered.jpg', quality=95)"
```

---

## 2. The five-minute script

### 0:00–0:30 — Home Dashboard

**Click:** `Home Dashboard`.

**Say:** “The app has six pages. The sidebar shows live module status: MVSS-Net ready,
AI detector unavailable, OCR ready. Status is read from the filesystem, not hard-coded.”

Point at the status rows. If OCR is missing, say the reference comparison still runs but
text-change comparison is reported as degraded.

### 0:30–2:30 — Module 1: Image Manipulation Detection

**Click:** `Image Manipulation Detection`.

1. Upload the untouched photo under **“Upload a photograph”**, click **“Analyse image”**.
2. Read the decision card:
   * **Authentic** — maximum pixel probability at or below `0.42 − 0.13 = 0.29`.
   * **Uncertain** — between `0.29` and `0.41`.
   * **Tampered** — at or above `0.42`.
3. Upload the edited copy, click **“Analyse image”** again.
4. Open the tabs **Visual explanation**, **Suspicious regions**, **Decision & thresholds**,
   **Model & limitations**.

**Say:**

* “There are two separate thresholds. The **image threshold 0.42** decides the whole
  image; the **pixel threshold 0.40** only builds the localisation mask. The ±0.13 band
  lets the system say *Uncertain* instead of guessing.”
* “The badge ‘Pixel-level signal: Possible manipulation’ is not proof the whole image is
  fake — it only means some pixels crossed 0.40.”
* “Thresholds were chosen on validation data only; the test split was never used to tune
  them.”
* “This checkpoint is the original CASIA-trained model. Its historical test set leaks
  duplicates across splits, so the old IoU of 0.7466 is optimistic. The group-aware
  retraining run exists specifically to remove that leakage; quote its metrics only from
  the exported JSON.”
* If the edited rectangle is **not** detected, say: “MVSS-Net was trained on CASIA
  splicing/copy-move manipulations, not on arbitrary black boxes. A miss is a real
  limitation, not a scripted result.”

Do not claim a fixed accuracy percentage for an arbitrary uploaded photo.

### 2:30–3:30 — Module 2: AI-Generated Image Detection

**Click:** `AI-Generated Image Detection`.

**Say:** “This page is deliberately honest: no detector manifest is integrated, so it
shows **unavailable** and produces no score. The training/export harness is complete and
tested on synthetic folders, but real training needs a licensed dataset.”

Facts to quote:

* **Synthbuster** — 12.37 GB, CC BY-NC-SA 4.0 (non-commercial), 9 generators × 1,000
  images, **no real photos**.
* A matched real class (RAISE-1k, non-commercial, tens of GB) is still needed.
* The planned protocol holds whole generators out, so cross-generator performance can be
  reported honestly.

**Say:** “I did not fake a prediction to make the demo look complete.”

### 3:30–5:00 — Module 3: Chat Screenshot Analysis

**Click:** `Chat Screenshot Analysis`.

1. **“Upload a chat screenshot”** → `conv_001_edited.png`.
2. **“Upload the trusted original screenshot”** → `conv_001.png`.
3. Click **“Analyse screenshot”**.
4. Open **Reference comparison**, then **Regions & text**.

**Say:**

* “Without a reference, the module is only a visual triage aid. On synthetic authentic
  screenshots MVSS-Net produced an authentic false-positive rate of 1.0 — it is out of
  domain on UI renders. The app discloses that.”
* “With a reference, it aligns the two images, thresholds per-pixel differences at
  25/255, and diffs the OCR transcripts. In the controlled synthetic set it recovered all
  10 injected text edits with 0 authentic false positives.”
* “Those are **synthetic** renders, not real WhatsApp evidence. A reference comparison
  also cannot prove which screenshot is the original; it can only describe differences
  relative to the reference you supply.”

Optional stress-test fact: under one-sided JPEG the text-change *recall* stays high
(1.0 at q85/q60, 0.9 at q40) but *precision* drops because compression creates spurious
OCR fragments; a 50% rescale drops recall to 0.7. Region recall remains 1.0.

### 5:00 — Close

**Say:** “Module 1 is measured with an explicit leakage caveat and a leakage-free
retraining pipeline. Module 2 is unavailable by design until a licensed dataset is
approved. Module 3 is strongest when a trusted reference exists, and all its current
numbers are synthetic.”

---

## 3. Exact definitions if asked

| Term | One-line explanation |
| --- | --- |
| IoU | Intersection over union of predicted and true tampered pixels; 1.0 is perfect. |
| Micro IoU | IoU computed over all pixels pooled together, dominated by large masks. |
| Mean image IoU | Per-image IoU averaged equally, so small or missed forgeries matter more. |
| AUC | Probability that a random tampered image scores above a random authentic one. |
| FPR | Fraction of authentic images wrongly flagged as tampered. |
| Precision / recall | Of the flagged changes, how many are real / of the real changes, how many were found. |
| Group-aware split | Whole duplicate families are kept in one split, so near-duplicates cannot leak across train/test. |
| Uncertain band | Scores within ±0.13 of the image threshold abstain instead of forcing a label. |

---

## 4. Likely questions and honest answers

**“Is the AI-image module working?”**
“The integration and training harness work and are unit-tested. The detector itself is
not trained, so the app reports unavailable. Training requires a licensed dataset; the
smallest route is Synthbuster at 12.37 GB under CC BY-NC-SA 4.0, plus a real-photo class.”

**“Why not use the CASIA authentic images as the real class?”**
“It is possible, but CASIA photos are a weak domain match for generated images and the
limitation would have to be disclosed. RAISE-1k is the better match but needs its own
licence check.”

**“How accurate is Module 1?”**
“On the historical split, global IoU 0.7466 and image-level AUC 0.9880, but 89.45% of the
test tampered images have duplicate relatives in training. Those numbers are optimistic.
The group-aware retraining is the defensible measurement; its result is whatever the
notebook’s exported JSON says, and it is not comparable with the old benchmark as if
nothing changed.”

**“Does Module 3 prove a conversation was faked?”**
“No. With a reference it reports visible pixel and text differences. Without one it is a
triage aid with a measured out-of-domain false-positive problem. It is not message
verification and not legal evidence.”

**“What happens if I upload a real WhatsApp screenshot?”**
“It will be analysed, but the synthetic evidence does not establish real-world accuracy.
Re-rendering, app versions, compression and OCR errors are not represented.”

---

## 5. Troubleshooting during the demo

| Problem | Exact fix |
| --- | --- |
| Page says MVSS-Net missing | Check `ckpt\mvssnet_casia.pt` exists; restart with the command in §1.1 |
| Wrong checkpoint displayed | In the same PowerShell window, set `$env:MVSS_CHECKPOINT` to the intended `.pt`, then restart Streamlit |
| App is slow on first image | Expected: checkpoint load plus CPU inference; pre-warm before the demo |
| Module 3 reference tab absent | A reference screenshot was not uploaded, or dimensions/layout differ too much; use `conv_001.png` with `conv_001_edited.png` |
| OCR unavailable | Reference pixel comparison still works; state that text-change comparison is degraded |
| Port already in use | `.\mvss_venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` |

---

## 6. Claims to avoid

* Do not call the old CASIA test metrics leakage-free.
* Do not quote Module 2 accuracy, AUC or “detects Midjourney” claims.
* Do not present synthetic Module 3 results as real-world accuracy.
* Do not say the system authenticates an image’s origin or proves legal authenticity.
* Do not say “Uncertain” means the image is safe — it means the score is too close to the
  threshold.
