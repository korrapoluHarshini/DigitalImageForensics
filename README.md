# Digital Image Forensics — MVSS-Net

A Streamlit digital-forensics application built around MVSS-Net, with three independent
modules, a leakage-aware Module 1 evaluation pipeline, and an automated Colab GPU
notebook.

## Project status

| Module | What it does | Status | Evidence |
| --- | --- | --- | --- |
| **1. Image manipulation** | MVSS-Net localisation + Authentic / Tampered / Uncertain image decision | **Ready** | Original checkpoint measured with a leakage caveat; group-aware retraining runs in the Colab notebook |
| **2. AI-generated image** | Pluggable detector with a TorchScript backend | **Pending / unavailable** | Harness and tests only; no licensed dataset, trained checkpoint or metrics |
| **3. Chat screenshot** | Visual screening, OCR and optional trusted-reference comparison | **Ready** (screening; reference comparison for visible differences) | Measured on locally generated synthetic screenshots only |

Honesty rules enforced by the project:

* Module 2 shows **unavailable** instead of inventing a prediction.
* Module 3 never claims real-world accuracy from synthetic renders, and never claims to
  recover the original conversation without a trusted reference.
* Module 1 thresholds (**0.40** pixel, **0.42** image, **±0.13** uncertainty) were selected
  on validation data only; the test split is never used for tuning.
* The original checkpoint `ckpt/mvssnet_casia.pt` and the group-aware split are treated as
  read-only, with their SHA-256 hashes verified before training.

## Quick start (Windows PowerShell)

```powershell
cd D:\projects\MVSS-Net\MVSS-Net-master
.\mvss_venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\mvss_venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://localhost:8501>. The first Module 1 analysis loads the 560 MB checkpoint.

## One-notebook Colab GPU workflow

[`notebooks/colab_end_to_end.ipynb`](notebooks/colab_end_to_end.ipynb) automates the whole
GPU path on a Tesla T4:

1. environment and disk check,
2. repository check / optional clone,
3. dataset layout detection with symlink staging (no files copied),
4. preservation hashes, split counts and mask audit,
5. preprocessing / model / threshold / app consistency gate,
6. pytest,
7. Module 1 smoke run,
8. group-aware training with **automatic resume** from `last.pt`,
9. checkpoint export and direct download (no Google Drive required),
10. independent held-out test evaluation,
11. honest Module 2 status,
12. synthetic Module 3 evaluation,
13. ZIP export of metrics, logs and summaries.

Defaults: 4 epochs, batch 4, 512 px, full train/validation splits. Re-running the
notebook resumes from the last completed epoch; a batch-size change after a CUDA
out-of-memory error does not restart training.

## Demo and documentation

* **Teacher demo:** [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — exact clicks, uploads,
  speaking notes and honest limitations for a five-minute walkthrough.
* **Application reference:** [`docs/APPLICATION.md`](docs/APPLICATION.md) — architecture,
  thresholds, environment variables, privacy rules and troubleshooting.
* **Measured results and caveats:** [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).
* **GPU details and manual commands:** [`docs/COLAB_RUNBOOK.md`](docs/COLAB_RUNBOOK.md).
* **Module 2 integration contract:** [`models/ai_detector/README.md`](models/ai_detector/README.md).

## Tests

```powershell
.\mvss_venv\Scripts\python.exe -m pytest tests -q
```

Expected on the current machine: **172 passed, 2 skipped** (both skips need symlink
permission; the notebook exercises those paths on Linux).

## Verification log (14 Nov 2026)

| Check | Verified result |
| --- | --- |
| Streamlit server | `/` and `/_stcore/health` return **HTTP 200 / `ok`**; six sidebar pages switch without app exceptions |
| Real dataset audit | 8,831 train / 1,694 validation / 2,089 test; 0 cross-split components; 0 missing images; 17 tampered rows without masks excluded and reported |
| Model + app contract (CPU, 512 px) | mask `[1,1,512,512]`, edge `[1,1,128,128]`, finite probabilities; a 450×600 upload returns a 450×600 map |
| Module 1 CPU smoke + resume | 1 epoch on 4 train / 3 validation images; resumed epoch keeps LR `5e-06` and accepts a batch-size change |
| Module 1 test evaluation plumbing | `--smoke` run on held-out test rows exits 0 and labels itself sampled/non-quotable |
| Module 3 synthetic reference comparison | 8 pairs: text precision/recall 1.0, region recall 1.0, pixel IoU 0.180, authentic-pair FPR 0.0 |
| Upload handling | valid images reach the decision card; invalid bytes show `Upload rejected`; uploader reports 25 MB |
| Full test suite | **172 passed, 2 skipped** |

Full-split Module 1 GPU training and its held-out metrics are still produced only by the
Colab notebook; no GPU is available on this machine.

## Use the retrained checkpoint

Keep the original file untouched and select the retrained weights with an environment
variable:

```powershell
$env:MVSS_CHECKPOINT = "D:\path\to\mvssnet_groupaware.pt"
.\mvss_venv\Scripts\python.exe -m streamlit run app.py
```

---

## Upstream MVSS-Net

**Code and models for ICCV 2021 paper: *Image Manipulation Detection by Multi-View Multi-Scale Supervision***

![Image text](https://raw.githubusercontent.com/dong03/picture/main/framework.jpg)
### Update
- ***22.06.02***, MVSS-Net++ is accepted by ***T-PAMI***, check the [paper](https://ieeexplore.ieee.org/document/9789576/).

- ***22.02.17***, Pretrained model for [Real-World Image Foregery Localization Challange](https://tianchi.aliyun.com/competition/entrance/531945/introduction)


- ***21.12.17***,  Something new: MVSS-Net++

We now have an improved version of MVSS-Net, denoted as MVSS-Net++. Check [here](mvssnetplus.md).

### Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6
+ cuda10.1+cudnn7.6.3

  

### Requirements
+ Install [nvidia-apex](https://github.com/NVIDIA/apex) and move it to current directory.
+ pip install [requirements.txt](requirements.tx)



### Usage
#### Dataset

An example of the dataset index file is given as [data/CASIAv1plus.txt](data/CASIAv1plus.txt), where each line contains:
```angular2html
img_path mask_path label
```
- 0 represents the authentic and 1 represents the manipulated. 
- For an authentic image,  the mask_path is "None".
- For wild images without mask groundtruth, the index should at least contain "img_path" per line.  

##### Training sets

+ [DEFACTO-84k](./data/DEFACTO84k-train.txt)
+ [CASIAv2](./data/CASIAv2.txt) / [Edge-Mask](./data/edge_mask.zip)

##### Test sets

+ [DEFACTO-12k](./data/DEFACTO12k-test.txt)
+ [Columbia](./data/Columbia.txt) 
+ [COVER](./data/COVERAGE.txt)
+ [NIST16](./data/NIST16.txt)
+ [CASIAv1plus](./data/CASIAv1plus.txt): Note that some of the authentic images in [CASIAv1](./data/CASIAv1.txt) also appear in CASIAv2. With those images fully replaced by Corel images that are new to both CASIAv1 and CASIAv2, we constructed a revision of CASIAv1 termed as CASIAv1plus. We recommend to use CASIAv1plus as an alternative to the original CASIAv1.  


#### <span id = "jump">Trained Models</span>

We offer FCNs and MVSS-Nets trained on CASIAv2 and DEFACTO_84k, respectively. Please download the models and place them in the [ckpt](ckpt) directory:
+ [百度网盘](https://pan.baidu.com/s/1qbBulvsJ9FReUlng5zDYPg) (提取码：mvss)
+ [Google drive](https://drive.google.com/drive/folders/1CztGkd91xF1QqEXuc2n8rVDTBJ7X695U?usp=sharing)


The performance of these models for image-level manipulation detection (metric: AUC and image-level F1) is as follows. More details are reported in the [paper](https://arxiv.org/abs/2104.06832).

##### Performance metric: AUC
|   Model  | Training data | CASIAv1plus | Columbia |  COVER | DEFACTO-12k |
|:--------:|:-------------:|:-------:|:--------:|:------:|:-----------:|
| MVSS_Net | CASIAv2       | 0.932   | 0.980    | 0.731  | 0.573       |
| MVSS_Net | DEFACTO-84k   | 0.771   | 0.563    | 0.525  | 0.886      |
| FCN      | CASIAv2       | 0.769   | 0.762    | 0.541  | 0.551       |
| FCN      | DEFACTO-84k   | 0.629   | 0.535    | 0.543  | 0.840       |


##### Performance metric: Image-level F1 (threshold=0.5)

|   Model  | Training data | CASIAv1plus | Columbia |  COVER | DEFACTO-12k |
|:--------:|:-------------:|:-------:|:--------:|:------:|:-----------:|
| MVSS_Net | CASIAv2       | 0.759   | 0.802    | 0.244  | 0.404       |
| MVSS_Net | DEFACTO-84k   | 0.685   | 0.353    | 0.360  | 0.799       |
| FCN      | CASIAv2       | 0.684   | 0.481    | 0.180  | 0.458       |
| FCN      | DEFACTO-84k   | 0.561   | 0.492    | 0.511  | 0.709       |



#### Inference & Evaluation
You can specify which pre-trained model to use by setting *model_path* in [do_pred_and_eval.sh](do_pred_and_eval.sh). Given a *test_collection* (e.g. CASIAv1plus or DEFACTO12k-test), the prediction maps and evaluation results will be saved under *save_dir*. The default threshold is set as 0.5.

```
bash do_pred_and_eval.sh $test_collection
#e.g. bash do_pred_and_eval.sh CASIAv1plus
```
For inference only, use following command to skip evaluation:
```
bash do_pred.sh $test_collection
#e.g. bash do_pred.sh CASIAv1plus
```


#### Demo

+ [demo.ipynb](./demo.ipynb): A step-by-step notebook tutorial showing the usage of a pre-trained model to detect manipulation in a specific image.



### Citation
If you find this work useful in your research, please consider citing:
```
@InProceedings{MVSS_2021ICCV,  
  author = {Chen, Xinru and Dong, Chengbo and Ji, Jiaqi and Cao, juan and Li, Xirong},  
  title = {Image Manipulation Detection by Multi-View Multi-Scale Supervision},  
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
  year = {2021}  
}

@ARTICLE{MVSS_2022TPAMI,
  author={Dong, Chengbo and Chen, Xinru and Hu, Ruohan and Cao, Juan and Li, Xirong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MVSS-Net: Multi-View Multi-Scale Supervised Networks for Image Manipulation Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2022.3180556}
}
```
### Acknowledgments
- [Nvidia-apex](https://github.com/NVIDIA/apex) is adopted for semi-precision training/inferencing.
- The implement of DA module is based on the  [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch).
### Contact

If you enounter any issue when running the code, please feel free to reach us either by creating a new issue in the github or by emailing

+ Xinru Chen (chen_xinru1999@163.com)
+ Chengbo Dong (dongchengbo@ruc.edu.cn)
