# AI-generated image detector — integration placeholder

This folder is intentionally empty of model weights. The application does **not**
ship an AI-generated-image classifier, and Module 2 will show an explicit
"not integrated" panel rather than a fake prediction.

## How another team member can plug in a model

1. Put the model file in this folder (for example `detector.ts` or `detector.onnx`).
2. Copy `manifest.example.json` to `manifest.json`.
3. Edit the fields to match the supplied model.
4. Restart the Streamlit app. The sidebar will change from
   "not integrated" to "ready".

Supported backend kinds:

| kind | file | contract |
| --- | --- | --- |
| `torchscript` | `*.ts`, `*.pt` exported with `torch.jit.trace/script` | input `1x3xHxW` float tensor, RGB, normalised with the declared mean/std; output logits or probabilities with ≥ 2 classes |
| `onnx` | `*.onnx` | same input/output contract; requires `pip install onnxruntime` |
| `python_adapter` | your Python module | callable `module:function` or `module:Class` receiving an RGB `uint8` NumPy array and returning class probabilities/logits |

### Manifest fields

| field | required | meaning |
| --- | --- | --- |
| `kind` | yes | `torchscript`, `onnx` or `python_adapter` |
| `class_names` | yes | ordered list of class names, e.g. `["real", "ai_generated"]` |
| `path` | torchscript/onnx | model file, relative to this folder |
| `adapter` | python_adapter | `"package.module:callable"` |
| `adapter_kwargs` | no | keyword arguments for the adapter constructor |
| `checkpoint` | no | checkpoint passed to the adapter as `checkpoint=` |
| `input_size` | no | int or `[height, width]` (default 224) |
| `mean`, `std` | no | three ImageNet-style values (defaults are standard ImageNet) |
| `ai_class_index` | no | index of the AI-generated class; inferred from the class name otherwise |

### Decision policy

The shared wrapper converts scores to probabilities and reports:

* `real` — real-class probability is higher and the margin is sufficient,
* `ai_generated` — AI-class probability is higher and the margin is sufficient,
* `uncertain` — top confidence < 0.60 or class margin < 0.20 (configurable via
  `AI_MIN_CONFIDENCE` / `AI_UNCERTAINTY_MARGIN`).

### Evaluation

Use `evaluation/evaluate_module2.py --real-dir ... --ai-dir ...` after
integration. It computes accuracy, macro precision/recall/F1, the confusion
matrix and the uncertain rate on a labelled dataset. Do not report numbers for
a model that has not been trained or evaluated.
