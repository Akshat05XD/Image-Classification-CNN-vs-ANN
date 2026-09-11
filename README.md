# Image Classification — ANN vs. CNN

Comparing a plain fully-connected network (ANN) against a convolutional network (CNN)
on the **Intel Image Classification** dataset — 6 natural scene categories, ~25,000 images.

## Classes
`buildings` · `forest` · `glacier` · `mountain` · `sea` · `street`

## Project Structure
```
├── Assignment1_Image_Classification.ipynb   # main notebook: data, models, training, results
├── intel-image-classification/              # extracted dataset (not included in repo)
│   ├── seg_train/seg_train/<class_name>/*.jpg
│   └── seg_test/seg_test/<class_name>/*.jpg
└── README.md
```

## Setup

### 1. Environment
Python **3.12** is recommended (avoid 3.14 for now — PyTorch wheel support is still catching up).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

**GPU (NVIDIA, CUDA 12.1) — recommended if available:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy nbformat
```

**CPU-only:**
```bash
pip install torch torchvision torchaudio
pip install matplotlib numpy nbformat
```

Check your GPU is CUDA-capable first:
```bash
nvidia-smi
```

### 3. Dataset
Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
from Kaggle, extract it, and set `DATA_DIR` in the notebook to the extracted folder path
(use a raw string on Windows, e.g. `r"C:\path\to\intel-image-classification"`).

### 4. Run
Open `Assignment1_Image_Classification.ipynb` in Jupyter / PyCharm and run all cells top to bottom.
Make sure the notebook's kernel is pointed at the `.venv` where you installed PyTorch.

## What the notebook does
1. **Preprocessing** — resizes all images to 150×150, normalizes pixel values, splits the
   training pool 80/20 into train/validation, keeps the official test set as a final holdout.
2. **Baseline ANN** — a flatten-then-fully-connected MLP with no convolution.
3. **CNN** — 4 conv+ReLU+maxpool blocks followed by a dense classifier head, with dropout
   for regularization.
4. **Training** — both models trained with the same optimizer, loss, and epoch budget so the
   comparison is apples-to-apples; per-epoch loss/accuracy tracked for both.
5. **Comparison** — trainable parameter count, total training time, and validation/test
   accuracy reported side by side, plus accuracy/loss curves.
6. **Analysis** — written discussion of *why* the CNN outperforms the ANN on spatial image
   data (weight sharing, translation tolerance, hierarchical features, parameter efficiency).
7. **Bonus** — discussion of how CUDA and TensorRT would speed up inference for a trained
   version of this model.

## Results

| Metric | ANN | CNN |
|---|---|---|
| Trainable parameters |34,725,510 |2,896,838 |
| Training time (s) |415.8s |434.0s |
| Validation accuracy |0.609 |0.847 |
| Test accuracy |0.592 |0.854 |

## Notes
- If you hit `CUDA out of memory` on a lower-VRAM GPU, reduce `BATCH_SIZE` in the notebook
  (e.g. from 64 down to 32 or 16).
- If training is running on CPU when you expect GPU, verify with:
  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))
  ```
