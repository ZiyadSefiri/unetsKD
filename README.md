# Attention-Guided Knowledge Distillation for Satellite Road Segmentation

> **Compress a heavy ResNet-101 U-Net into a lightweight MobileNetV2 U-Net — while preserving *where* the model looks, not just *what* it predicts.**

---

## Overview

This project blends **Knowledge Distillation (KD)** with **Explainable AI (XAI)** to create an efficient road segmentation model optimised for satellite imagery. By using **Grad-CAM heatmaps as a distillation signal**, the Student learns not just to match the Teacher's predictions but to focus on the same critical *transition zones* (road edges vs. surrounding terrain).

### Architecture

| | Teacher | Student |
|---|---|---|
| **Backbone** | ResNet-101 | MobileNetV2 |
| **Head** | U-Net decoder | U-Net decoder |
| **Parameters** | ~65 M | ~5 M |
| **CAM target** | `encoder.layer4` | `encoder.features.18` |

### Composite Loss

$$L_{total} = \alpha L_{task} + \beta L_{KD} + \gamma L_{attn}$$

| Loss | Symbol | Description |
|---|---|---|
| Segmentation | $L_{task}$ | BCE + Soft Dice against ground truth |
| Knowledge Distillation | $L_{KD}$ | KL-Divergence on temperature-scaled soft logits |
| Attention Alignment | $L_{attn}$ | MSE between Teacher & Student Grad-CAM heatmaps |

Default weights: `α=0.5, β=0.25, γ=0.25` (configurable via `configs/config.yaml`)

---

## Dataset

**DeepGlobe 2018 Road Extraction Dataset**

| Split | Images |
|---|---|
| Train | 6,226 |
| Val | 621 |
| Test | 550 |

- Resolution: 1024×1024 (resized to 512×512 during training)
- Binary labels: Road = white (255), Background = black (0)

---

## Project Structure

```
unetKD/
├── archive(1)/          # DeepGlobe dataset (train/ valid/ test/ metadata.csv)
├── configs/
│   └── config.yaml      # All hyperparameters
├── src/
│   ├── dataset.py       # Dataset loader + Albumentations augmentation
│   ├── models.py        # Teacher & Student model factories
│   ├── gradcam.py       # Differentiable hook-based Grad-CAM
│   ├── losses.py        # BCE/Dice, KD, Attention, Composite loss
│   ├── train_teacher.py # Phase 1: Train Teacher
│   ├── train_student.py # Phase 2: Distillation training
│   ├── evaluate.py      # Metrics: IoU, F1, FLOPs, latency, CAM correlation
│   └── visualize.py     # Attention Gap visualisation + training curves
├── checkpoints/         # Saved model weights
├── outputs/             # Figures, evaluation reports, training logs
└── requirements.txt
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Phase 1: Train the Teacher

```bash
# Full training (~50 epochs)
python src/train_teacher.py

# Fast debug mode (3 epochs, 200 samples)
python src/train_teacher.py --fast
```

### 3 – Phase 2: Distillation training

```bash
# Requires checkpoints/teacher_best.pth from Phase 1
python src/train_student.py

# Fast debug mode
python src/train_student.py --fast

# Point to a specific Teacher checkpoint
python src/train_student.py --teacher checkpoints/teacher_best.pth
```

### 4 – Evaluate

```bash
python src/evaluate.py
# Output: outputs/evaluation_report.csv  +  console table
```

### 5 – Visualise Attention Gap

```bash
# Generate 10 side-by-side Teacher/Student attention figures
python src/visualize.py --n 10

# Plot training curves only
python src/visualize.py --curves
```

---

## GPU Requirements

Tested on **GTX 1650 (4 GB VRAM)** with:
- Mixed precision (AMP FP16) enabled by default
- Teacher batch size: **2** | Student batch size: **4**
- Image size: **512×512**

If you run out of VRAM, reduce batch size in `configs/config.yaml` or lower `img_size` to `384`.

---

## Evaluation Metrics

| Category | Metric |
|---|---|
| Segmentation | mIoU, F1 / Dice, Precision, Recall |
| Efficiency | Parameter count, GFLOPs, Inference latency (ms) |
| Attention Quality | Pearson correlation between Teacher / Student Grad-CAM maps |

---

## Output Visualisation

Each figure (`outputs/attention_gap_*.png`) shows:

| Satellite Image | Ground Truth | Teacher Prediction | Teacher Grad-CAM |
|---|---|---|---|
| | | **Student Prediction** | **Student Grad-CAM** |

---

## Configuration (`configs/config.yaml`)

| Key | Default | Description |
|---|---|---|
| `data.img_size` | 512 | Training resolution |
| `train_teacher.batch_size` | 2 | Teacher batch size |
| `train_student.batch_size` | 4 | Student batch size |
| `loss.alpha` | 0.5 | Weight for $L_{task}$ |
| `loss.beta` | 0.25 | Weight for $L_{KD}$ |
| `loss.gamma` | 0.25 | Weight for $L_{attn}$ |
| `loss.kd_temperature` | 4.0 | Softmax temperature |
| `fast_mode.enabled` | false | Debug with 200 samples |

---

## References

- Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network*
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*
- DeepGlobe 2018 Road Extraction Challenge
- Iakubovskii, P. (2019). *Segmentation Models PyTorch*
