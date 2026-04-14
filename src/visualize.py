"""
visualize.py – "Attention Gap" visualisation.

For each sample, renders a 6-panel figure:
  [Satellite Image] [Ground Truth] [Teacher Pred] [Teacher CAM]
  [Student Pred]    [Student CAM]

Saves multi-panel PNGs to outputs/attention_gap_{image_id}.png
Also produces a training curve plot from the CSV logs.

Usage
-----
  python src/visualize.py                   # 10 random test images
  python src/visualize.py --n 20            # 20 images
  python src/visualize.py --image_id 12345  # specific image
  python src/visualize.py --curves          # only plot training curves
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import DeepGlobeDataset, get_val_transforms
from src.models  import get_teacher, get_student, get_feature_layer
from torch.utils.data import DataLoader


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def denormalize(tensor):
    """(3,H,W) → (H,W,3) uint8 RGB numpy."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def mask_to_rgb(mask_tensor):
    """(1,H,W) float → (H,W,3) uint8."""
    m = mask_tensor.squeeze().cpu().numpy()
    m = (m > 0.5).astype(np.uint8) * 255
    return np.stack([m, m, m], axis=-1)


def cam_to_rgb(cam_tensor, colormap=cv2.COLORMAP_JET):
    """(1,H,W) float [0,1] → (H,W,3) uint8 coloured heatmap."""
    cam = cam_tensor.squeeze().cpu().numpy()
    cam = (cam * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(cam, colormap), cv2.COLOR_BGR2RGB)


def overlay_cam(image_rgb, cam_rgb, alpha=0.5):
    """Blend CAM heatmap over original image."""
    return (image_rgb * (1 - alpha) + cam_rgb * alpha).astype(np.uint8)


def compute_cam(model, target_layer, x):
    """Shared Grad-CAM helper (eval mode, single image)."""
    model.eval()
    _act, _grad = [None], [None]

    def fh(m, inp, out): _act[0] = out
    def bh(m, gi, go):   _grad[0] = go[0]

    hf = target_layer.register_forward_hook(fh)
    hb = target_layer.register_full_backward_hook(bh)

    x = x.clone().requires_grad_(True)
    logits = model(x)
    logits.mean().backward()

    g = _grad[0]; a = _act[0]
    w   = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    hf.remove(); hb.remove()
    model.zero_grad()

    return logits.detach(), cam.detach()


# ──────────────────────────────────────────────────────────────────────────────
#  Attention-Gap figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_gap(image, mask, teacher, student, t_layer, s_layer, device, save_path):
    """
    6-panel figure:
      Satellite | GT Mask | Teacher Pred | Teacher CAM Overlay
      Student Pred | Student CAM Overlay
    """
    x = image.unsqueeze(0).to(device)

    # Teacher
    t_logits, t_cam = compute_cam(teacher, t_layer, x)
    t_pred = (torch.sigmoid(t_logits) > 0.5).float()

    # Student
    s_logits, s_cam = compute_cam(student, s_layer, x)
    s_pred = (torch.sigmoid(s_logits) > 0.5).float()

    # Convert to numpy
    img_np    = denormalize(image)
    mask_np   = mask_to_rgb(mask)
    t_pred_np = mask_to_rgb(t_pred.squeeze(0))
    s_pred_np = mask_to_rgb(s_pred.squeeze(0))
    t_cam_np  = overlay_cam(img_np, cam_to_rgb(t_cam.squeeze(0)))
    s_cam_np  = overlay_cam(img_np, cam_to_rgb(s_cam.squeeze(0)))

    panels = [
        (img_np,    "Satellite Image"),
        (mask_np,   "Ground Truth"),
        (t_pred_np, "Teacher Prediction"),
        (t_cam_np,  "Teacher Grad-CAM"),
        (s_pred_np, "Student Prediction"),
        (s_cam_np,  "Student Grad-CAM"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, (panel, title) in zip(axes.flat, panels):
        ax.imshow(panel)
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Attention Gap: Teacher vs. Student Grad-CAM",
        color="white", fontsize=17, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves():
    import pandas as pd
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    colors = {"teacher": "#e94560", "student": "#0f3460"}

    for split, path, label in [
        ("teacher", "outputs/teacher_training_log.csv", "Teacher"),
        ("student", "outputs/student_training_log.csv", "Student"),
    ]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        col = colors[split]

        # Loss
        axes[0].plot(df["epoch"], df["tr_loss"] if "tr_loss" in df else df.get("tr_total", []),
                     label=f"{label} Train", color=col, linewidth=2)
        axes[0].plot(df["epoch"], df["va_loss"],
                     label=f"{label} Val",   color=col, linewidth=2, linestyle="--")

        # IoU
        iou_col = "tr_iou" if "tr_iou" in df else "tr_iou"
        if iou_col in df:
            axes[1].plot(df["epoch"], df[iou_col],
                         label=f"{label} Train IoU", color=col, linewidth=2)
        axes[1].plot(df["epoch"], df["va_iou"],
                     label=f"{label} Val IoU", color=col, linewidth=2, linestyle="--")

    for ax, title, ylabel in zip(
        axes,
        ["Loss Curves", "IoU Curves"],
        ["Loss", "IoU"],
    ):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", color="#a0a0b0")
        ax.set_ylabel(ylabel, color="#a0a0b0")
        ax.tick_params(colors="#a0a0b0")
        ax.spines[:].set_color("#2a2a4a")
        legend = ax.legend(facecolor="#2a2a4a", edgecolor="none",
                           labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = "outputs/training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training curves saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--n",        type=int, default=10,
                        help="Number of test images to visualise")
    parser.add_argument("--teacher",  default=None)
    parser.add_argument("--student",  default=None)
    parser.add_argument("--curves",   action="store_true",
                        help="Only plot training curves, skip attention gap")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.curves:
        plot_training_curves()
        return

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cfg["data"]["img_size"]

    # ── Load models ───────────────────────────────────────────────────────────
    teacher = get_teacher(cfg).to(device)
    student = get_student(cfg).to(device)

    t_ckpt = args.teacher or cfg["teacher"]["checkpoint"]
    s_ckpt = args.student or cfg["student"]["checkpoint"]

    if os.path.exists(t_ckpt):
        teacher.load_state_dict(torch.load(t_ckpt, map_location=device)["model"])
    if os.path.exists(s_ckpt):
        student.load_state_dict(torch.load(s_ckpt, map_location=device)["model"])

    t_layer = get_feature_layer(teacher, f"encoder.{cfg['teacher']['cam_layer']}")
    s_layer = get_feature_layer(student, f"encoder.{cfg['student']['cam_layer']}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = DeepGlobeDataset(
        cfg["data"]["root"], "test",
        get_val_transforms(img_size),
        max_samples=args.n,
    )

    os.makedirs("outputs", exist_ok=True)
    print(f"\nGenerating attention-gap figures for {len(test_ds)} samples …")

    for idx in range(min(args.n, len(test_ds))):
        image, mask = test_ds[idx]
        img_id = os.path.basename(test_ds.image_paths[idx]).replace("_sat.jpg", "")
        save_path = f"outputs/attention_gap_{img_id}.png"
        plot_attention_gap(image, mask, teacher, student,
                           t_layer, s_layer, device, save_path)

    # Also regenerate training curves if logs exist
    plot_training_curves()


if __name__ == "__main__":
    main()
