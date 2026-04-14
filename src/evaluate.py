"""
evaluate.py – Full evaluation of Teacher and Student models.

Metrics reported
----------------
Segmentation  : IoU (Road), F1 / Dice, Precision, Recall
Efficiency    : Parameter count, FLOPs (via torchinfo), Inference latency (ms)
Attention     : Pearson correlation between Teacher / Student Grad-CAM maps

Usage
-----
  python src/evaluate.py
  python src/evaluate.py --teacher checkpoints/teacher_best.pth \\
                         --student checkpoints/student_best.pth
  python src/evaluate.py --fast          # quick check on 50 test images
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset  import DeepGlobeDataset, get_val_transforms
from src.models   import get_teacher, get_student, get_feature_layer, \
                         count_parameters, format_params
from src.gradcam  import GradCAMExtractor
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
#  Pixel-wise metrics
# ──────────────────────────────────────────────────────────────────────────────

def pixel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Returns (iou, f1, precision, recall) as floats."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds   = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return iou, f1, precision, recall


# ──────────────────────────────────────────────────────────────────────────────
#  Inference latency (warm-up then timed runs)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_latency(model, device, img_size=512, n_runs=50, warm_up=10):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    for _ in range(warm_up):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed_ms


# ──────────────────────────────────────────────────────────────────────────────
#  FLOPs via torchinfo
# ──────────────────────────────────────────────────────────────────────────────

def measure_flops(model, img_size=512):
    try:
        from torchinfo import summary
        info = summary(model, input_size=(1, 3, img_size, img_size),
                       verbose=0, device="cpu")
        return info.total_mult_adds
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────────────────────
#  Grad-CAM Pearson correlation
# ──────────────────────────────────────────────────────────────────────────────

def cam_correlation(t_cam: torch.Tensor, s_cam: torch.Tensor) -> float:
    """Pearson-r between Teacher and Student CAMs (per image, mean)."""
    B = t_cam.shape[0]
    rs = []
    for i in range(B):
        tc = t_cam[i].view(-1).cpu().numpy()
        sc = s_cam[i].view(-1).cpu().numpy()
        r, _ = pearsonr(tc, sc)
        rs.append(r)
    return float(np.mean(rs))


# ──────────────────────────────────────────────────────────────────────────────
#  Teacher CAM (no-grad version for evaluation only)
# ──────────────────────────────────────────────────────────────────────────────

def compute_cam_eval(model, target_layer, x: torch.Tensor):
    """Compute Grad-CAM not for training but for correlation measurement."""
    model.zero_grad()
    _act, _grad = [None], [None]

    def fwd_hook(m, inp, out):  _act[0] = out
    def bwd_hook(m, gi, go):    _grad[0] = go[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    logits.mean().backward()

    g = _grad[0]; a = _act[0]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    B = cam.shape[0]
    flat = cam.view(B, -1)
    mn   = flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
    mx   = flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
    cam  = (cam - mn) / (mx - mn + 1e-8)

    fh.remove(); bh.remove()
    model.zero_grad()
    return logits.detach(), cam.detach()


# ──────────────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, target_layer, loader, device, compute_cams=True):
    model.eval()
    all_iou, all_f1, all_pre, all_rec, all_corr = [], [], [], [], []

    for images, masks in tqdm(loader, desc="  evaluating", leave=False):
        images, masks = images.to(device), masks.to(device)

        if compute_cams:
            images.requires_grad_(True)
            logits, t_cam = compute_cam_eval(model, target_layer, images)
            images.requires_grad_(False)
        else:
            with torch.no_grad():
                logits = model(images)
            t_cam = None

        iou, f1, pre, rec = pixel_metrics(logits.detach(), masks)
        all_iou.append(iou); all_f1.append(f1)
        all_pre.append(pre); all_rec.append(rec)

    return {
        "iou":       float(np.mean(all_iou)),
        "f1":        float(np.mean(all_f1)),
        "precision": float(np.mean(all_pre)),
        "recall":    float(np.mean(all_rec)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--student", default=None)
    parser.add_argument("--fast",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cfg["data"]["img_size"]
    max_s    = 50 if args.fast else 0

    # ── Test data ─────────────────────────────────────────────────────────────
    test_ds = DeepGlobeDataset(
        cfg["data"]["root"], "test",
        get_val_transforms(img_size), max_s
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    # ── Load models ───────────────────────────────────────────────────────────
    teacher = get_teacher(cfg).to(device)
    student = get_student(cfg).to(device)

    t_ckpt = args.teacher or cfg["teacher"]["checkpoint"]
    s_ckpt = args.student or cfg["student"]["checkpoint"]

    if os.path.exists(t_ckpt):
        teacher.load_state_dict(torch.load(t_ckpt, map_location=device)["model"])
        print(f"Teacher loaded from {t_ckpt}")
    else:
        print(f"WARNING: Teacher checkpoint not found – evaluating with random weights")

    if os.path.exists(s_ckpt):
        student.load_state_dict(torch.load(s_ckpt, map_location=device)["model"])
        print(f"Student loaded from {s_ckpt}")
    else:
        print(f"WARNING: Student checkpoint not found – evaluating with random weights")

    # ── Target layers ─────────────────────────────────────────────────────────
    t_layer = get_feature_layer(teacher, f"encoder.{cfg['teacher']['cam_layer']}")
    s_layer = get_feature_layer(student, f"encoder.{cfg['student']['cam_layer']}")

    # ── Segmentation metrics ──────────────────────────────────────────────────
    print("\nEvaluating Teacher …")
    t_seg = evaluate_model(teacher, t_layer, test_loader, device, compute_cams=False)
    print("Evaluating Student …")
    s_seg = evaluate_model(student, s_layer, test_loader, device, compute_cams=False)

    # ── Efficiency ────────────────────────────────────────────────────────────
    print("Measuring latency …")
    t_lat   = measure_latency(teacher, device, img_size)
    s_lat   = measure_latency(student, device, img_size)
    t_flops = measure_flops(teacher.cpu(), img_size)
    s_flops = measure_flops(student.cpu(), img_size)
    teacher.to(device); student.to(device)

    # ── Attention correlation ─────────────────────────────────────────────────
    print("Computing Grad-CAM correlation …")
    correlations = []
    n_corr = min(50, len(test_ds))
    corr_ds = DeepGlobeDataset(
        cfg["data"]["root"], "test",
        get_val_transforms(img_size), n_corr
    )
    corr_loader = DataLoader(corr_ds, batch_size=1, shuffle=False, num_workers=2)

    for images, _ in tqdm(corr_loader, desc="  CAM corr", leave=False):
        images = images.to(device)
        images.requires_grad_(True)
        _, tc = compute_cam_eval(teacher, t_layer, images)
        teacher.zero_grad(); images.grad = None
        _, sc = compute_cam_eval(student, s_layer, images)
        student.zero_grad()
        correlations.append(cam_correlation(tc, sc))

    mean_corr = float(np.mean(correlations))

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  EVALUATION REPORT")
    print("═"*60)

    header = f"{'Metric':<28} {'Teacher':>12} {'Student':>12}"
    print(header)
    print("-" * 55)

    rows = []
    metrics = [
        ("IoU (Road)",       t_seg["iou"],       s_seg["iou"]),
        ("F1 / Dice",        t_seg["f1"],        s_seg["f1"]),
        ("Precision",        t_seg["precision"], s_seg["precision"]),
        ("Recall",           t_seg["recall"],    s_seg["recall"]),
        ("Parameters",       count_parameters(teacher), count_parameters(student)),
        ("FLOPs",            t_flops,            s_flops),
        ("Latency (ms)",     t_lat,              s_lat),
        ("CAM Pearson-r",    mean_corr,          mean_corr),
    ]

    for name, tv, sv in metrics:
        if isinstance(tv, float):
            print(f"  {name:<26} {tv:>12.4f} {sv:>12.4f}")
        else:
            print(f"  {name:<26} {format_params(tv):>12} {format_params(sv):>12}")
        rows.append({"metric": name, "teacher": tv, "student": sv})

    print("═"*60)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv("outputs/evaluation_report.csv", index=False)
    print("Report saved → outputs/evaluation_report.csv")


if __name__ == "__main__":
    main()
