"""
train_student.py – Phase 2: Attention-Guided Knowledge Distillation.

The Teacher is FROZEN.  The Student is trained with:
  L_total = α · BCE/Dice  +  β · KL-Div (soft logits)  +  γ · MSE (Grad-CAM)

Usage
-----
  python src/train_student.py
  python src/train_student.py --fast
  python src/train_student.py --teacher checkpoints/teacher_best.pth
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset  import DeepGlobeDataset, get_train_transforms, get_val_transforms
from src.models   import get_teacher, get_student, get_feature_layer, \
                         count_parameters, format_params
from src.gradcam  import GradCAMExtractor
from src.losses   import TotalKDLoss, BCEDiceLoss

from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
#  IoU helper (same as train_teacher.py)
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-8)).item()


# ──────────────────────────────────────────────────────────────────────────────
#  Training step
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    teacher_ext, student_ext,
    loader, optimizer, loss_fn, val_loss_fn,
    scaler, device, grad_clip
):
    """
    teacher_ext : GradCAMExtractor wrapping the frozen Teacher
    student_ext : GradCAMExtractor wrapping the Student
    """
    # Teacher is always frozen
    for p in teacher_ext.model.parameters():
        p.requires_grad_(False)
    teacher_ext.model.eval()
    student_ext.model.train()

    total = {"total": 0.0, "task": 0.0, "kd": 0.0, "attn": 0.0, "iou": 0.0}
    n = 0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            # ── Teacher forward + CAM (no gradient storage for teacher params)
            with torch.no_grad():
                t_logits = teacher_ext.model(images)
                # Re-run Grad-CAM in a separate, isolated context for teacher
            # We need teacher CAM but do NOT want teacher grads in the graph.
            # Solution: compute teacher CAM once (detached), store result.
            t_logits_d = t_logits.detach()

            # ── Dummy teacher CAM (detached, no grad) ─────────────────────────
            # Teacher CAM is computed outside autocast for numerical precision.

        # Teacher Grad-CAM (no autocast, isolated graph)
        teacher_ext.model.zero_grad()
        t_logits_cam = teacher_ext.model(images)
        t_score = t_logits_cam.mean()
        t_score.backward()
        # Build teacher CAM from hooks then detach completely
        with torch.no_grad():
            g  = teacher_ext._gradients    # (B,C,h,w)
            a  = teacher_ext._activations  # (B,C,h,w)
            w  = g.mean(dim=(2, 3), keepdim=True)
            import torch.nn.functional as F
            tcam = F.relu((w * a).sum(dim=1, keepdim=True))
            tcam = F.interpolate(tcam, size=images.shape[2:],
                                 mode="bilinear", align_corners=False)
            B = tcam.shape[0]
            flat = tcam.view(B, -1)
            mn   = flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
            mx   = flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
            t_cam = ((tcam - mn) / (mx - mn + 1e-8)).detach()

        teacher_ext.model.zero_grad()   # clean up teacher grads

        # ── Student forward + CAM (in-graph for student) ──────────────────────
        with autocast(enabled=(scaler is not None)):
            s_logits, s_cam = student_ext(images)

            losses = loss_fn(
                student_logits=s_logits,
                teacher_logits=t_logits_d,
                student_cam=s_cam,
                teacher_cam=t_cam,
                targets=masks,
            )

        # ── Backward (student only) ────────────────────────────────────────────
        if scaler:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student_ext.model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(student_ext.model.parameters(), grad_clip)
            optimizer.step()

        bs = images.size(0)
        for k in ("total", "task", "kd", "attn"):
            total[k] += losses[k].item() * bs
        total["iou"] += compute_iou(s_logits.detach(), masks) * bs
        n += bs

    return {k: v / n for k, v in total.items()}


@torch.no_grad()
def validate(student, loader, loss_fn, device):
    student.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = student(images)
        loss   = loss_fn(logits, masks)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou  += compute_iou(logits, masks) * bs
        n += bs
    return total_loss / n, total_iou / n


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--fast",    action="store_true")
    parser.add_argument("--teacher", default=None,
                        help="Path to Teacher checkpoint (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Student] Device: {device}")

    fast   = args.fast or cfg["fast_mode"]["enabled"]
    epochs = cfg["fast_mode"]["epochs"] if fast else cfg["train_student"]["epochs"]
    max_s  = cfg["fast_mode"]["max_samples"] if fast else 0

    # ── Data ──────────────────────────────────────────────────────────────────
    root     = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    workers  = cfg["data"]["num_workers"]
    bs       = cfg["train_student"]["batch_size"]

    train_ds = DeepGlobeDataset(root, "train", get_train_transforms(img_size), max_s)
    val_ds   = DeepGlobeDataset(root, "valid", get_val_transforms(img_size),
                                max_s // 4 if fast else 0)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=workers, pin_memory=pin)

    print(f"[Student] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Teacher (frozen) ──────────────────────────────────────────────────────
    teacher = get_teacher(cfg).to(device)
    t_ckpt  = args.teacher or cfg["teacher"]["checkpoint"]
    if os.path.exists(t_ckpt):
        ckpt = torch.load(t_ckpt, map_location=device)
        teacher.load_state_dict(ckpt["model"])
        print(f"[Student] Loaded Teacher from {t_ckpt}")
    else:
        print(f"[Student] WARNING: Teacher checkpoint not found at {t_ckpt}. "
              "Using random weights – run train_teacher.py first.")

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ── Student ───────────────────────────────────────────────────────────────
    student = get_student(cfg).to(device)
    print(f"[Student] Params – Teacher: {format_params(count_parameters(teacher))} "
          f"| Student: {format_params(count_parameters(student))}")

    # ── Grad-CAM extractors ───────────────────────────────────────────────────
    t_layer = get_feature_layer(teacher, f"encoder.{cfg['teacher']['cam_layer']}")
    s_layer = get_feature_layer(student, f"encoder.{cfg['student']['cam_layer']}")

    teacher_ext = GradCAMExtractor(teacher, t_layer)
    student_ext = GradCAMExtractor(student, s_layer)

    # ── Loss / Optimiser ──────────────────────────────────────────────────────
    loss_fn = TotalKDLoss(
        alpha=cfg["loss"]["alpha"],
        beta=cfg["loss"]["beta"],
        gamma=cfg["loss"]["gamma"],
        temperature=cfg["loss"]["kd_temperature"],
    )
    val_loss_fn = BCEDiceLoss()   # validation uses task loss only

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["train_student"]["lr"],
        weight_decay=cfg["train_student"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    use_amp   = cfg["train_student"]["amp"] and device.type == "cuda"
    scaler    = GradScaler("cuda") if use_amp else None
    grad_clip = cfg["train_student"]["grad_clip"]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_iou = 0.0
    log_rows = []
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()

        tr = train_one_epoch(
            teacher_ext, student_ext,
            train_loader, optimizer, loss_fn, val_loss_fn,
            scaler, device, grad_clip
        )
        va_loss, va_iou = validate(student, val_loader, val_loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"total={tr['total']:.4f} task={tr['task']:.4f} "
            f"kd={tr['kd']:.4f} attn={tr['attn']:.4f} "
            f"tr_iou={tr['iou']:.4f} | "
            f"va_loss={va_loss:.4f} va_iou={va_iou:.4f} | "
            f"{elapsed:.0f}s"
        )

        log_rows.append({
            "epoch": epoch + 1,
            **{f"tr_{k}": v for k, v in tr.items()},
            "va_loss": va_loss, "va_iou": va_iou,
        })

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(
                {"epoch": epoch, "model": student.state_dict(),
                 "iou": best_iou, "cfg": cfg},
                "checkpoints/student_best.pth",
            )
            print(f"  ↑  Best Student checkpoint saved (IoU={best_iou:.4f})")

        every = cfg["train_student"]["save_every"]
        if (epoch + 1) % every == 0:
            torch.save(
                {"epoch": epoch, "model": student.state_dict(), "cfg": cfg},
                f"checkpoints/student_epoch{epoch+1:03d}.pth",
            )

    # ── Cleanup & log ─────────────────────────────────────────────────────────
    teacher_ext.remove_hooks()
    student_ext.remove_hooks()

    import pandas as pd
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(log_rows).to_csv("outputs/student_training_log.csv", index=False)
    print(f"\n[Student] Training done. Best val IoU: {best_iou:.4f}")
    print("Log saved to outputs/student_training_log.csv")


if __name__ == "__main__":
    main()
