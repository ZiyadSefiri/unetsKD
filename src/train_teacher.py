"""
train_teacher.py – Phase 1: train the ResNet-101 U-Net Teacher.

Usage
-----
  # Full training
  python src/train_teacher.py

  # Fast debug (few images, 3 epochs)
  python src/train_teacher.py --fast

  # Resume from a checkpoint
  python src/train_teacher.py --resume checkpoints/teacher_epoch10.pth
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

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import build_dataloaders, DeepGlobeDataset, get_train_transforms, get_val_transforms
from src.models  import get_teacher, count_parameters, format_params
from src.losses  import BCEDiceLoss
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
#  Metric helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Pixel-level binary IoU (Road class)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union        = preds.sum() + targets.sum() - intersection
    return (intersection / (union + 1e-8)).item()


# ──────────────────────────────────────────────────────────────────────────────
#  One-epoch routines
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, grad_clip):
    model.train()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast("cuda", enabled=(scaler is not None)):
            logits = model(images)
            loss   = loss_fn(logits, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou  += compute_iou(logits.detach(), masks) * bs
        n += bs

    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0

    for images, masks in tqdm(loader, desc="  val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
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
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fast",   action="store_true",
                        help="Debug mode: few samples, 3 epochs")
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    
    # Robust device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            # Test if CUDA actually works
            torch.cuda.init()
        except Exception as e:
            print(f"[Warning] CUDA detected but failed to initialize: {e}")
            print("Falling back to CPU. (Check your PyTorch/CUDA installation)")
            device = torch.device("cpu")
            
    print(f"[Teacher] Device: {device}")

    fast    = args.fast or cfg["fast_mode"]["enabled"]
    epochs  = cfg["fast_mode"]["epochs"] if fast else cfg["train_teacher"]["epochs"]
    max_s   = cfg["fast_mode"]["max_samples"] if fast else 0

    # ── Data ──────────────────────────────────────────────────────────────────
    root     = cfg["data"]["root"]
    img_size = cfg["data"]["img_size"]
    workers  = cfg["data"]["num_workers"]
    bs       = cfg["train_teacher"]["batch_size"]

    train_ds = DeepGlobeDataset(root, "train", get_train_transforms(img_size), max_s)
    val_ds   = DeepGlobeDataset(root, "valid", get_val_transforms(img_size),
                                max_s // 4 if fast else 0)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=workers, pin_memory=pin)

    print(f"[Teacher] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_teacher(cfg).to(device)
    print(f"[Teacher] Params: {format_params(count_parameters(model))}")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Teacher] Resumed from epoch {start_epoch}")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train_teacher"]["lr"],
        weight_decay=cfg["train_teacher"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    loss_fn   = BCEDiceLoss()
    use_amp   = cfg["train_teacher"]["amp"] and device.type == "cuda"
    scaler    = GradScaler("cuda") if use_amp else None
    grad_clip = cfg["train_teacher"]["grad_clip"]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_iou   = 0.0
    log_rows   = []
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, grad_clip
        )
        va_loss, va_iou = validate(model, val_loader, loss_fn, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch+1:03d}/{epochs}] "
            f"tr_loss={tr_loss:.4f} tr_iou={tr_iou:.4f} | "
            f"va_loss={va_loss:.4f} va_iou={va_iou:.4f} | "
            f"{elapsed:.0f}s"
        )

        log_rows.append({
            "epoch": epoch + 1,
            "tr_loss": tr_loss, "tr_iou": tr_iou,
            "va_loss": va_loss, "va_iou": va_iou,
        })

        # ── Save best ─────────────────────────────────────────────────────────
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "iou": best_iou, "cfg": cfg},
                "checkpoints/teacher_best.pth",
            )
            print(f"  ↑  Best Teacher checkpoint saved (IoU={best_iou:.4f})")

        # ── Periodic checkpoint ────────────────────────────────────────────────
        every = cfg["train_teacher"]["save_every"]
        if (epoch + 1) % every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                f"checkpoints/teacher_epoch{epoch+1:03d}.pth",
            )

    # ── Save training log ─────────────────────────────────────────────────────
    import pandas as pd
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(log_rows).to_csv("outputs/teacher_training_log.csv", index=False)
    print(f"\n[Teacher] Training done. Best val IoU: {best_iou:.4f}")
    print("Log saved to outputs/teacher_training_log.csv")


if __name__ == "__main__":
    main()
