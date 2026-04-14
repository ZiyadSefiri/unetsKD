"""
dataset.py – DeepGlobe Road Extraction Dataset loader.

Each sample is a pair:
  sat_image : RGB satellite tile   (H × W × 3 uint8)
  mask      : binary road mask     (H × W, float32, values in {0,1})

Augmentation (train only):
  RandomCrop → HorizontalFlip → VerticalFlip → RandomRotate90
  → ColorJitter → GaussianBlur → Normalize → ToTensorV2
"""

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import yaml


# ──────────────────────────────────────────────────────────────────────────────
#  Augmentation pipelines
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0, p=0.2
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DeepGlobeDataset(Dataset):
    """
    Args:
        root        : path to the archive(1) folder
        split       : 'train' | 'valid' | 'test'
        transforms  : Albumentations Compose pipeline
        max_samples : if > 0, subsample for fast debug mode
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: A.Compose = None,
        max_samples: int = 0,
    ):
        self.root = root
        self.split = split
        self.transforms = transforms

        meta = pd.read_csv(os.path.join(root, "metadata.csv"))
        meta = meta[meta["split"] == split].reset_index(drop=True)

        if max_samples > 0:
            meta = meta.sample(
                n=min(max_samples, len(meta)), random_state=42
            ).reset_index(drop=True)

        self.image_paths = [
            os.path.join(root, p) for p in meta["sat_image_path"]
        ]

        # Build mask paths: use metadata if valid, else derive from sat path
        mask_paths = []
        for i, row in meta.iterrows():
            mp = row.get("mask_path", None)
            if isinstance(mp, str) and mp.strip():
                # Use the metadata mask path
                candidate = os.path.join(root, mp)
            else:
                # Derive: replace _sat.jpg → _mask.png
                sat_p = os.path.join(root, row["sat_image_path"])
                candidate = sat_p.replace("_sat.jpg", "_mask.png")
            mask_paths.append(candidate if os.path.exists(candidate) else None)

        self.mask_paths = mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # ── Load image ────────────────────────────────────────────────────────
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # (H,W,3) uint8

        # ── Load mask (may not exist for test split) ──────────────────────────
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)        # (H,W) float32
        else:
            # No ground-truth mask available (test split without labels)
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)

        # ── Augment ───────────────────────────────────────────────────────────
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]   # (3,H,W) float32 tensor
            mask  = augmented["mask"]    # (H,W)   float32 tensor

        mask = mask.unsqueeze(0)         # (1,H,W) for BCEWithLogitsLoss

        return image, mask


# ──────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, fast: bool = False):
    """
    Returns (train_loader, val_loader, test_loader) from config dict.
    """
    root       = cfg["data"]["root"]
    img_size   = cfg["data"]["img_size"]
    workers    = cfg["data"]["num_workers"]
    batch_tr   = cfg["train_student"]["batch_size"]  # caller overrides if needed
    max_s      = cfg["fast_mode"]["max_samples"] if fast else 0

    train_ds = DeepGlobeDataset(
        root, "train",
        transforms=get_train_transforms(img_size),
        max_samples=max_s,
    )
    val_ds = DeepGlobeDataset(
        root, "valid",
        transforms=get_val_transforms(img_size),
        max_samples=max_s // 4 if fast else 0,
    )
    test_ds = DeepGlobeDataset(
        root, "test",
        transforms=get_val_transforms(img_size),
        max_samples=max_s // 4 if fast else 0,
    )

    pin = True  # pin memory for fast GPU transfer
    train_loader = DataLoader(train_ds, batch_size=batch_tr, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,        shuffle=False,
                              num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1,        shuffle=False,
                              num_workers=workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, torch

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ds = DeepGlobeDataset(
        root=cfg["data"]["root"],
        split="train",
        transforms=get_train_transforms(cfg["data"]["img_size"]),
        max_samples=10,
    )
    img, msk = ds[0]
    print(f"Image shape : {img.shape}  dtype={img.dtype}")
    print(f"Mask  shape : {msk.shape}  min={msk.min():.2f}  max={msk.max():.2f}")
    print(f"Dataset size: {len(ds)}")
