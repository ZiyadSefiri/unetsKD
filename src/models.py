"""
models.py – Teacher and Student U-Net definitions.

Both built with segmentation_models_pytorch (smp) for clean encoder swap.

Teacher : ResNet-101  backbone  (~65 M params)
Student : MobileNetV2 backbone  (~ 5 M params)

The helper `get_feature_layer()` returns the exact sub-module that Grad-CAM
should hook into, resolved by the layer name string in config.yaml.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from functools import reduce
import operator


# ──────────────────────────────────────────────────────────────────────────────
#  Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_unet(encoder_name: str, encoder_weights: str) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,          # raw logits – sigmoid applied in loss
    )


def get_teacher(cfg: dict) -> nn.Module:
    """Return Teacher U-Net (ResNet-101)."""
    return _build_unet(
        encoder_name=cfg["teacher"]["encoder"],
        encoder_weights=cfg["teacher"]["encoder_weights"],
    )


def get_student(cfg: dict) -> nn.Module:
    """Return Student U-Net (MobileNetV2)."""
    return _build_unet(
        encoder_name=cfg["student"]["encoder"],
        encoder_weights=cfg["student"]["encoder_weights"],
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Layer resolver  (dot-path → nn.Module)
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_layer(model: nn.Module, layer_path: str) -> nn.Module:
    """
    Traverse a dot-separated path through the model to get the target layer.

    Examples
    --------
    get_feature_layer(teacher, "encoder.layer4")
    get_feature_layer(student, "encoder.features.18")
    """
    parts = layer_path.split(".")
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer


# ──────────────────────────────────────────────────────────────────────────────
#  Param / FLOP summary
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f} K"
    return str(n)


# ──────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    teacher = get_teacher(cfg)
    student = get_student(cfg)

    print(f"Teacher params : {format_params(count_parameters(teacher))}")
    print(f"Student params : {format_params(count_parameters(student))}")

    # Verify Grad-CAM target layers exist
    t_layer = get_feature_layer(teacher, f"encoder.{cfg['teacher']['cam_layer']}")
    s_layer_path = f"encoder.{cfg['student']['cam_layer']}"
    s_layer = get_feature_layer(student, s_layer_path)
    print(f"Teacher CAM layer : {t_layer.__class__.__name__}")
    print(f"Student CAM layer : {s_layer.__class__.__name__}")

    # Forward pass check
    import torch
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        print(f"Teacher output : {teacher(x).shape}")
        print(f"Student output : {student(x).shape}")
