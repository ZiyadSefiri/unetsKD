"""
gradcam.py – Hook-based Grad-CAM extractor.

Design goal: gradients must remain in-graph so that L_attn
(MSE between Teacher CAM and Student CAM) can back-propagate
through the Student network during distillation training.

Usage
-----
    extractor = GradCAMExtractor(model, target_layer)
    logits, cam = extractor(x)          # cam: (B,1,H,W), values in [0,1]
    extractor.remove_hooks()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExtractor(nn.Module):
    """
    Differentiable Grad-CAM that keeps the computation graph intact.

    Parameters
    ----------
    model        : nn.Module – full segmentation model
    target_layer : nn.Module – the conv layer to visualise (encoder bottleneck)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        super().__init__()
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        # output: (B, C, H', W')  – keep on same device, detach not needed
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: (B, C, H', W')
        self._gradients = grad_output[0]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args
        ----
        x : (B, 3, H, W)

        Returns
        -------
        logits : (B, 1, H, W)  – raw model output (no sigmoid)
        cam    : (B, 1, H, W)  – Grad-CAM heatmap, normalised to [0,1]
        """
        logits = self.model(x)           # full forward pass

        # ── Compute gradients ─────────────────────────────────────────────────
        # Scalar score: mean of ALL logits (for segmentation Grad-CAM).
        # This focuses attention on where the model is most "active" overall,
        # which captures road boundary focus even without a class index.
        score = logits.mean()

        self.model.zero_grad()
        score.backward(retain_graph=True)   # keep graph for outer loss.backward()

        # ── Build CAM ────────────────────────────────────────────────────────
        gradients   = self._gradients    # (B, C, H', W')
        activations = self._activations  # (B, C, H', W')

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. "
                "Make sure target_layer is part of the forward graph."
            )

        # α_k = global average pooling of gradients  → (B, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations  → (B, 1, H', W')
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU + upsample to input resolution
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Normalise per sample to [0, 1]
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max  = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return logits, cam

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def remove_hooks(self):
        """Call this after training to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import yaml
    from src.models import get_teacher, get_feature_layer

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = get_teacher(cfg)
    target = get_feature_layer(model, f"encoder.{cfg['teacher']['cam_layer']}")

    extractor = GradCAMExtractor(model, target)
    x = torch.randn(2, 3, 256, 256)   # small for speed

    logits, cam = extractor(x)
    print(f"Logits : {logits.shape}")
    print(f"CAM    : {cam.shape}  min={cam.min():.3f}  max={cam.max():.3f}")

    extractor.remove_hooks()
    print("Hooks removed. Done.")
