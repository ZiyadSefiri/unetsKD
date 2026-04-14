"""
losses.py – All loss components for the KD pipeline.

L_total = α · L_task  +  β · L_KD  +  γ · L_attn

L_task  : BCE + Dice          (against ground-truth binary mask)
L_KD    : KL-Divergence       (Student soft-logits ↔ Teacher soft-logits)
L_attn  : MSE                 (Student Grad-CAM ↔ Teacher Grad-CAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Segmentation loss  (L_task)
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss.

    Inputs
    ------
    logits : (B, 1, H, W)  – raw pre-sigmoid outputs
    targets: (B, 1, H, W)  – binary float masks {0, 1}
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice  (standard combo for imbalanced binary segmentation)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight  * self.bce(logits, targets) +
            self.dice_weight * self.dice(logits, targets)
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Knowledge Distillation loss  (L_KD)
# ──────────────────────────────────────────────────────────────────────────────

class KDLoss(nn.Module):
    """
    Soft-label KD via pixel-wise KL-Divergence.

    For binary segmentation we treat each pixel as a 2-class problem:
      p = [σ(logit), 1 - σ(logit)]

    Temperature T sharpens/smooths the distributions.

    Args
    ----
    temperature : float  – temperature T (≥1); higher → softer targets
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,   # (B, 1, H, W)
        teacher_logits: torch.Tensor,   # (B, 1, H, W)
    ) -> torch.Tensor:

        # Build 2-class soft distributions along the channel dim
        def to_soft_dist(logits):
            p_road     = torch.sigmoid(logits / self.T)
            p_bg       = 1.0 - p_road
            return torch.cat([p_road, p_bg], dim=1)   # (B, 2, H, W)

        s_dist = to_soft_dist(student_logits)
        t_dist = to_soft_dist(teacher_logits)

        # KL(student || teacher) — averaged over all pixels
        loss = F.kl_div(
            input  = torch.log(s_dist + 1e-8),
            target = t_dist,
            reduction="batchmean",
        )
        # Scale by T² as in Hinton et al., 2015
        return loss * (self.T ** 2)


# ──────────────────────────────────────────────────────────────────────────────
#  Attention alignment loss  (L_attn)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionAlignmentLoss(nn.Module):
    """
    MSE between the Student's Grad-CAM heatmap and the Teacher's.

    Both maps are expected to be already normalised to [0,1] by GradCAMExtractor.

    Args
    ----
    student_cam : (B, 1, H, W)
    teacher_cam : (B, 1, H, W)  – detached from Teacher graph (no grad for teacher)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_cam: torch.Tensor,
        teacher_cam: torch.Tensor,
    ) -> torch.Tensor:
        # Teacher CAM should NOT flow gradients back through the Teacher
        return self.mse(student_cam, teacher_cam.detach())


# ──────────────────────────────────────────────────────────────────────────────
#  Composite loss  (L_total)
# ──────────────────────────────────────────────────────────────────────────────

class TotalKDLoss(nn.Module):
    """
    L_total = alpha * L_task  +  beta * L_KD  +  gamma * L_attn

    Args
    ----
    alpha, beta, gamma : loss weights  (should sum to 1, but not required)
    temperature        : KD temperature
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta:  float = 0.25,
        gamma: float = 0.25,
        temperature: float = 4.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.l_task  = BCEDiceLoss()
        self.l_kd    = KDLoss(temperature=temperature)
        self.l_attn  = AttentionAlignmentLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_cam:    torch.Tensor,
        teacher_cam:    torch.Tensor,
        targets:        torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with individual losses and the weighted total.
        """
        task  = self.l_task(student_logits, targets)
        kd    = self.l_kd(student_logits, teacher_logits)
        attn  = self.l_attn(student_cam, teacher_cam)

        total = self.alpha * task + self.beta * kd + self.gamma * attn

        return {
            "total": total,
            "task":  task,
            "kd":    kd,
            "attn":  attn,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, H, W = 2, 64, 64

    logits_t = torch.randn(B, 1, H, W)
    logits_s = torch.randn(B, 1, H, W, requires_grad=True)
    cam_t    = torch.rand(B, 1, H, W)
    cam_s    = torch.rand(B, 1, H, W, requires_grad=True)
    targets  = (torch.rand(B, 1, H, W) > 0.8).float()

    loss_fn = TotalKDLoss(alpha=0.5, beta=0.25, gamma=0.25, temperature=4.0)
    losses  = loss_fn(logits_s, logits_t, cam_s, cam_t, targets)

    for k, v in losses.items():
        print(f"  {k:6s}: {v.item():.4f}")

    losses["total"].backward()
    print("Backward ✓  (student logit grad norm:",
          logits_s.grad.norm().item(), ")")
