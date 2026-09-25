import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeL1Loss(nn.Module):
    """
    Relative L1 Loss for HDR Monte Carlo Radiance.
    Balances dark shadows and intense bright specular highlights.
    
    Formula: |pred - target| / (|target| + eps)
    """
    def __init__(self, eps: float = 0.01):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        norm = torch.abs(target) + self.eps
        return torch.mean(diff / norm)


class GradientDifferenceLoss(nn.Module):
    """
    Penalizes edge blurring by comparing spatial image gradients in X and Y directions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If input is 5D sequence [B, T, C, H, W], flatten B and T
        if pred.dim() == 5:
            b, t, c, h, w = pred.shape
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)

        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = torch.mean(torch.abs(pred_dx - target_dx))
        loss_dy = torch.mean(torch.abs(pred_dy - target_dy))
        return loss_dx + loss_dy


class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes temporal flicker (boiling noise) between consecutive frames.
    For sequences [B, T, C, H, W].
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        if pred_seq.dim() != 5 or pred_seq.size(1) < 2:
            return torch.tensor(0.0, device=pred_seq.device, requires_grad=True)

        pred_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
        target_diff = target_seq[:, 1:] - target_seq[:, :-1]

        return torch.mean(torch.abs(pred_diff - target_diff))


class CombinedDenoisingLoss(nn.Module):
    """
    Weighted combination of Relative L1, Gradient Edge Loss, and Temporal Consistency.
    """
    def __init__(
        self,
        weight_l1: float = 1.0,
        weight_grad: float = 0.2,
        weight_temporal: float = 0.5,
        eps: float = 0.01,
    ):
        super().__init__()
        self.rel_l1 = RelativeL1Loss(eps=eps)
        self.grad_loss = GradientDifferenceLoss()
        self.temporal_loss = TemporalConsistencyLoss()

        self.w_l1 = weight_l1
        self.w_grad = weight_grad
        self.w_temp = weight_temporal

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, is_sequence: bool = False
    ) -> dict:
        l1 = self.rel_l1(pred, target)
        grad = self.grad_loss(pred, target)
        temp = self.temporal_loss(pred, target) if is_sequence else torch.tensor(0.0, device=pred.device)

        total = self.w_l1 * l1 + self.w_grad * grad + (self.w_temp * temp if is_sequence else 0.0)

        return {
            "loss": total,
            "rel_l1": l1,
            "grad": grad,
            "temporal": temp,
        }


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return float('inf')
        return 20.0 * math.log10(max_val / math.sqrt(mse))


def calc_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates simplified Structural Similarity Index (SSIM)."""
    with torch.no_grad():
        # Flatten to 4D if 5D
        if pred.dim() == 5:
            pred = pred.view(-1, *pred.shape[2:])
            target = target.view(-1, *target.shape[2:])
            
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)

        sigma_x_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
        sigma_y_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return torch.mean(ssim_map).item()
