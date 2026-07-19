from typing import Literal

import torch
from einops import reduce
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

from metrics.emd_assignment import emd_module as EMD


def mean_squared_error(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


def mean_squared_error_sum(pred: Tensor, gt: Tensor) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "sum")
    return loss


def l1(pred: Tensor, gt: Tensor) -> Tensor:
    loss = l1_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


def rodr_x0(pred_x0: Tensor, clean: Tensor, noisy: Tensor, tangent_weight: float = 1.0, eps: float = 1e-8) -> Tensor:
    """RODR loss on denoising displacement pred_x0 - noisy against clean - noisy."""
    pred_dir = pred_x0 - noisy
    gt_dir = clean - noisy
    normal = gt_dir / (torch.sqrt(torch.sum(gt_dir**2, dim=1, keepdim=True)) + eps)
    pred_normal = torch.sum(pred_dir * normal, dim=1, keepdim=True) * normal
    pred_tangent = pred_dir - pred_normal
    normal_loss = mse_loss(pred_normal, gt_dir, reduction="none")
    tangent_loss = pred_tangent**2
    loss = normal_loss + tangent_weight * tangent_loss
    return reduce(loss, "b ... -> b", "mean")


class EmdLoss:
    def __init__(self):
        self.emd = EMD.emdModule()

    def __call__(self, pred: Tensor, gt: Tensor) -> Tensor:
        if pred.shape[-1] != 3:
            pred = pred.transpose(1, 2)
        if gt.shape[-1] != 3:
            gt = gt.transpose(1, 2)

        distances, _ = self.emd(pred, gt, eps=0.005, iters=50)

        loss = torch.sqrt(distances)
        loss = reduce(loss, "b ... -> b", "mean")
        return loss


def get_loss(type: Literal["mse", "mse_sum", "l1", "emd", "rodr"]) -> callable:
    """

    Args:
        type (Literal["mse", "mse_sum", "l1", "emd"]): The type of loss to get.

    Returns:
        callable: The loss function.
    """
    if type == "mse":
        return mean_squared_error
    if type == "mse_sum":
        return mean_squared_error_sum
    if type == "l1":
        return l1
    if type == "emd":
        return EmdLoss()
    if type == "rodr":
        return rodr_x0
    raise ValueError(f"Unknown loss type: {type}")
