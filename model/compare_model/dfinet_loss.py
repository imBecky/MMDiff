"""
DFINet 论文 / notebook 联合损失（Gao et al. 2022）。

与 DFINet.ipynb 中 calc_loss 一致：L = CE + α·L_dist + β·L_cons，
其中 L_cons = MSE(feat_hsi, feat_msi)，L_dist 为跨模态/同模态相似度项。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _calc_label_sim(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """batch 内标签共现矩阵（与 notebook calc_label_sim 等价，device 安全）。"""
    device = labels.device
    batch_size = labels.shape[0]
    label = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float32)
    label.scatter_(1, labels.unsqueeze(1).long(), 1.0)
    return label @ label.t()


def _cos_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """notebook cos：余弦相似度 / 2。"""
    num = x @ y.t()
    den = (
        (x ** 2).sum(1, keepdim=True).sqrt()
        @ (y ** 2).sum(1, keepdim=True).sqrt().t()
    ).clamp(min=1e-6)
    return num / den / 2.0


def dfinet_calc_loss(
    feature_1: torch.Tensor,
    feature_2: torch.Tensor,
    hsi_1: torch.Tensor,
    lidar_1: torch.Tensor,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.01,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    feature_* : (B, C, H, W) stem 输出；hsi_1/lidar_1 : (B, D) 互相关展平向量；
    outputs : logits；labels : (B,) long。
    """
    theta = _cos_batch(hsi_1, lidar_1)
    sim = _calc_label_sim(labels, num_classes)
    theta1 = _cos_batch(hsi_1, hsi_1)
    theta2 = _cos_batch(lidar_1, lidar_1)

    term1 = ((1 + theta.exp()).log() + sim * theta).mean()
    term2 = ((1 + theta1.exp()).log() + sim * theta1).mean()
    term3 = ((1 + theta2.exp()).log() + sim * theta2).mean()
    loss2 = term1 + term2 + term3

    loss3 = F.cross_entropy(outputs, labels.long())
    loss1 = (feature_1 - feature_2).pow(2).mean()

    return loss3 + alpha * loss2 + beta * loss1
