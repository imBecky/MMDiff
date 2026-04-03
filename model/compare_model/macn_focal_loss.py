"""
MACN 官方 FocalLoss（like413/MACN trento/FocalLoss.py），去掉已废弃的 Variable。

论文对比实验：与 trentoTrain.py 一致使用 FocalLoss + Adam。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MACNFocalLoss(nn.Module):
    """与官方 FocalLoss 数值一致；class_num 随数据集类别数变化。"""

    def __init__(
        self,
        class_num: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = F.softmax(inputs, dim=1)
        class_mask = torch.zeros_like(inputs)
        class_mask.scatter_(1, targets.unsqueeze(1).long(), 1.0)
        probs = (p * class_mask).sum(1).view(-1, 1).clamp(min=1e-12)
        log_p = probs.log()
        batch_loss = -self.alpha * torch.pow(1.0 - probs, self.gamma) * log_p
        if self.size_average:
            return batch_loss.mean()
        return batch_loss.sum()
