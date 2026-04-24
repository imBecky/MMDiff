"""轻量 RGB patch 编码器：从 RGB patch 直接映射为与融合头维度一致的 token 序列。

说明：类名/文件名中的 student 仅为历史命名；当前实现为独立轻量 CNN，不依赖教师网络或蒸馏监督。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LightweightRgbEncoder(nn.Module):
    """
    输入 B×3×H×W（通常为 11×11 patch），输出 B×num_tokens×d_model。

    结构：两层 Conv-BN-ReLU stem → 全局平均池化 → 线性层展开为 num_tokens 个 d_model 维 token。
    当前默认输出 1 个 RGB token；若教师缓存有多 token，会在上游做聚合后再监督/融合。
    """

    def __init__(
        self,
        in_ch: int = 3,
        patch_h: int = 11,
        patch_w: int = 11,
        d_model: int = 256,
        num_tokens: int = 1,
        hidden: int = 128,
    ):
        super().__init__()
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, self.num_tokens * self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.pool(h).flatten(1)
        out = self.fc(h)
        return out.view(-1, self.num_tokens, self.d_model)
