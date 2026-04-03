"""轻量 RGB patch 编码器：输出与 MultimodalClassifier 扩散分支一致的 token 序列。"""
from __future__ import annotations

import torch
import torch.nn as nn


class LightweightRgbEncoder(nn.Module):
    """
    输入 B×3×H×W（通常为 11×11 patch），输出 B×num_tokens×d_model，
    与 len(diffusion_ts)×len(feat_scales) 个 RGB token 对齐。
    """

    def __init__(
        self,
        in_ch: int = 3,
        patch_h: int = 11,
        patch_w: int = 11,
        d_model: int = 256,
        num_tokens: int = 3,
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
