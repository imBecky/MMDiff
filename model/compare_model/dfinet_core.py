"""
Gao et al. 2022 DFINet（Depthwise Feature Interaction Network）PyTorch 实现。

论文 / 官方 notebook: https://github.com/formango/HSI_MSI_Multisource_Classification
（DFINet.ipynb；原代码中 HSI 称 hyperspectral，MSI 对应本项目的 LiDAR/多光谱 patch）

说明：全连接输入维应为 depthwise 互相关后的展平长度。互相关以整幅特征图为核，stem 输出为 h'×w'（h'=w'=patch_size-4）
时空间维缩为 1×1，故为 128；此前误用 128*h'*w' 会导致与 `fc1` 权重形状不一致。
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _feat_hw_after_stem(patch_size: int) -> Tuple[int, int]:
    """HSINet/MSINet：conv1 same padding，conv2/conv3 无 padding，各边 -2。"""
    s = patch_size
    s = s - 2
    s = s - 2
    return s, s


class _LayerNorm(nn.Module):
    """与 notebook LayerNorm 一致（最后一维）。"""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class HSINet(nn.Module):
    """notebook HSINet。"""

    def __init__(self, channel_hsi: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_hsi, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class MSINet(nn.Module):
    """notebook MSINet（LiDAR/MSI 支路）。"""

    def __init__(self, channel_msi: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_msi, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class CAM(nn.Module):
    """Cross attention module（与 notebook 一致）。"""

    def __init__(self) -> None:
        super().__init__()
        k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def get_attention(self, a: torch.Tensor) -> torch.Tensor:
        input_a = a
        a = a.mean(3)
        a = a.transpose(1, 3)
        a = self.conv(a.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        a = a.transpose(1, 3)
        a = a.unsqueeze(3)
        a = torch.mean(input_a * a, -1)
        a = F.softmax(a / 0.025, dim=-1) + 1.0
        return a

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)
        f1 = f1.view(b, n1, c, -1)
        f2 = f2.view(b, n2, c, -1)
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
        f2_norm = f2_norm.unsqueeze(1)
        a1 = torch.matmul(f1_norm, f2_norm)
        a2 = a1.transpose(3, 4)
        a1 = self.get_attention(a1)
        a2 = self.get_attention(a2)
        f1 = f1 * a1
        f1 = f1.view(b, c, h, w)
        f2 = f2 * a2
        f2 = f2.view(b, c, h, w)
        return f1, f2


class DFINetBackbone(nn.Module):
    """DFINet 分类头输出 logits（无 softmax）。"""

    def __init__(
        self,
        patch_size: int,
        channel_hsi: int,
        channel_msi: int,
        class_num: int,
    ) -> None:
        super().__init__()
        h, w = _feat_hw_after_stem(patch_size)
        if h < 1 or w < 1:
            raise ValueError(f'DFINet 需要足够大的 patch_size，当前 {patch_size} 导致特征图非正')

        self.featnet1 = HSINet(channel_hsi)
        self.featnet2 = MSINet(channel_msi)
        self.cam = CAM()
        # 互相关以另一支路整幅 (h×w) 特征图为 depthwise 核，valid conv 输出空间为 1×1 → 展平维 128（与原版 Linear(1*1*128,64)）
        kh, kw = h, w
        out_h, out_w = h - kh + 1, w - kw + 1
        flat_dim = 128 * out_h * out_w
        self.proj_norm = _LayerNorm(64)
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, class_num)
        self.dropout = nn.Dropout(p=0.5)

    @staticmethod
    def xcorr_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        return out.view(batch, channel, out.size(2), out.size(3))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        feature_1 = self.featnet1(x)
        feature_2 = self.featnet2(y)
        hsi_feat = feature_1.unsqueeze(1)
        lidar_feat = feature_2.unsqueeze(1)
        hsi, lidar = self.cam(hsi_feat, lidar_feat)
        xa = self.xcorr_depthwise(hsi, lidar)
        yb = self.xcorr_depthwise(lidar, hsi)
        x1 = xa.contiguous().view(xa.size(0), -1)
        y1 = yb.contiguous().view(yb.size(0), -1)
        z = x1 + y1
        z = F.relu(self.proj_norm(self.fc1(z)))
        z = self.dropout(z)
        logits = self.fc2(z)
        if return_aux:
            return feature_1, feature_2, x1, y1, logits
        return logits
