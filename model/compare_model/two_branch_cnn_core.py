"""
Xu et al. 2017 双分支 CNN（多源遥感分类）PyTorch 复现骨架。

对应 Keras 实现: https://github.com/BUCT-Vision/Two-branch-CNN-Multisource-RS-classification
（models.py: simple_cnn_branch + pixel_branch + cascade_Net + 融合头）

说明：原仓库分阶段训练 HSI / LiDAR 再 finetune；此处为端到端训练，结构与 finetune 拼接后的
主干一致（HSI 空间支路 + 光谱 1D 支路 + LiDAR cascade，再接 BN/Dropout/Dense 分类头）。
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def _ceil_half(n: int) -> int:
    """与 Keras MaxPool2D padding=same、ceil 行为一致：边长 ceil(n/2)。"""
    return (n + 1) // 2


def _fusion_feature_dim(patch_size: int, hsi_channels: int) -> tuple[int, int, int, int]:
    """返回 (simple_flat, pixel_flat, lidar_flat, total)。"""
    ph = _ceil_half(patch_size)
    simple_dim = ph * ph * 512
    # Conv1d valid: L -> L-10 -> L-12 -> floor((L-12)/2)
    l1 = hsi_channels - 10
    l2 = l1 - 2
    l3 = l2 // 2
    pixel_dim = l3 * 128
    lidar_dim = ph * ph * 128
    return simple_dim, pixel_dim, lidar_dim, simple_dim + pixel_dim + lidar_dim


class _CascadeBlock(nn.Module):
    """cascade_block（与 Keras 版逐层对应）。"""

    def __init__(self, in_channels: int, nb_filter: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1_1 = nn.Conv2d(in_channels, nb_filter * 2, kernel_size, padding=pad)
        self.bn1_1 = nn.BatchNorm2d(nb_filter * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(nb_filter * 2, nb_filter, 1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(nb_filter)
        self.skip_conv = nn.Conv2d(in_channels, nb_filter * 2, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nb_filter, nb_filter * 2, kernel_size, padding=pad)
        self.bn2_1 = nn.BatchNorm2d(nb_filter * 2)
        self.conv2_2 = nn.Conv2d(nb_filter * 2, nb_filter, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(nb_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.relu1(self.bn1_1(self.conv1_1(x)))
        c2 = F.leaky_relu(self.bn1_2(self.conv1_2(c1)), 0.2, inplace=True)
        skip = self.skip_conv(x)
        c3 = F.leaky_relu(self.bn2_1(self.conv2_1(c2) + skip), 0.2, inplace=True)
        c4 = self.bn2_2(self.conv2_2(c3))
        return F.leaky_relu(c2 + c4, 0.2, inplace=True)


class _CascadeNet(nn.Module):
    """LiDAR 支路 cascade_Net。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        f = [16, 32, 64, 96, 128, 192, 256, 512]
        self.conv0 = nn.Conv2d(in_channels, f[2], 3, padding=1)
        self.cb1 = _CascadeBlock(f[2], f[2])
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.cb2 = _CascadeBlock(f[2], f[4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x = self.cb1(x)
        x = self.pool(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.cb2(x)
        return x.flatten(1)


class _SimpleCnnBranch(nn.Module):
    """HSI 空间支路 simple_cnn_branch（small_mode=False，通道与 Keras 硬编码一致）。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 512, 1, padding=0)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.pool(x)
        return x.flatten(1)


class _PixelBranch(nn.Module):
    """中心像元光谱 1D 支路 pixel_branch（Conv1d 64→128，与 Keras 一致）。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv1d(1, 64, 11, padding=0)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 128, 3, padding=0)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.pool(x)
        return x.flatten(1)


class TwoBranchCNNBackbone(nn.Module):
    """双分支特征提取 + 融合分类头（输出 logits）。"""

    def __init__(
        self,
        patch_size: int,
        hsi_channels: int,
        lidar_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hsi_channels < 13:
            raise ValueError(
                'Two-branch CNN 的 pixel_branch 需要 hsi_channels >= 13（Conv1D 有效感受野）'
            )
        _, _, _, fusion_dim = _fusion_feature_dim(patch_size, hsi_channels)
        self.simple = _SimpleCnnBranch(hsi_channels)
        self.pixel = _PixelBranch()
        self.lidar_net = _CascadeNet(lidar_channels)

        self.bn_fusion = nn.BatchNorm1d(fusion_dim)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        _, _, h, w = hsi.shape
        cy, cx = h // 2, w // 2
        pixel_in = hsi[:, :, cy, cx].unsqueeze(1)

        h_s = self.simple(hsi)
        h_p = self.pixel(pixel_in)
        h_hsi = torch.cat([h_s, h_p], dim=1)
        h_l = self.lidar_net(lidar)
        z = torch.cat([h_hsi, h_l], dim=1)

        z = self.bn_fusion(z)
        z = self.drop(z)
        z = F.leaky_relu(self.fc1(z), 0.2, inplace=True)
        return self.fc2(z)
