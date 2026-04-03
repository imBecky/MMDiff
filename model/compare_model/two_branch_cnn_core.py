"""
Xu et al. 2017 双分支 CNN（多源遥感分类）PyTorch 实现。

对应 Keras: https://github.com/BUCT-Vision/Two-branch-CNN-Multisource-RS-classification
（models.py: hsi_branch / lidar_branch / finetune_Net）

训练协议（与官方 main.py 一致）：
  1) 仅训 HSI 支路（simple_cnn + pixel → Dropout → Dense）
  2) 仅训 LiDAR 支路（cascade_Net → Dropout → Dense）
  3) 加载两路权重，去掉末端分类头，冻结支路，仅训融合头（BN + Dropout + Dense(128) + logits）
     优化器：SGD(lr=0.005, momentum=1e-6)；支路 Adam lr=1e-4

端到端单阶段训练（非论文协议）仅用于调试：环境变量 MMDIFF_TWO_BRANCH_END2END=1。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ceil_half(n: int) -> int:
    """与 Keras MaxPool2D padding=same、ceil 行为一致：边长 ceil(n/2)。"""
    return (n + 1) // 2


def fusion_feature_dims(patch_size: int, hsi_channels: int) -> tuple[int, int, int, int]:
    """返回 (simple_flat, pixel_flat, lidar_flat, total)。"""
    ph = _ceil_half(patch_size)
    simple_dim = ph * ph * 512
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
    """HSI 空间支路 simple_cnn_branch（small_mode=False）。"""

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
    """中心像元光谱 1D 支路 pixel_branch。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv1d(1, 64, 11, padding=0)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 128, 3, padding=0)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.pool(x)
        return x.flatten(1)


class TwoBranchHSIStage(nn.Module):
    """hsi_branch：仅 HSI + 中心光谱，输出 logits（与 Keras 一致）。"""

    def __init__(self, patch_size: int, hsi_channels: int, num_classes: int) -> None:
        super().__init__()
        if hsi_channels < 13:
            raise ValueError('pixel_branch 需要 hsi_channels >= 13')
        self._patch_size = patch_size
        self.simple = _SimpleCnnBranch(hsi_channels)
        self.pixel = _PixelBranch()
        sd, pd, _, _ = fusion_feature_dims(patch_size, hsi_channels)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(sd + pd, num_classes)

    def forward(self, data_dict: dict) -> torch.Tensor:
        hsi = data_dict['hsi']
        _, _, h, w = hsi.shape
        cy, cx = h // 2, w // 2
        pixel_in = hsi[:, :, cy, cx].unsqueeze(1)
        h_s = self.simple(hsi)
        h_p = self.pixel(pixel_in)
        x = torch.cat([h_s, h_p], dim=1)
        x = self.drop(x)
        return self.fc(x)


class TwoBranchLiDARStage(nn.Module):
    """lidar_branch：仅 LiDAR cascade，输出 logits。"""

    def __init__(self, patch_size: int, lidar_channels: int, num_classes: int) -> None:
        super().__init__()
        if patch_size < 3:
            raise ValueError('patch_size 过小')
        self.lidar_net = _CascadeNet(lidar_channels)
        _, _, ld, _ = fusion_feature_dims(patch_size, 13)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(ld, num_classes)

    def forward(self, data_dict: dict) -> torch.Tensor:
        x = self.lidar_net(data_dict['lidar'])
        x = self.drop(x)
        return self.fc(x)


class TwoBranchFinetuneModel(nn.Module):
    """finetune_Net：两路特征拼接后 BN + Dropout + Dense(128) + logits（支路可冻结）。"""

    def __init__(
        self,
        patch_size: int,
        hsi_channels: int,
        lidar_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hsi_channels < 13:
            raise ValueError('Two-branch CNN 的 pixel_branch 需要 hsi_channels >= 13')
        self.simple = _SimpleCnnBranch(hsi_channels)
        self.pixel = _PixelBranch()
        self.lidar_net = _CascadeNet(lidar_channels)
        _, _, _, fusion_dim = fusion_feature_dims(patch_size, hsi_channels)
        self.bn_fusion = nn.BatchNorm1d(fusion_dim)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def load_from_hs_lidar_stages(self, hsi_sd: dict, lidar_sd: dict, strict: bool = True) -> None:
        """从阶段 1/2 的 state_dict 中抽取 simple、pixel、lidar_net。"""
        def strip_prefix(d: dict, prefix: str) -> dict:
            out = {}
            for k, v in d.items():
                if k.startswith(prefix):
                    out[k[len(prefix) :]] = v
            return out

        self.simple.load_state_dict(strip_prefix(hsi_sd, 'simple.'), strict=strict)
        self.pixel.load_state_dict(strip_prefix(hsi_sd, 'pixel.'), strict=strict)
        self.lidar_net.load_state_dict(strip_prefix(lidar_sd, 'lidar_net.'), strict=strict)

    def freeze_encoders(self) -> None:
        """与 Keras finetune_Net trainable=False 一致。"""
        for m in (self.simple, self.pixel, self.lidar_net):
            for p in m.parameters():
                p.requires_grad = False

    def trainable_fusion_parameters(self):
        return list(self.bn_fusion.parameters()) + list(self.drop.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())

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


# 兼容旧名
TwoBranchCNNBackbone = TwoBranchFinetuneModel
