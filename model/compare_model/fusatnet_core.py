"""
FusAtNet 主干（与 ../FusAtNet/model2.py 对齐：Keras 版见 model.py）。
光谱/空间注意力 + 模态注意力 + 分类头；仅 HSI + LiDAR。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        # BatchNorm 在 ReLU 之后，与原始 Keras 顺序一致
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HSFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 256)
        self.conv2 = ConvBnRelu(256, 256)
        self.conv3 = ConvBnRelu(256, 256)
        self.conv4 = ConvBnRelu(256, 256)
        self.conv5 = ConvBnRelu(256, 256)
        self.conv6 = ConvBnRelu(256, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class SpectralMask(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 256)
        self.conv2 = ConvBnRelu(256, 256)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = ConvBnRelu(512, 256)
        self.conv4 = ConvBnRelu(256, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = ConvBnRelu(512, 256)
        self.conv6 = ConvBnRelu(256, 1024)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        res1 = torch.cat([conv1, conv2], dim=1)
        mp1 = self.pool1(res1)

        conv3 = self.conv3(mp1)
        conv4 = self.conv4(conv3)
        res2 = torch.cat([conv3, conv4], dim=1)
        mp2 = self.pool2(res2)

        conv5 = self.conv5(mp2)
        conv6 = self.conv6(conv5)
        mp3 = self.pool3(conv6)
        return self.gap(mp3)


class SpatialMask(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 128)
        self.conv2 = ConvBnRelu(128, 128)
        self.conv3 = ConvBnRelu(256, 128)
        self.conv4 = ConvBnRelu(128, 256)
        self.conv5 = ConvBnRelu(384, 256)
        self.conv6 = ConvBnRelu(256, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        res1 = torch.cat([conv1, conv2], dim=1)

        conv3 = self.conv3(res1)
        conv4 = self.conv4(conv3)
        res2 = torch.cat([conv3, conv4], dim=1)

        conv5 = self.conv5(res2)
        conv6 = self.conv6(conv5)
        return conv6


class Main2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 256)
        self.conv2 = ConvBnRelu(256, 256)
        self.conv3 = ConvBnRelu(256, 256)
        self.conv4 = ConvBnRelu(256, 256)
        self.conv5 = ConvBnRelu(256, 256)
        self.conv6 = ConvBnRelu(256, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class Att2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 128)
        self.conv2 = ConvBnRelu(128, 128)
        self.conv3 = ConvBnRelu(256, 128)
        self.conv4 = ConvBnRelu(128, 256)
        self.conv5 = ConvBnRelu(384, 256)
        self.conv6 = ConvBnRelu(256, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        res1 = torch.cat([conv1, conv2], dim=1)

        conv3 = self.conv3(res1)
        conv4 = self.conv4(conv3)
        res2 = torch.cat([conv3, conv4], dim=1)

        conv5 = self.conv5(res2)
        conv6 = self.conv6(conv5)
        return conv6


class FusAtNetClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, 256, padding=0)
        self.conv2 = ConvBnRelu(256, 256, padding=0)
        self.conv3 = ConvBnRelu(256, 256, padding=0)
        self.conv4 = ConvBnRelu(256, 256, padding=0)
        self.conv5 = ConvBnRelu(256, 1024, padding=0)
        self.conv6 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return torch.flatten(x, 1)


class FusAtNetBackbone(nn.Module):
    """原 FusAtNet.forward(hsi, lidar) -> logits。"""

    def __init__(self, num_hsi_bands: int, num_lidar_bands: int, num_classes: int):
        super().__init__()
        self.hs = HSFeatureExtractor(num_hsi_bands)
        self.mask_spec = SpectralMask(num_hsi_bands)
        self.mask_spat = SpatialMask(num_lidar_bands)

        concat_channels = num_hsi_bands + num_lidar_bands + 1024 + 1024
        self.main2 = Main2(concat_channels)
        self.att2 = Att2(concat_channels)
        self.classifier = FusAtNetClassifierHead(1024, num_classes)

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        feats_new = self.hs(hsi)

        spec_mask = self.mask_spec(hsi)
        spec = feats_new * spec_mask

        spat = feats_new * self.mask_spat(lidar)

        conc = torch.cat([hsi, lidar, spec, spat], dim=1)
        feats2 = self.main2(conc)
        mask2 = self.att2(conc)
        at_feats = feats2 * mask2

        return self.classifier(at_feats)
