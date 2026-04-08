"""
Xu et al. 2017 双分支 CNN（多源遥感分类）— PyTorch 复现，与 BUCT 官方 Keras 对齐。

官方仓库（论文结构来源）：
  https://github.com/BUCT-Vision/Two-branch-CNN-Multisource-RS-classification
  - models.py: cascade_block / cascade_Net / simple_cnn_branch(small_mode=False) /
               pixel_branch / hsi_branch / lidar_branch / finetune_Net

对齐要点（相对旧版本仓库实现）：
  - hsi_branch 使用 simple_cnn_branch(..., small_mode=False)：Conv 256→512 + MaxPool，无额外 Dense。
  - pixel_branch：Conv1D 64@11 valid → Conv1D 128@3 valid → MaxPool1D 2 valid；通道与官方 filters[3]/[5] 一致。
  - cascade_Net：conv0(3×3)→LeakyReLU→cascade(64)→MaxPool→LeakyReLU→cascade(128)；conv0 无 BN（与官方注释掉的一致）。
  - cascade_block：与 Keras 顺序一致（conv1_1 后 BN+ReLU；conv1_2 后 BN+LeakyReLU；shortcut 自 input；等）。
  - 分类头前 Dropout(0.5) / finetune 融合处 BN+Dropout(0.25)+Dense(128)+LeakyReLU。
  - 特征维度用 dummy 前向推断，避免手工 spatial 公式与 TF/Keras SAME/valid 舍入不一致。

训练协议（与官方 main.py 一致）仍由 pipeline/two_branch_protocol.py 负责：
  1) 仅训 HSI 支路  2) 仅训 LiDAR  3) 加载两路、去头、冻结支路、SGD 训融合头。

端到端单阶段（非论文协议）：环境变量 MMDIFF_TWO_BRANCH_END2END=1。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 与 models.py cascade_block / cascade_Net 一致
# ---------------------------------------------------------------------------


class _CascadeBlock(nn.Module):
    """Keras models.cascade_block 逐层对应。"""

    def __init__(self, in_channels: int, nb_filter: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1_1 = nn.Conv2d(in_channels, nb_filter * 2, kernel_size, padding=pad)
        self.bn1_1 = nn.BatchNorm2d(nb_filter * 2)
        self.conv1_2 = nn.Conv2d(nb_filter * 2, nb_filter, 1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(nb_filter)
        self.skip_conv = nn.Conv2d(in_channels, nb_filter * 2, 1, bias=False)
        self.conv2_1 = nn.Conv2d(nb_filter, nb_filter * 2, kernel_size, padding=pad)
        self.bn2_1 = nn.BatchNorm2d(nb_filter * 2)
        self.conv2_2 = nn.Conv2d(nb_filter * 2, nb_filter, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(nb_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv1_1 -> BN -> ReLU（官方 Activation('relu')，非 Leaky）
        c1 = F.relu(self.bn1_1(self.conv1_1(x)), inplace=True)
        c2 = F.leaky_relu(self.bn1_2(self.conv1_2(c1)), 0.2, inplace=True)
        skip = self.skip_conv(x)
        c3 = F.leaky_relu(self.bn2_1(self.conv2_1(c2) + skip), 0.2, inplace=True)
        c4 = self.bn2_2(self.conv2_2(c3))
        return F.leaky_relu(c2 + c4, 0.2, inplace=True)


class _CascadeNet(nn.Module):
    """Keras models.cascade_Net（LiDAR 支路主干）。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        f = [16, 32, 64, 96, 128, 192, 256, 512]
        # filters[2]=64；官方未对 conv0 做 BN
        self.conv0 = nn.Conv2d(in_channels, f[2], 3, padding=1)
        self.cb1 = _CascadeBlock(f[2], f[2])
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # filters[4]=128
        self.cb2 = _CascadeBlock(f[2], f[4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x = self.cb1(x)
        x = self.pool(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.cb2(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# simple_cnn_branch(..., small_mode=False) — 官方 hsi_branch 所用
# ---------------------------------------------------------------------------


class _SimpleCnnBranch(nn.Module):
    """Keras simple_cnn_branch(input_tensor, small_mode=False)。"""

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


# ---------------------------------------------------------------------------
# pixel_branch — 官方：Conv1D(64,11) -> Conv1D(128,3) -> MaxPool1D(2)
# ---------------------------------------------------------------------------


class _PixelBranch(nn.Module):
    """Keras pixel_branch；输入 (N, 1, hchn)，与 main.py 中中心像元光谱一致。"""

    def __init__(self) -> None:
        super().__init__()
        # filters[3]=64, filters[5]=128
        self.conv0 = nn.Conv1d(1, 64, 11, padding=0)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 128, 3, padding=0)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.pool(x)
        return x.flatten(1)


def _infer_simple_flat(in_channels: int, patch_size: int) -> int:
    m = _SimpleCnnBranch(in_channels)
    with torch.no_grad():
        y = m(torch.zeros(1, in_channels, patch_size, patch_size))
    return int(y.shape[1])


def _infer_pixel_flat(hsi_channels: int) -> int:
    m = _PixelBranch()
    with torch.no_grad():
        y = m(torch.zeros(1, 1, hsi_channels))
    return int(y.shape[1])


def _infer_lidar_flat(in_channels: int, patch_size: int) -> int:
    m = _CascadeNet(in_channels)
    with torch.no_grad():
        y = m(torch.zeros(1, in_channels, patch_size, patch_size))
    return int(y.shape[1])


# ---------------------------------------------------------------------------
# 三阶段 / Finetune 外壳（与 pipeline 衔接）
# ---------------------------------------------------------------------------


class TwoBranchHSIStage(nn.Module):
    """Keras hsi_branch：双输入在训练里由 data_dict 组装（空间 patch + 中心光谱向量）。"""

    def __init__(self, patch_size: int, hsi_channels: int, num_classes: int) -> None:
        super().__init__()
        if hsi_channels < 13:
            raise ValueError('pixel_branch 需要光谱长度 >= 13（与官方 Conv1D 有效卷积一致）')
        self._patch_size = patch_size
        self.simple = _SimpleCnnBranch(hsi_channels)
        self.pixel = _PixelBranch()
        sd = _infer_simple_flat(hsi_channels, patch_size)
        pd = _infer_pixel_flat(hsi_channels)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(sd + pd, num_classes)

    def forward(
        self,
        data_dict: dict,
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_supcon_proj:
            raise RuntimeError('Two-branch HSI 阶段未启用 SupCon')
        hsi = data_dict['hsi']
        _, _, h, w = hsi.shape
        cy, cx = h // 2, w // 2
        # 官方：Xh[:, r, r, :, np.newaxis]，奇数 patch 时 (r,r) 与 (h//2,w//2) 一致
        pixel_in = hsi[:, :, cy, cx].unsqueeze(1)
        h_s = self.simple(hsi)
        h_p = self.pixel(pixel_in)
        x = torch.cat([h_s, h_p], dim=1)
        x = self.drop(x)
        logits = self.fc(x)
        if return_center_logits:
            return logits, logits
        return logits


class TwoBranchLiDARStage(nn.Module):
    """Keras lidar_branch。"""

    def __init__(self, patch_size: int, lidar_channels: int, num_classes: int) -> None:
        super().__init__()
        if patch_size < 3:
            raise ValueError('patch_size 过小')
        self.lidar_net = _CascadeNet(lidar_channels)
        ld = _infer_lidar_flat(lidar_channels, patch_size)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(ld, num_classes)

    def forward(
        self,
        data_dict: dict,
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_supcon_proj:
            raise RuntimeError('Two-branch LiDAR 阶段未启用 SupCon')
        x = self.lidar_net(data_dict['lidar'])
        x = self.drop(x)
        logits = self.fc(x)
        if return_center_logits:
            return logits, logits
        return logits


class TwoBranchFinetuneModel(nn.Module):
    """Keras finetune_Net（融合头前拼接 hsi 特征与 lidar 特征）。"""

    def __init__(
        self,
        patch_size: int,
        hsi_channels: int,
        lidar_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hsi_channels < 13:
            raise ValueError('Two-branch CNN 的 pixel_branch 需要光谱长度 >= 13')
        self.simple = _SimpleCnnBranch(hsi_channels)
        self.pixel = _PixelBranch()
        self.lidar_net = _CascadeNet(lidar_channels)
        sd = _infer_simple_flat(hsi_channels, patch_size)
        pd = _infer_pixel_flat(hsi_channels)
        ld = _infer_lidar_flat(lidar_channels, patch_size)
        fusion_dim = sd + pd + ld
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
        for m in (self.simple, self.pixel, self.lidar_net):
            for p in m.parameters():
                p.requires_grad = False

    def trainable_fusion_parameters(self):
        return (
            list(self.bn_fusion.parameters())
            + list(self.drop.parameters())
            + list(self.fc1.parameters())
            + list(self.fc2.parameters())
        )

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


def fusion_feature_dims(
    patch_size: int, hsi_channels: int, lidar_channels: int = 1
) -> tuple[int, int, int, int]:
    """(simple_flat, pixel_flat, lidar_flat, fusion_total)，与当前网络 dummy 前向一致。"""
    sd = _infer_simple_flat(hsi_channels, patch_size)
    pd = _infer_pixel_flat(hsi_channels)
    ld = _infer_lidar_flat(lidar_channels, patch_size)
    return sd, pd, ld, sd + pd + ld
