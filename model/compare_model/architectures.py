"""对比实验模型：与当前 Houston patch 数据管线一致（HSI+LiDAR）。"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CompareClassifierBase
from .exvit_core import MViTBackbone
from .fusatnet_core import FusAtNetBackbone
from .two_branch_cnn_core import TwoBranchCNNBackbone
from .dfinet_core import DFINetBackbone
from .macn_core import MACNBackbone
from .ss_mae.mae import VisionTransfromers


# ---------------------------------------------------------------------------
# FusAtNet（与 ../FusAtNet/model2.py 一致）
# ---------------------------------------------------------------------------


class FusAtNetClassifier(CompareClassifierBase):
    """FusAtNet：光谱/空间注意力 + 模态注意力；优化器与原版一致为 NAdam（lr/wd 仍读 param.train.optimizer）。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        self.net = FusAtNetBackbone(
            num_hsi_bands=self.hsi_channels,
            num_lidar_bands=self.lidar_channels,
            num_classes=self.num_classes,
        )
        self.projections = nn.ModuleDict({'hs': self.net.hs, 'mask_spat': self.net.mask_spat})
        self._init_optimizer_and_scheduler()

    def _build_optimizer(self, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        optim_cfg = train_cfg.get('optimizer', {})
        lr = float(optim_cfg.get('lr') or 2e-5)
        weight_decay = float(optim_cfg.get('weight_decay') or 0.01)
        betas = tuple(optim_cfg.get('betas') or (0.9, 0.999))
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError('FusAtNetClassifier 无可训练参数')
        return torch.optim.NAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        logits = self.net(hsi, lidar)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


# ---------------------------------------------------------------------------
# ExViT / MViT（TGRS 2023 — jingyao16/ExViT）
# ---------------------------------------------------------------------------


class ExViTClassifier(CompareClassifierBase):
    """ExViT（MViT）：双支深度可分离 CNN + 模态内 ViT + 融合 ViT + token 加权池化。

    与原版 MViT_demo 默认超参一致：dim=64, depth=6, heads=4, mlp_dim=32, dropout=0.1。
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        patch_size = int(self.opt.get('dataset', {}).get('patch_size') or 11)
        self.net = MViTBackbone(
            patch_size=patch_size,
            num_patches=(self.hsi_channels, self.lidar_channels),
            num_classes=self.num_classes,
            dim=64,
            depth=6,
            heads=4,
            mlp_dim=32,
            dropout=0.1,
            emb_dropout=0.1,
            dim_head=16,
            mode='MViT',
        )
        self.projections = nn.ModuleDict({
            'separable1': self.net.separable1,
            'separable2': self.net.separable2,
        })
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        logits = self.net(hsi, lidar)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


# ---------------------------------------------------------------------------
# SS-MAE 微调网络（Gao et al. / TGRS 2023 — summitgao/SS-MAE）
# https://github.com/summitgao/SS-MAE
# ---------------------------------------------------------------------------


class SSMAEClassifier(CompareClassifierBase):
    """与官方 `net/VIT/mae.py` 中 `VisionTransfromers` 结构一致（类名保留原文拼写 Transfromers）。

    官方预训练/数据管线见仓库 `get_dat.py` 与 `data/dataset`；本仓库用 `param` patch 尺寸，
    将 HSI/LiDAR **双线性** 对齐到 ``crop_size``（默认 7，与 README Berlin 示例一致）。
    ``hsi_pca`` 默认取光谱 **前 pca_num 维**；若需与官方一致的全数据 PCA，请提供
    ``MMDIFF_SSMAE_PCA_MAT``（``.npy``，形状 ``(spectral_dim, pca_num)``，右乘 ``hsi`` 展平谱向量）。

    环境变量（可选）：``MMDIFF_SSMAE_CROP_SIZE``、``MMDIFF_SSMAE_PCA_NUM``、``MMDIFF_SSMAE_DEPTH``、
    ``MMDIFF_SSMAE_HEAD``、``MMDIFF_SSMAE_DIM``、``MMDIFF_SSMAE_PATCH_SIZE``、
    ``MMDIFF_SSMAE_CHANNEL_NUM``（拼接通道数，默认 HSI+LiDAR）、``MMDIFF_SSMAE_WEIGHT_DECAY``（默认 0.05）、
    ``MMDIFF_SSMAE_DATASET``（``hsi_e``/``lidar_e`` 分支用，默认 Houston2018）。
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        self._ss = self._build_ss_args()
        ch_raw = self.hsi_channels + self.lidar_channels
        ch = int(os.environ.get('MMDIFF_SSMAE_CHANNEL_NUM') or ch_raw)
        self.net = VisionTransfromers(
            channel_number=ch,
            img_size=self._ss.crop_size,
            patch_size=self._ss.patch_size,
            embed_dim=self._ss.dim,
            depth=self._ss.depth,
            num_heads=self._ss.head,
            num_classes=self.num_classes,
            args=self._ss,
        )
        self.projections = nn.ModuleDict({'vit': self.net.model})
        if int(os.environ.get('MMDIFF_SSMAE_LOAD_PRETRAIN', '0')):
            self.net._load_mae_pretrain(self._ss)
        self._init_optimizer_and_scheduler()

    def _build_ss_args(self) -> SimpleNamespace:
        ds = self.opt.get('dataset', {})
        crop = int(os.environ.get('MMDIFF_SSMAE_CROP_SIZE') or ds.get('ss_mae_crop_size') or 7)
        pca = int(os.environ.get('MMDIFF_SSMAE_PCA_NUM') or ds.get('ss_mae_pca_num') or 30)
        depth = int(os.environ.get('MMDIFF_SSMAE_DEPTH') or 2)
        head = int(os.environ.get('MMDIFF_SSMAE_HEAD') or 8)
        dim = int(os.environ.get('MMDIFF_SSMAE_DIM') or 256)
        patch = int(os.environ.get('MMDIFF_SSMAE_PATCH_SIZE') or 1)
        ds_name = os.environ.get('MMDIFF_SSMAE_DATASET', 'Houston2018')
        device = os.environ.get('MMDIFF_SSMAE_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        return SimpleNamespace(
            dataset=ds_name,
            pca_num=pca,
            crop_size=crop,
            patch_size=patch,
            depth=depth,
            dim=dim,
            head=head,
            pretrain_num=int(os.environ.get('MMDIFF_SSMAE_PRETRAIN_NUM', '50000')),
            mask_ratio=float(os.environ.get('MMDIFF_SSMAE_MASK_RATIO', '0.3')),
            device=device,
            is_load_pretrain=int(os.environ.get('MMDIFF_SSMAE_LOAD_PRETRAIN', '0')),
            is_pretrain=0,
            is_train=1,
        )

    def _build_optimizer(self, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        optim_cfg = train_cfg.get('optimizer', {})
        lr = float(optim_cfg.get('lr') or 1e-4)
        wd = float(os.environ.get('MMDIFF_SSMAE_WEIGHT_DECAY', optim_cfg.get('weight_decay', 0.05)))
        betas = tuple(optim_cfg.get('betas') or (0.9, 0.999))
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError('SSMAEClassifier 无可训练参数')
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

    def _resize_to_crop(self, x: torch.Tensor) -> torch.Tensor:
        s = self._ss.crop_size
        if x.shape[-1] == s and x.shape[-2] == s:
            return x
        return F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)

    def _hsi_pca_tensor(self, hsi: torch.Tensor) -> torch.Tensor:
        """返回 (B,1,pca_num,H,W)，供 `hsi_e` 使用。"""
        pca = self._ss.pca_num
        if hsi.shape[1] >= pca:
            z = hsi[:, :pca].contiguous()
        else:
            z = torch.zeros(
                hsi.shape[0],
                pca,
                hsi.shape[2],
                hsi.shape[3],
                device=hsi.device,
                dtype=hsi.dtype,
            )
            z[:, : hsi.shape[1]] = hsi
        mat = os.environ.get('MMDIFF_SSMAE_PCA_MAT', '').strip()
        if mat and os.path.isfile(mat):
            w_np = np.load(mat)
            w_t = torch.from_numpy(w_np).to(device=hsi.device, dtype=hsi.dtype)
            c = hsi.shape[1]
            if w_t.shape == (c, pca):
                wt = w_t
            elif w_t.shape == (pca, c):
                wt = w_t.T
            else:
                raise ValueError(
                    f'MMDIFF_SSMAE_PCA_MAT 形状应为 ({c},{pca}) 或 ({pca},{c})，当前 {tuple(w_t.shape)}'
                )
            hc = hsi.permute(0, 2, 3, 1).reshape(-1, c)
            z = (hc @ wt).reshape(hsi.shape[0], hsi.shape[2], hsi.shape[3], pca).permute(0, 3, 1, 2)
        return z.unsqueeze(1)

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        hsi, lidar = self._hsi_lidar(data_dict)
        hsi = self._resize_to_crop(hsi)
        lidar = self._resize_to_crop(lidar)
        ch = int(os.environ.get('MMDIFF_SSMAE_CHANNEL_NUM') or (self.hsi_channels + self.lidar_channels))
        if hsi.shape[1] + lidar.shape[1] != ch:
            raise ValueError(
                f'SS-MAE：HSI+LiDAR 通道数 {hsi.shape[1]}+{lidar.shape[1]} 与期望 {ch} 不一致，'
                f'请调整数据或设置 MMDIFF_SSMAE_CHANNEL_NUM'
            )
        hsi_pca = self._hsi_pca_tensor(hsi)
        logits, _ = self.net(hsi, lidar, hsi_pca)
        if return_center_logits:
            return logits, logits
        return logits


# ---------------------------------------------------------------------------
# Two-branch CNN（Xu et al. TGRS 2017 — BUCT Keras 仓库的 PyTorch 版）
# https://github.com/BUCT-Vision/Two-branch-CNN-Multisource-RS-classification
# ---------------------------------------------------------------------------


class TwoBranchCNNClassifier(CompareClassifierBase):
    """双分支 CNN（BUCT Xu 2017 结构）。

    对比实验默认走三阶段训练（HSI → LiDAR → 加载权重并冻结支路后 Finetune），见 pipeline/two_branch_protocol.py。
    仅当 MMDIFF_TWO_BRANCH_END2END=1 时由 runner 走单阶段端到端（调试/非论文协议）。
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        patch_size = int(self.opt.get('dataset', {}).get('patch_size') or 11)
        self.net = TwoBranchCNNBackbone(
            patch_size=patch_size,
            hsi_channels=self.hsi_channels,
            lidar_channels=self.lidar_channels,
            num_classes=self.num_classes,
        )
        self.projections = nn.ModuleDict()
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        logits = self.net(hsi, lidar)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


# ---------------------------------------------------------------------------
# DFINet（Gao et al. TGRS 2022 — formango/HSI_MSI_Multisource_Classification）
# https://github.com/formango/HSI_MSI_Multisource_Classification
# ---------------------------------------------------------------------------


class DFINetClassifier(CompareClassifierBase):
    """DFINet：HSINet + MSINet（LiDAR）+ CAM + depthwise 互相关 + 全连接分类。

    对比实验默认由 `pipeline/dfinet_protocol.py` 按论文联合损失 + SGD 训练；
    `return_aux=True` 时返回 (feat_hsi, feat_msi, h_flat, l_flat, logits) 供 calc_loss。
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        patch_size = int(self.opt.get('dataset', {}).get('patch_size') or 11)
        self.net = DFINetBackbone(
            patch_size=patch_size,
            channel_hsi=self.hsi_channels,
            channel_msi=self.lidar_channels,
            class_num=self.num_classes,
        )
        self.projections = nn.ModuleDict({'hsi_stem': self.net.featnet1.conv1, 'msi_stem': self.net.featnet2.conv1})
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
        return_aux: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        if return_aux:
            return self.net(hsi, lidar, return_aux=True)
        logits = self.net(hsi, lidar)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


# ---------------------------------------------------------------------------
# MACN（Li et al. TGRS 2023 — like413/MACN）
# https://github.com/like413/MACN
# ---------------------------------------------------------------------------


class MACNClassifier(CompareClassifierBase):
    """MACN：Conv3D+Conv2D 双支特征 + MACT + token + MCGF 交叉融合。

    与 like413/MACN trentoTrain.py 一致：FocalLoss（α=0.25, γ=2）+ Adam（lr 读 param，默认 1e-3）；
    不使用 piecewise 学习率调度（原文训练循环无 Lambda 调度）。
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        self.net = MACNBackbone(
            hsi_channels=self.hsi_channels,
            lidar_channels=self.lidar_channels,
            num_classes=self.num_classes,
        )
        self.projections = nn.ModuleDict({'mact': self.net.mact})
        from .macn_focal_loss import MACNFocalLoss

        self.loss_func = MACNFocalLoss(self.num_classes, alpha=0.25, gamma=2.0)
        self.optimizer = None
        self.exp_lr_scheduler = None
        self._init_optimizer_and_scheduler()

    def _build_optimizer(self, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        optim_cfg = train_cfg.get('optimizer', {})
        lr = float(optim_cfg.get('lr') or 1e-3)
        betas = tuple(optim_cfg.get('betas') or (0.9, 0.999))
        weight_decay = float(optim_cfg.get('weight_decay') or 0.0)
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError('MACNClassifier 无可训练参数')
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg: Dict[str, Any]):
        self._scheduler_lr_total_steps = 0
        return None

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        logits = self.net(hsi, lidar)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


# ---------------------------------------------------------------------------
# FGCNN（Fuzzy Graph CNN / Fuzzy Graph Convolutional Network, EAAI）
# 原仓库目录名为 F-GCN: https://github.com/liziyilzy/F-GCN
# ---------------------------------------------------------------------------


class _FuzzySmoothing(nn.Module):
    """模糊空间平滑（对应原始 FuzzyImage）。

    在规则网格上对每个节点取 k 近邻，以高斯核加权平均邻居特征。
    原始参数: k=5, std=4。
    """

    def __init__(self, patch_size: int, k: int = 5, std: float = 4.0):
        super().__init__()
        N = patch_size * patch_size
        coords = np.stack(
            np.mgrid[:patch_size, :patch_size], axis=-1
        ).reshape(N, 2).astype(np.float32)

        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = (diff * diff).sum(axis=-1)

        indices = np.argsort(dist_sq, axis=1)[:, :k]

        weights = np.zeros((N, k), dtype=np.float32)
        for i in range(N):
            s = coords[indices[i]] - coords[i]
            arg = -(s * s).sum(axis=1) / (2.0 * std * std)
            h = np.exp(arg)
            h_sum = h.sum()
            if h_sum > 0:
                h /= h_sum
            weights[i] = h

        self.register_buffer('_indices', torch.from_numpy(indices).long())
        self.register_buffer('_weights', torch.from_numpy(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        idx = self._indices
        nb = x[:, idx.reshape(-1), :].reshape(B, N, idx.shape[1], C)
        return (nb * self._weights.unsqueeze(0).unsqueeze(-1)).sum(dim=2)


class _GraphConvolution(nn.Module):
    """图卷积层（对应原始 GraphConvolution）: act(support @ (X @ W) + b)。

    权重用 Uniform[-0.05, 0.05] 初始化，与原始 TF 实现一致。
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.uniform_(self.weight, -0.05, 0.05)
        self.use_bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.empty(output_dim))
            nn.init.uniform_(self.bias_param, -0.05, 0.05)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        pre_sup = torch.matmul(x, self.weight)
        output = torch.matmul(support, pre_sup)
        if self.use_bias:
            output = output + self.bias_param
        return output


class FGCNNClassifier(CompareClassifierBase):
    """FGCNN: Fuzzy Graph CNN for Hyperspectral Image Classification（论文常用名 FGCNN）。

    核心结构（严格对应原始 GCN.py / GCNLayer.py / fuzzy_learn.py）：
      1. FuzzySmoothing — 基于 k 近邻高斯核的空间模糊平滑 (k=5, std=4)
      2. 每尺度双层 GCN:
         Layer-1: GraphConv(input_dim → hidden=100, softplus, bias)
         Layer-2: GraphConv(hidden → num_classes, identity, bias)
      3. 多尺度输出逐元素求和
      4. 取中心像素的 logits 作为分类输出

    Support 矩阵归一化: D^{-1/2} (A + I) D^{-1/2}  (对应原始 CalSupport)
    """

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)

        patch_size = int(self.opt.get('dataset', {}).get('patch_size') or 11)
        hidden_dim = 100
        fuzzy_k = 5
        fuzzy_std = 4.0
        scale_ks = [4, 8, 12]

        input_dim = self.hsi_channels + self.lidar_channels
        N = patch_size * patch_size
        self._patch_size = patch_size
        self._center_idx = N // 2
        self.num_scales = len(scale_ks)

        self.fuzzy = _FuzzySmoothing(patch_size, k=fuzzy_k, std=fuzzy_std)

        for i, k_nn in enumerate(scale_ks):
            support = self._build_support(patch_size, k_nn)
            self.register_buffer(f'support_{i}', support)

        self.gcn_layers = nn.ModuleList()
        for _ in range(self.num_scales):
            self.gcn_layers.append(nn.ModuleList([
                _GraphConvolution(input_dim, hidden_dim, bias=True),
                _GraphConvolution(hidden_dim, self.num_classes, bias=True),
            ]))

        self.projections = nn.ModuleDict()
        self._init_optimizer_and_scheduler()

    @staticmethod
    def _build_support(patch_size: int, k: int) -> torch.Tensor:
        """构建 k-NN 空间邻接矩阵并做 GCN 归一化 D^{-1/2}(A+I)D^{-1/2}。"""
        N = patch_size * patch_size
        coords = np.stack(
            np.mgrid[:patch_size, :patch_size], axis=-1
        ).reshape(N, 2).astype(np.float32)
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = (diff * diff).sum(axis=-1)

        A = np.zeros((N, N), dtype=np.float32)
        indices = np.argsort(dist_sq, axis=1)
        for i in range(N):
            neighbors = indices[i, 1:k + 1]
            A[i, neighbors] = 1.0
        A = np.maximum(A, A.T)

        A_hat = A + np.eye(N, dtype=np.float32)
        D = A_hat.sum(axis=1)
        D_inv_sqrt = np.diag(D ** (-0.5))
        support = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return torch.from_numpy(support.astype(np.float32))

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        B = hsi.shape[0]

        x = torch.cat([hsi, lidar], dim=1)
        x = x.flatten(2).transpose(1, 2)

        x_fuzzy = self.fuzzy(x)

        output = torch.zeros(
            B, x.shape[1], self.num_classes, device=x.device, dtype=x.dtype,
        )
        for i in range(self.num_scales):
            support = getattr(self, f'support_{i}')
            h = F.softplus(self.gcn_layers[i][0](x_fuzzy, support))
            h = self.gcn_layers[i][1](h, support)
            output = output + h

        logits = output[:, self._center_idx, :]

        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits
