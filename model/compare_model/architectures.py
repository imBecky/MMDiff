"""官方方法对应的 PyTorch 对照实现：与当前 Houston patch 数据管线一致（HSI+LiDAR）。"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CompareClassifierBase


def _init_conv_linear(module: nn.Module, scale: float = 0.1) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class CoupledCNNClassifier(CompareClassifierBase):
    """Coupled-CNN 风格：HSI / LiDAR 两分支卷积 + 耦合全连接。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        self.hsi_net = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.lidar_net = nn.Sequential(
            nn.Conv2d(self.lidar_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )
        self.projections = nn.ModuleDict({'hsi': self.hsi_net, 'lidar': self.lidar_net})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        fh = self.hsi_net(hsi).flatten(1)
        fl = self.lidar_net(lidar).flatten(1)
        logits = self.fuse(torch.cat([fh, fl], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class FusAtNetClassifier(CompareClassifierBase):
    """FusAtNet 风格：光谱-空间双路注意力 + LiDAR 融合。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        mid = 64
        self.stem = nn.Sequential(
            nn.Conv2d(c, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, mid // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid // 8, mid, 1),
            nn.Sigmoid(),
        )
        self.sa = nn.Conv2d(mid, 1, kernel_size=1)
        self.lidar_enc = nn.Sequential(
            nn.Conv2d(self.lidar_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(mid + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes),
        )
        self.projections = nn.ModuleDict({'stem': self.stem, 'lidar_enc': self.lidar_enc})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        x = self.stem(hsi)
        w = self.ca(x)
        x = x * w
        a = torch.sigmoid(self.sa(x))
        x = (x * a).mean(dim=(2, 3))
        lf = self.lidar_enc(lidar).flatten(1)
        logits = self.head(torch.cat([x, lf], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class MACNClassifier(CompareClassifierBase):
    """MACN 风格：多尺度并行卷积 + 聚合。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        mid = 48
        self.ms = nn.ModuleList(
            [
                nn.Conv2d(c, mid, kernel_size=k, padding=k // 2, bias=False)
                for k in (1, 3, 5)
            ]
        )
        self.bn = nn.BatchNorm2d(mid * 3)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lidar_fc = nn.Sequential(
            nn.Conv2d(self.lidar_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(mid * 3 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )
        self.projections = nn.ModuleDict({'ms': nn.Sequential(*self.ms)})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        outs = [m(hsi) for m in self.ms]
        x = torch.cat(outs, dim=1)
        x = self.act(self.bn(x))
        x = self.pool(x).flatten(1)
        lf = self.lidar_fc(lidar).flatten(1)
        logits = self.head(torch.cat([x, lf], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class _HCTMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HCTTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _HCTMLPBlock(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _HCTTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [_HCTTransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _HCTCrossTokenBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, cls_tok: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        ctx = torch.cat([cls_tok, patch_tokens], dim=1)
        q = self.norm_q(cls_tok)
        k = self.norm_ctx(ctx)
        out, _ = self.attn(q, k, k, need_weights=False)
        return cls_tok + out


class _HCTFusionEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        ct_attn_depth: int,
        ct_attn_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        'h_enc': _HCTTransformer(dim, 1, heads, mlp_dim, dropout),
                        'l_enc': _HCTTransformer(dim, 1, heads, mlp_dim, dropout),
                        'h_to_l': nn.ModuleList(
                            [_HCTCrossTokenBlock(dim, ct_attn_heads, dropout) for _ in range(ct_attn_depth)]
                        ),
                        'l_to_h': nn.ModuleList(
                            [_HCTCrossTokenBlock(dim, ct_attn_heads, dropout) for _ in range(ct_attn_depth)]
                        ),
                    }
                )
            )

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            h_tokens = layer['h_enc'](h_tokens)
            l_tokens = layer['l_enc'](l_tokens)
            h_cls, h_patch = h_tokens[:, :1], h_tokens[:, 1:]
            l_cls, l_patch = l_tokens[:, :1], l_tokens[:, 1:]
            for blk_h, blk_l in zip(layer['h_to_l'], layer['l_to_h']):
                h_cls = blk_h(h_cls, l_patch)
                l_cls = blk_l(l_cls, h_patch)
            h_tokens = torch.cat([h_cls, h_patch], dim=1)
            l_tokens = torch.cat([l_cls, l_patch], dim=1)
        return h_tokens, l_tokens


class HCTClassifier(CompareClassifierBase):
    """按官方 HCTnet 思路实现：HSI 3D+2D 编码、tokenization、双分支 CTA 融合。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        dim = 64
        num_tokens = 4
        heads = 8
        mlp_dim = 8
        depth = 1
        dropout = 0.1
        emb_dropout = 0.1
        ct_attn_depth = 1
        ct_attn_heads = 8

        hsi_2d_in = 8 * max(1, self.hsi_channels - 2)
        self.hsi_conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.hsi_conv2d = nn.Sequential(
            nn.Conv2d(hsi_2d_in, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.lidar_conv2d = nn.Sequential(
            nn.Conv2d(self.lidar_channels, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.num_tokens = num_tokens
        self.dim = dim
        self.token_wA = nn.Parameter(torch.empty(1, num_tokens, dim))
        self.token_wV = nn.Parameter(torch.empty(1, dim, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.fusion_encoder = _HCTFusionEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            ct_attn_depth=ct_attn_depth,
            ct_attn_heads=ct_attn_heads,
            dropout=dropout,
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_classes),
        )

        self.projections = nn.ModuleDict(
            {
                'hsi_conv3d': self.hsi_conv3d,
                'hsi_conv2d': self.hsi_conv2d,
                'lidar_conv2d': self.lidar_conv2d,
            }
        )
        _init_conv_linear(self)
        nn.init.xavier_normal_(self.token_wA)
        nn.init.xavier_normal_(self.token_wV)
        nn.init.normal_(self.pos_embedding, std=0.02)
        self._init_optimizer_and_scheduler()

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        wa = self.token_wA.transpose(1, 2)
        attn = torch.matmul(x, wa).transpose(1, 2).softmax(dim=-1)
        values = torch.matmul(x, self.token_wV)
        return torch.matmul(attn, values)

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)

        x1 = self.hsi_conv3d(hsi.unsqueeze(1))
        b, c3, d3, h3, w3 = x1.shape
        x1 = x1.reshape(b, c3 * d3, h3, w3)
        x1 = self.hsi_conv2d(x1).flatten(2).transpose(1, 2)

        x2 = self.lidar_conv2d(lidar).flatten(2).transpose(1, 2)

        t1 = self._tokenize(x1)
        t2 = self._tokenize(x2)

        cls1 = self.cls_token.expand(b, -1, -1)
        cls2 = self.cls_token.expand(b, -1, -1)
        x1 = self.dropout(torch.cat([cls1, t1], dim=1) + self.pos_embedding)
        x2 = self.dropout(torch.cat([cls2, t2], dim=1) + self.pos_embedding)

        x1, x2 = self.fusion_encoder(x1, x2)
        logits = self.mlp_head(x1[:, 0]) + self.mlp_head(x2[:, 0])
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class ExViTClassifier(CompareClassifierBase):
    """ExViT 风格：patch embedding + 轻量 Transformer encoder。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        patch = 11
        dim = 128
        self.patch_embed = nn.Conv2d(c, dim, kernel_size=patch, stride=patch)
        # 11×11 patch、kernel=11 → 单空间 token；序列长度 = 1(cls) + 1(patch) = 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 2, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=256, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.lidar_mlp = nn.Sequential(nn.Linear(11 * 11 * self.lidar_channels, 64), nn.ReLU(inplace=True))
        self.head = nn.Linear(dim + 64, self.num_classes)
        self.projections = nn.ModuleDict({'patch_embed': self.patch_embed})
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        b = hsi.size(0)
        x = self.patch_embed(hsi).flatten(2).transpose(1, 2)
        cls = self.cls.expand(b, -1, -1)
        tok = torch.cat([cls, x], dim=1)
        tok = tok + self.pos[:, : tok.size(1), :]
        z = self.encoder(tok)[:, 0]
        lf = self.lidar_mlp(lidar.flatten(1))
        logits = self.head(torch.cat([z, lf], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class SSMAEClassifier(CompareClassifierBase):
    """SS-MAE 风格：空间池化后沿光谱维自注意力 + 分类头。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        self.spatial = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.spec_tokens = nn.Linear(64, 128)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.spec_enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.lidar_fc = nn.Linear(11 * 11 * self.lidar_channels, 64)
        self.head = nn.Linear(128 + 64, self.num_classes)
        self.projections = nn.ModuleDict({'spatial': self.spatial})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        b = hsi.size(0)
        x = self.spatial(hsi).flatten(1)
        t = self.spec_tokens(x).view(b, 1, 128)
        z = self.spec_enc(t).mean(dim=1)
        lf = torch.relu(self.lidar_fc(lidar.flatten(1)))
        logits = self.head(torch.cat([z, lf], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class _MambaBlockFallback(nn.Module):
    """无 mamba_ssm 时用 Conv1d+GLU 近似序列混合。"""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d * 2, kernel_size=5, padding=2, groups=max(1, d // 8)),
            nn.GLU(dim=1),
            nn.Conv1d(d, d, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, L, C
        xt = x.transpose(1, 2)
        return x + self.net(xt).transpose(1, 2)


class MSFMambaClassifier(CompareClassifierBase):
    """MSFMamba：优先 mamba_ssm.Mamba，否则回退为轻量序列块。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        d = 128
        self.stem = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),
        )
        self.proj = nn.Linear(64 * 9, d)
        self._use_mamba = False
        self.seq = _MambaBlockFallback(d)
        try:
            from mamba_ssm import Mamba  # type: ignore

            self.seq = Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
            self._use_mamba = True
        except Exception:
            pass
        self.lidar_fc = nn.Linear(9 * self.lidar_channels, 64)
        self.head = nn.Linear(d + 64, self.num_classes)
        self.projections = nn.ModuleDict({'stem': self.stem})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        b = hsi.size(0)
        x = self.stem(hsi).flatten(1)
        t = self.proj(x).view(b, 1, -1)
        z = self.seq(t).mean(dim=1)
        lf = torch.relu(self.lidar_fc(F.adaptive_avg_pool2d(lidar, 3).flatten(1)))
        logits = self.head(torch.cat([z, lf], dim=1))
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits


class DCMNetClassifier(CompareClassifierBase):
    """DCMNet 风格：双模态特征 + 交叉注意力融合。"""

    def __init__(self, opt, diffusion=None):
        super().__init__(opt, diffusion)
        c = self.hsi_channels
        d = 128
        self.hsi_enc = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d),
        )
        self.lidar_enc = nn.Sequential(
            nn.Conv2d(self.lidar_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, d),
        )
        self.cross = nn.MultiheadAttention(d, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True), nn.Linear(128, self.num_classes))
        self.projections = nn.ModuleDict({'hsi_enc': self.hsi_enc, 'lidar_enc': self.lidar_enc})
        _init_conv_linear(self)
        self._init_optimizer_and_scheduler()

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ):
        hsi, lidar = self._hsi_lidar(data_dict)
        qh = self.hsi_enc(hsi).unsqueeze(1)
        kl = self.lidar_enc(lidar).unsqueeze(1)
        out, _ = self.cross(qh, kl, kl)
        z = self.norm((out + qh).squeeze(1))
        logits = self.head(z)
        if return_center_logits:
            return logits, logits
        if return_supcon_proj:
            raise RuntimeError('对比模型未启用 SupCon')
        return logits
