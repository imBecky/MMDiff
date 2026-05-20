from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.train_scheduler import build_lr_scheduler

from model.rgb_student import LightweightRgbEncoder
from model.spatial_fusion_decoder import (
    SpatialFusionDecoder,
    SpatialFusionDecoderLayer,
)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.2,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        nl = int(num_hidden_layers)
        if nl < 1:
            raise ValueError(f"num_hidden_layers 须 >= 1，当前 {num_hidden_layers!r}")
        parts: List[nn.Module] = [
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]
        for _ in range(nl - 1):
            parts.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
        parts.append(nn.Linear(hidden_channels, num_classes))
        self.net = nn.Sequential(*parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _hsi_crop_11x11(x: torch.Tensor) -> torch.Tensor:
    """B,C,H,W -> B,C,11,11，不足则 replicate pad 再取中心 11×11。"""
    ph, pw = 11, 11
    _, c, h, w = x.shape
    need_y = max(0, ph - h)
    need_x = max(0, pw - w)
    if need_y or need_x:
        pad_top = need_y // 2
        pad_bottom = need_y - pad_top
        pad_left = need_x // 2
        pad_right = need_x - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        h, w = x.shape[2], x.shape[3]
    cy, cx = h // 2, w // 2
    y0 = cy - ph // 2
    x0 = cx - pw // 2
    return x[:, :, y0 : y0 + ph, x0 : x0 + pw]


# backward compat for utils/hsi_branch_sanity.py
_crop_center_3x3 = _hsi_crop_11x11


class _HSISpectralResidualBlock(nn.Module):
    """沿光谱轴 1D 卷积残差块：Conv-BN-ReLU-Conv-BN + 恒等映射。"""
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class HSICenterSpectralEncoder(nn.Module):
    """HSI 固定 11×11：每位置 1D 光谱卷积 + SE；可输出空间图 B×C×11×11 供融合。"""

    _AGG_MODES = frozenset({"mean", "attn_pool", "multi_token"})

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv_hidden: int = 64,
        se_ratio: int = 8,
        residual_blocks: int = 2,
        agg_mode: str = "mean",
    ):
        super().__init__()
        mode = str(agg_mode).strip().lower()
        if mode not in self._AGG_MODES:
            raise ValueError(f"hsi_agg_mode 须为 {sorted(self._AGG_MODES)}，当前 {agg_mode!r}")
        self.agg_mode = mode
        c = int(in_channels)
        h = max(32, int(conv_hidden))
        self.backbone_channels = h
        se_ratio = int(se_ratio)
        se_mid = max(8, h // max(1, se_ratio)) if se_ratio > 0 else 0
        self.stem = nn.Sequential(
            nn.Conv1d(1, h // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(h // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(h // 2, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Conv1d(h, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        )
        n_res = max(0, int(residual_blocks))
        self.res_blocks = nn.Sequential(*[_HSISpectralResidualBlock(h) for _ in range(n_res)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.se = (
            nn.Sequential(
                nn.Linear(h, se_mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(se_mid, h, bias=False),
                nn.Sigmoid(),
            )
            if se_ratio > 0
            else None
        )
        self._d_model = int(d_model)
        self.proj = nn.Linear(h, int(d_model))
        self.spatial_attn = nn.Linear(h, 1, bias=True) if self.agg_mode == "attn_pool" else None

    @property
    def n_output_tokens(self) -> int:
        """兼容旧工具脚本；主模型融合已改为空间 121×模态。"""
        return 3 if self.agg_mode == "multi_token" else 1

    def forward_spatial_map(self, hsi: torch.Tensor) -> torch.Tensor:
        """B×C×H×W → B×backbone_channels×11×11（光谱维已池化到 1D 特征/格点）。"""
        patch = _hsi_crop_11x11(hsi)
        b, c, _, _ = patch.shape
        n = 121
        x = patch.permute(0, 2, 3, 1).contiguous().view(b * n, c).unsqueeze(1)
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        if self.se is not None:
            gate = self.se(feat.mean(dim=2))
            feat = feat * gate.unsqueeze(2)
        feat = self.pool(feat).squeeze(-1)
        feat = feat.view(b, n, -1)
        hdim = feat.shape[-1]
        return feat.transpose(1, 2).reshape(b, hdim, 11, 11)

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        """旧聚合路径（工具/兼容性）：token 序列或单 token。"""
        patch = _hsi_crop_11x11(hsi)
        b, c, _, _ = patch.shape
        n = 121
        x = patch.permute(0, 2, 3, 1).contiguous().view(b * n, c).unsqueeze(1)
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        if self.se is not None:
            gate = self.se(feat.mean(dim=2))
            feat = feat * gate.unsqueeze(2)
        feat = self.pool(feat).squeeze(-1)
        feat = feat.view(b, n, -1)

        if self.agg_mode == "multi_token":
            center = feat[:, 60]
            corner = feat[:, [0, 10, 110, 120]].mean(dim=1)
            edge = feat[:, [5, 115, 55, 65]].mean(dim=1)
            toks = torch.stack([center, corner, edge], dim=1)
            return self.proj(toks)
        if self.agg_mode == "attn_pool":
            assert self.spatial_attn is not None
            w = F.softmax(self.spatial_attn(feat).squeeze(-1), dim=-1)
            feat = (feat * w.unsqueeze(-1)).sum(dim=1)
        else:
            feat = feat.mean(dim=1)
        return self.proj(feat)


class _LidarSpatialResidualBlock(nn.Module):
    """空间 2D 卷积残差块。"""
    def __init__(self, channels: int):
        super().__init__()
        ch = int(channels)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class LidarMorphEncoder(nn.Module):
    """LiDAR CNN；可输出空间特征图 B×fc×11×11。"""
    def __init__(
        self,
        in_ch: int,
        hidden: int,
        feat_ch: int,
        d_model: int,
        extra_blocks: int = 0,
    ):
        super().__init__()
        h = max(8, int(hidden))
        fc = max(16, int(feat_ch))
        self.feat_channels = fc
        self._d_model = int(d_model)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, fc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fc),
            nn.ReLU(inplace=True),
        )
        eb = max(0, int(extra_blocks))
        self.extra = (
            nn.Sequential(*[_LidarSpatialResidualBlock(fc) for _ in range(eb)])
            if eb > 0
            else nn.Identity()
        )
        self.proj = nn.Linear(fc, d_model)

    def forward_spatial(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.extra(self.stem(lidar))

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        feat = self.forward_spatial(lidar)
        # mean over spatial dims：确定性，等价于 adaptive_avg_pool2d(output_size=1)
        pooled = feat.mean(dim=(2, 3))
        return self.proj(pooled)


def _spatial_flatten(x: torch.Tensor) -> torch.Tensor:
    """B×D×11×11 → B×121×D"""
    b, d, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def _det_grid_avg_pool(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    """确定性 grid 平均池化：B×D×H×W → B×D×G×G。

    用等分 bin 手写，替换 adaptive_avg_pool2d（其 CUDA backward 无确定性实现）。
    H/W 须能被 grid_size 整除，否则最后一 bin 多包含一行/列（与 adaptive 语义一致）。
    对 H=W=11、G=6 的典型情况：前 5 个 bin 长 1，最后一个 bin 长 6（11=5×1+6/11 非整除
    时 adaptive 的 bin 边界按 floor/ceil 分配，此函数复现同一 bin 划分）。
    """
    b, d, h, w = x.shape
    g = int(grid_size)
    # 按 adaptive_avg_pool2d 的 bin 边界：start_i = floor(i*H/G), end_i = floor((i+1)*H/G)
    rows = [x[:, :, int(i * h // g):int((i + 1) * h // g), :] for i in range(g)]
    # 对每行 bin 再沿 W 分 bin
    cells = []
    for row_slice in rows:
        cols = [row_slice[:, :, :, int(j * w // g):int((j + 1) * w // g)] for j in range(g)]
        cells.append(torch.stack([c.mean(dim=(2, 3)) for c in cols], dim=-1))  # B×D×G
    # cells: list of G tensors each B×D×G → stack → B×D×G×G
    return torch.stack(cells, dim=2)  # B×D×G×G


def _make_spatial_distance_vector(num_modalities: int) -> torch.Tensor:
    """每个模态重复相同的 121 格欧氏距离（中心为 (5,5)）。"""
    yy, xx = torch.meshgrid(
        torch.arange(11), torch.arange(11), indexing="ij"
    )
    cy, cx = 5.0, 5.0
    dist = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2).reshape(-1)
    return dist.repeat(int(num_modalities))


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return str(default)
    return str(v).strip()


def _env_bool01(name: str, default: int = 0) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return bool(int(default))
    s = str(v).strip().lower()
    return s not in ("0", "", "false", "off", "no")


def _dist121_flat(center_y: float = 5.0, center_x: float = 5.0) -> torch.Tensor:
    """11×11 格点中心的欧氏距离（展平行列优先），长度为 121。"""
    yy, xx = torch.meshgrid(torch.arange(11), torch.arange(11), indexing="ij")
    return torch.sqrt((yy.float() - center_y) ** 2 + (xx.float() - center_x) ** 2).reshape(-1)


def _grid_centers_ij_float(grid_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """近似将 G×G 池化胞元中心映射到连续坐标 [0,10]²（与原 11×11 对齐）。"""
    g = int(grid_size)
    if g < 1:
        raise ValueError(f"GRID_SIZE 须 >= 1，当前 {grid_size}")
    jj, ii = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    scale = 10.0 / float(g)
    cy = jj.float() * scale + scale * 0.5
    cx = ii.float() * scale + scale * 0.5
    return cy.reshape(-1), cx.reshape(-1)


def _dist_from_center_for_grid_centers(grid_size: int, center_y: float = 5.0, center_x: float = 5.0) -> torch.Tensor:
    """G² 长度的距离向量，与 adaptive_avg_pool2d 输出的行列 flatten 顺序一致。"""
    cy, cx = _grid_centers_ij_float(grid_size)
    dist = torch.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2)
    return dist


def _row_softmax_aggregate_dist(linear_weight_rows: torch.Tensor, dist121: torch.Tensor) -> torch.Tensor:
    """rows softmax(W) @ dist121，每压缩 token 聚合到标量距离。"""
    w_row = torch.nn.functional.softmax(linear_weight_rows, dim=-1)
    return torch.matmul(w_row, dist121.unsqueeze(-1)).squeeze(-1)


class DualQueryCouplingSpatialFusionDecoderLayer(SpatialFusionDecoderLayer):
    """在 self-attention 子块后：用对组池化向量经 MLP 作为残差耦合到另一 query 组。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        *,
        global_query_tokens: int,
        center_query_tokens: int,
    ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation=activation)
        self.global_query_tokens = int(global_query_tokens)
        self.center_query_tokens = int(center_query_tokens)
        if self.global_query_tokens <= 0 or self.center_query_tokens <= 0:
            raise ValueError(
                f"global/center query 须为正数，当前 g={self.global_query_tokens} c={self.center_query_tokens}"
            )
        factor = max(1, _env_int("MMDIFF_COUPLING_HIDDEN_FACTOR", 4))
        hid = max(64, d_model // factor)
        self.sa_couple_g2c = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.sa_couple_c2g = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x, need_weights=False)
        g_qk = self.global_query_tokens
        c_qk = self.center_query_tokens
        split = g_qk + c_qk
        g = out[:, :g_qk]
        c = out[:, g_qk:split]
        g_ctx = g.mean(dim=1)
        c_ctx = c.mean(dim=1)
        c = c + self.sa_couple_g2c(g_ctx).unsqueeze(1)
        g = g + self.sa_couple_c2g(c_ctx).unsqueeze(1)
        out = torch.cat([g, c], dim=1)
        return self.dropout1(out)


def build_spatial_fusion_decoder_with_dual_query_coupling(
    *,
    global_query_tokens: int,
    center_query_tokens: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    activation: str = "gelu",
) -> SpatialFusionDecoder:
    layers = [
        DualQueryCouplingSpatialFusionDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation=activation,
            global_query_tokens=global_query_tokens,
            center_query_tokens=center_query_tokens,
        )
        for _ in range(num_layers)
    ]
    return SpatialFusionDecoder(layers)


class MultimodalClassifier(nn.Module):
    """
    空间融合：各模态保留 11×11 特征，对齐到 d_model 后经共享瓶颈，拼为 memory。
    可选可学习模态嵌入（MMDIFF_MODALITY_EMBED）加在 memory 上；Center query 对 memory 施加
    alpha*exp(-dist/tau) 的 logit bias，可由 MMDIFF_DISTANCE_BIAS_HSI_ONLY 限制为仅 HSI token 列。
    """
    SPATIAL_TOKENS = 121

    def __init__(self, opt, diffusion=None):
        super().__init__()
        if diffusion is not None:
            warnings.warn(
                "MultimodalClassifier 已移除扩散 RGB 路径，diffusion 参数将被忽略。",
                stacklevel=2,
            )
        self.opt = opt

        ds_cfg = opt.get("dataset", {})
        cls_cfg = opt.get("model_cls", {})
        train_cfg = opt.get("train", {})
        proj_cfg = opt.get("module_cast3") or {}

        enabled_modalities = list(cls_cfg.get("enabled_modalities") or [])
        if not enabled_modalities:
            enabled_modalities = ["hsi", "rgb", "lidar"]
        enabled_set = set(enabled_modalities)
        self.use_hsi = "hsi" in enabled_set
        self.use_rgb = "rgb" in enabled_set
        self.use_lidar = "lidar" in enabled_set
        if not (self.use_hsi or self.use_rgb or self.use_lidar):
            raise ValueError(f"enabled_modalities 不能为空：{enabled_modalities!r}")

        self.rgb_source = "student" if self.use_rgb else ""

        self.num_classes = int(cls_cfg.get("out_channels") or ds_cfg.get("n_cls") or 2)
        self.hsi_channels = int(ds_cfg.get("hsi_channels") or 32)
        self.lidar_in_ch = int(ds_cfg.get("lidar_channel") or 1)

        d_model = int(cls_cfg.get("token_dim") or 256)
        self.d_model = d_model
        nhead = int(cls_cfg.get("transformer_heads") or 4)
        if d_model % nhead != 0:
            raise ValueError(
                f"model_cls.token_dim={d_model} 必须能被 transformer_heads={nhead} 整除"
            )
        n_tx = int(cls_cfg.get("transformer_layers") or 2)
        ff = int(cls_cfg.get("transformer_ff_dim") or max(512, d_model * 2))
        tx_dropout = float(cls_cfg.get("transformer_dropout") or 0.1)
        head_hidden = int(cls_cfg.get("head_hidden") or 128)

        self.center_distance_bias_alpha = float(
            cls_cfg.get("center_distance_bias_alpha") or 0.2
        )
        self.center_distance_bias_tau = float(
            cls_cfg.get("center_distance_bias_tau") or 2.0
        )
        if self.center_distance_bias_tau <= 0.0:
            raise ValueError(
                f"center_distance_bias_tau 须 > 0，当前 {self.center_distance_bias_tau}"
            )

        lidar_hidden = int(proj_cfg.get("lidar_hidden") or 16)
        lidar_extra_blocks = int(proj_cfg.get("lidar_extra_blocks") or 0)
        lidar_feat_ch = max(32, lidar_hidden * 2)
        hsi_conv_hidden = int(proj_cfg.get("hsi_conv_hidden") or 64)
        hsi_se_ratio = int(proj_cfg.get("hsi_se_ratio") or 8)
        hsi_residual_blocks = int(proj_cfg.get("hsi_residual_blocks") or 2)
        hsi_agg_mode = str(proj_cfg.get("hsi_agg_mode") or "multi_token").strip().lower()

        g_qk = _env_int("MMDIFF_GLOBAL_QUERY_TOKENS", 4)
        c_qk = _env_int("MMDIFF_CENTER_QUERY_TOKENS", 4)
        if g_qk <= 0 or c_qk <= 0:
            raise ValueError("MMDIFF_GLOBAL_QUERY_TOKENS / MMDIFF_CENTER_QUERY_TOKENS 须为正整数")
        self.global_query_tokens = g_qk
        self.center_query_tokens = c_qk
        # 旧脚本/工具曾假设两侧对称；若 g≠c，请用 global_query_tokens / center_query_tokens
        self.query_tokens_per_query = g_qk

        self._use_modality_embed = _env_bool01("MMDIFF_MODALITY_EMBED", 1)
        self._distance_bias_hsi_only = _env_bool01("MMDIFF_DISTANCE_BIAS_HSI_ONLY", 1)
        self._hsi_mem_tokens = 0

        head_layers = _env_int("MMDIFF_CLS_HEAD_LAYERS", 2)

        self.rgb_student: Optional[LightweightRgbEncoder] = None
        if self.use_rgb:
            self.rgb_student = LightweightRgbEncoder(
                in_ch=3,
                patch_h=11,
                patch_w=11,
                d_model=d_model,
                num_tokens=1,
            )

        self.hsi_encoder = HSICenterSpectralEncoder(
            self.hsi_channels,
            d_model,
            conv_hidden=hsi_conv_hidden,
            se_ratio=hsi_se_ratio,
            residual_blocks=hsi_residual_blocks,
            agg_mode=hsi_agg_mode,
        )
        self.lidar_encoder = LidarMorphEncoder(
            self.lidar_in_ch,
            lidar_hidden,
            lidar_feat_ch,
            d_model,
            extra_blocks=lidar_extra_blocks,
        )

        raw_r2l = str(cls_cfg.get("rgb_to_lidar_guidance_mode") or "none").strip().upper()
        if raw_r2l in ("", "NONE", "OFF", "0", "FALSE"):
            self._rgb_to_lidar_guidance = "none"
        elif raw_r2l in ("FILM", "A"):
            self._rgb_to_lidar_guidance = "film"
        else:
            raise ValueError(
                "model_cls.rgb_to_lidar_guidance_mode 须为 none|film|A，当前 "
                f"{raw_r2l!r}"
            )
        if self._rgb_to_lidar_guidance == "film":
            if not (self.use_rgb and self.use_lidar):
                warnings.warn(
                    "rgb_to_lidar_guidance_mode=film 需要同时启用 rgb 与 lidar，已关闭引导。",
                    stacklevel=2,
                )
                self._rgb_to_lidar_guidance = "none"

        self.rgb_to_lidar_film_mlp: Optional[nn.Module] = None
        if self._rgb_to_lidar_guidance == "film":
            self.rgb_to_lidar_film_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 2 * d_model),
            )

        raw_mc_mode = (_env_str("MMDIFF_MEMORY_COMPRESS_MODE", "none") or "").lower()
        memory_compress_allowed = frozenset({"none", "grid", "linear", "latent"})
        if raw_mc_mode not in memory_compress_allowed:
            raise ValueError(
                f"MMDIFF_MEMORY_COMPRESS_MODE 须为 none|grid|linear|latent，当前 {raw_mc_mode!r}"
            )
        self.memory_compress_mode = raw_mc_mode
        memory_grid_sz = max(1, _env_int("MMDIFF_MEMORY_GRID_SIZE", 4))
        self.memory_grid_size = int(memory_grid_sz)
        memory_k = max(1, _env_int("MMDIFF_MEMORY_COMPRESS_TOKENS", 16))
        self.memory_compress_tokens = int(memory_k)
        self.memory_keep_center_token = _env_bool01("MMDIFF_MEMORY_KEEP_CENTER_TOKEN", 0)

        n_modal = int(sum([self.use_hsi, self.use_rgb, self.use_lidar]))
        tokens_per_modal_block = int(self.SPATIAL_TOKENS)
        if self.memory_compress_mode == "grid":
            tokens_per_modal_block = self.memory_grid_size * self.memory_grid_size
        elif self.memory_compress_mode in ("linear", "latent"):
            tokens_per_modal_block = self.memory_compress_tokens
        if self.memory_compress_mode != "none" and self.memory_keep_center_token:
            tokens_per_modal_block += 1
        self._tokens_per_modality_block = int(tokens_per_modal_block)
        if self.use_hsi:
            self._hsi_mem_tokens = self._tokens_per_modality_block
        self.mem_len = self._tokens_per_modality_block * n_modal

        self.memory_linear_compress_121_K: Optional[nn.Linear] = None
        self.memory_latent_attn: Optional[nn.MultiheadAttention] = None
        self.memory_latent_queries: Optional[nn.Parameter] = None
        self.memory_latent_dist_logits: Optional[nn.Parameter] = None
        if self.memory_compress_mode == "linear":
            self.memory_linear_compress_121_K = nn.Linear(
                self.SPATIAL_TOKENS, self.memory_compress_tokens, bias=False
            )
        elif self.memory_compress_mode == "latent":
            self.memory_latent_queries = nn.Parameter(torch.randn(self.memory_compress_tokens, d_model))
            self.memory_latent_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=tx_dropout,
                batch_first=True,
            )
            # 常量 buffer 上用 softmax(rows) @ dist121 聚合几何距离；与 latent 注意力分离
            self.memory_latent_dist_logits = nn.Parameter(
                torch.zeros(self.memory_compress_tokens, self.SPATIAL_TOKENS)
            )

        self._spatial_hsi_proj = nn.Conv2d(
            self.hsi_encoder.backbone_channels, d_model, kernel_size=1, bias=True
        )
        if self.use_rgb:
            assert self.rgb_student is not None
            self._spatial_rgb_proj = nn.Conv2d(
                self.rgb_student.hidden_channels, d_model, kernel_size=1, bias=True
            )
        self._spatial_lidar_proj = nn.Conv2d(
            self.lidar_encoder.feat_channels, d_model, kernel_size=1, bias=True
        )

        self._bottleneck_dim = 128
        self.spatial_bn_down = nn.Conv2d(d_model, self._bottleneck_dim, kernel_size=1, bias=True)
        self.spatial_bn_up = nn.Conv2d(self._bottleneck_dim, d_model, kernel_size=1, bias=True)

        self.pos_embed_mem = nn.Parameter(torch.randn(1, self.mem_len, d_model))
        self.modality_embed: Optional[nn.Parameter] = None
        if self._use_modality_embed:
            self.modality_embed = nn.Parameter(torch.zeros(n_modal, d_model))
        self.global_cls = nn.Parameter(torch.randn(1, g_qk, d_model))
        self.center_cls = nn.Parameter(torch.randn(1, c_qk, d_model))
        self.pos_embed_tgt = nn.Parameter(torch.randn(1, g_qk + c_qk, d_model))

        self.decoder = build_spatial_fusion_decoder_with_dual_query_coupling(
            global_query_tokens=g_qk,
            center_query_tokens=c_qk,
            d_model=d_model,
            nhead=nhead,
            num_layers=n_tx,
            dim_feedforward=ff,
            dropout=tx_dropout,
            activation="gelu",
        )

        dist121_cpu = _dist121_flat()
        blk: torch.Tensor
        if self.memory_compress_mode == "none":
            blk = dist121_cpu
        elif self.memory_compress_mode == "grid":
            blk = _dist_from_center_for_grid_centers(self.memory_grid_size).float()
            if self.memory_keep_center_token:
                z = blk.new_zeros(1)
                blk = torch.cat([blk, z], dim=0)
        elif self.memory_compress_mode == "linear":
            assert self.memory_linear_compress_121_K is not None
            wt = self.memory_linear_compress_121_K.weight.detach().cpu().float()
            blk = _row_softmax_aggregate_dist(wt, dist121_cpu.float())
            if self.memory_keep_center_token:
                z = blk.new_zeros(1)
                blk = torch.cat([blk, z], dim=0)
        else:
            assert self.memory_latent_dist_logits is not None
            wt = self.memory_latent_dist_logits.detach().cpu().float()
            blk = _row_softmax_aggregate_dist(wt, dist121_cpu.float())
            if self.memory_keep_center_token:
                z = blk.new_zeros(1)
                blk = torch.cat([blk, z], dim=0)
        self.register_buffer("_spatial_dist_mem", blk.repeat(n_modal))

        self.global_head = ClassifierHead(
            d_model, head_hidden, self.num_classes, num_hidden_layers=head_layers
        )
        self.center_head = ClassifierHead(
            d_model, head_hidden, self.num_classes, num_hidden_layers=head_layers
        )

        proj_dict = {
            "hsi": self.hsi_encoder,
            "lidar": self.lidar_encoder,
        }
        if self.use_rgb:
            proj_dict["rgb"] = self.rgb_student
        self.projections = nn.ModuleDict(proj_dict)

        self._init_weights(
            init_type=str(cls_cfg.get("init_type") or "kaiming"),
            scale=float(cls_cfg.get("scale") or 1.0),
        )
        nn.init.normal_(self.pos_embed_mem, std=0.02)
        nn.init.normal_(self.global_cls, std=0.02)
        nn.init.normal_(self.center_cls, std=0.02)
        nn.init.normal_(self.pos_embed_tgt, std=0.02)
        if self.memory_latent_queries is not None:
            nn.init.normal_(self.memory_latent_queries, std=0.02)

        self.loss_func = self._build_loss(cls_cfg)
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_bn_down(x)
        x = F.gelu(x)
        return self.spatial_bn_up(x)

    def _memory_tokens_from_spatial(self, feats_bottle: torch.Tensor) -> torch.Tensor:
        """B×D×11×11 瓶颈后特征图 → 单模态 memory token 序列 (B, L_token, D)。"""
        mode = self.memory_compress_mode
        bsz = feats_bottle.size(0)

        if mode == "none":
            toks = _spatial_flatten(feats_bottle)
        elif mode == "grid":
            # 确定性手写 grid 池化，替换 adaptive_avg_pool2d（backward 无确定性 CUDA 实现）
            pooled = _det_grid_avg_pool(feats_bottle, self.memory_grid_size)
            toks = pooled.flatten(2).transpose(1, 2).contiguous()
        elif mode == "linear":
            if self.memory_linear_compress_121_K is None:
                raise RuntimeError("linear compress 但未初始化 Linear")
            flat = _spatial_flatten(feats_bottle)
            w = self.memory_linear_compress_121_K.weight
            toks = torch.einsum("kj,bjd->bkd", w, flat)
        else:
            if self.memory_latent_queries is None or self.memory_latent_attn is None:
                raise RuntimeError("latent compress 但未初始化 MultiheadAttention / queries")
            flat_kv = _spatial_flatten(feats_bottle)
            q = self.memory_latent_queries.unsqueeze(0).expand(bsz, -1, -1).contiguous()
            toks_mha, _ = self.memory_latent_attn(q, flat_kv, flat_kv, need_weights=False)
            toks = toks_mha

        if mode != "none" and self.memory_keep_center_token:
            ctr = feats_bottle[:, :, 5, 5].unsqueeze(1).contiguous()
            toks = torch.cat([toks, ctr], dim=1)

        return toks

    def _build_cross_attn_logit_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """(1,1,Lq,mem_len)。

        center 行：bias = alpha * exp(-dist / tau)；若 MMDIFF_DISTANCE_BIAS_HSI_ONLY=1 且启用 HSI，
        仅前 _hsi_mem_tokens 列（HSI memory 块）获得该 bias。
        global 行（可选）：-MMDIFF_GLOBAL_ANTICENTER_BIAS * exp(-dist/tau)，与 center 反号拉大分工。
        """
        g_qk = self.global_query_tokens
        c_qk = self.center_query_tokens
        total_q = g_qk + c_qk
        mem_len = self.mem_len
        dist = self._spatial_dist_mem.to(device=device, dtype=dtype)
        b1d = self.center_distance_bias_alpha * torch.exp(-dist / self.center_distance_bias_tau)
        m = torch.zeros(total_q, mem_len, device=device, dtype=dtype)
        if self._distance_bias_hsi_only and self.use_hsi and self._hsi_mem_tokens > 0:
            hsi_b1d = self.center_distance_bias_alpha * torch.exp(
                -dist[: self._hsi_mem_tokens] / self.center_distance_bias_tau
            )
            m[g_qk : total_q, : self._hsi_mem_tokens] = hsi_b1d.unsqueeze(0).expand(c_qk, -1)
        else:
            m[g_qk : total_q, :] = b1d.unsqueeze(0).expand(c_qk, -1)

        g_ac = float(os.environ.get("MMDIFF_GLOBAL_ANTICENTER_BIAS") or 0.0)
        if g_ac != 0.0:
            neg = -g_ac * torch.exp(-dist / self.center_distance_bias_tau)
            m[:g_qk, :] = neg.unsqueeze(0).expand(g_qk, -1)

        return m.unsqueeze(0).unsqueeze(0)

    def refresh_optimizer_after_param_freeze(self) -> None:
        train_cfg = self.opt["train"]
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _build_loss(self, _cls_cfg) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _build_optimizer(self, train_cfg):
        optim_cfg = train_cfg.get("optimizer", {})
        optim_type = str(optim_cfg.get("type") or "adamw").lower()
        lr = float(optim_cfg.get("lr") or 1e-3)
        weight_decay = float(optim_cfg.get("weight_decay") or 0.0)
        betas = tuple(optim_cfg.get("betas") or (0.9, 0.999))
        params = [p for p in self.parameters() if p.requires_grad]
        if optim_type == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg):
        sched_cfg = train_cfg.get("scheduler", {})
        if not sched_cfg:
            self._scheduler_lr_total_steps = 0
            return None
        sch, ts = build_lr_scheduler(self.optimizer, train_cfg, self.opt)
        self._scheduler_lr_total_steps = int(ts)
        return sch

    def _init_weights(self, init_type: str, scale: float) -> None:
        init_name = init_type.lower()
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if init_name == "xavier":
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                module.weight.data.mul_(scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_tokens(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        memory_parts: List[torch.Tensor] = []
        b: Optional[int] = None
        rgb_map_d: Optional[torch.Tensor] = None

        if self.use_hsi:
            if "hsi" not in data_dict:
                raise KeyError("需要 hsi 模态，但 data_dict 中缺少 hsi")
            hsi = data_dict["hsi"]
            h = self.hsi_encoder.forward_spatial_map(hsi)
            h = self._spatial_hsi_proj(h)
            h = self._bottleneck(h)
            memory_parts.append(self._memory_tokens_from_spatial(h))
            b = hsi.shape[0]

        if self.use_rgb:
            if "rgb" not in data_dict:
                raise KeyError(
                    "需要 rgb 模态：请准备 train_rgb_patches.npy / test_rgb_patches.npy 并确保 MMDIFF_MODALITY_COMBO 含 rgb"
                )
            assert self.rgb_student is not None
            r = self.rgb_student.forward_spatial(data_dict["rgb"])
            r = self._spatial_rgb_proj(r)
            r = self._bottleneck(r)
            rgb_map_d = r
            memory_parts.append(self._memory_tokens_from_spatial(r))
            b = r.shape[0] if b is None else b

        if self.use_lidar:
            if "lidar" not in data_dict:
                raise KeyError("需要 lidar 模态，但 data_dict 中缺少 lidar")
            l = self.lidar_encoder.forward_spatial(data_dict["lidar"])
            l = self._spatial_lidar_proj(l)
            if self._rgb_to_lidar_guidance == "film":
                if self.rgb_to_lidar_film_mlp is None:
                    raise RuntimeError("rgb_to_lidar_guidance=film 但未初始化 rgb_to_lidar_film_mlp")
                if rgb_map_d is None:
                    raise RuntimeError("RGB→LiDAR FiLM 需要本 batch 含 RGB 特征图")
                rgb_ctx = rgb_map_d.mean(dim=(2, 3))
                gb = self.rgb_to_lidar_film_mlp(rgb_ctx)
                gamma, beta = gb.chunk(2, dim=-1)
                l = l * (1.0 + torch.tanh(gamma.unsqueeze(-1).unsqueeze(-1))) + beta.unsqueeze(
                    -1
                ).unsqueeze(-1)
            l = self._bottleneck(l)
            memory_parts.append(self._memory_tokens_from_spatial(l))
            b = l.shape[0] if b is None else b

        if b is None:
            raise RuntimeError("未启用任何模态，无法构造 memory")
        memory = torch.cat(memory_parts, dim=1)
        if memory.shape[1] != self.mem_len:
            raise RuntimeError(
                f"memory 长度 {memory.shape[1]} 与预期 {self.mem_len} 不一致"
            )
        if self.modality_embed is not None:
            modal_bias = self.modality_embed.repeat_interleave(
                self._tokens_per_modality_block, dim=0
            )
            memory = memory + modal_bias.unsqueeze(0)
        memory = memory + self.pos_embed_mem

        g_qk = self.global_query_tokens
        c_qk = self.center_query_tokens
        g_tokens = self.global_cls.expand(b, -1, -1)
        c_tokens = self.center_cls.expand(b, -1, -1)
        tgt = torch.cat([g_tokens, c_tokens], dim=1)
        tgt = tgt + self.pos_embed_tgt

        cross_bias = self._build_cross_attn_logit_bias(tgt.device, tgt.dtype)
        out, _ = self.decoder(tgt, memory, cross_logit_bias=cross_bias, need_attn_weights=False)

        global_rep = out[:, :g_qk, :].mean(dim=1)
        center_rep = out[:, g_qk : g_qk + c_qk, :].mean(dim=1)
        return global_rep, center_rep

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        g_rep, c_rep = self._forward_tokens(data_dict)
        logits_g = self.global_head(g_rep)
        logits_c = self.center_head(c_rep)
        if return_center_logits:
            return logits_g, logits_c
        return logits_c


def create_multimodal_classifier(opt, diffusion=None) -> MultimodalClassifier:
    return MultimodalClassifier(opt, diffusion=diffusion)
