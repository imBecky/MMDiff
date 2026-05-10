from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.train_scheduler import build_lr_scheduler

from model.rgb_student import LightweightRgbEncoder
from model.spatial_fusion_decoder import build_spatial_fusion_decoder


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_classes),
        )

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
        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return self.proj(pooled)


def _spatial_flatten(x: torch.Tensor) -> torch.Tensor:
    """B×D×11×11 → B×121×D"""
    b, d, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def _make_spatial_distance_vector(num_modalities: int) -> torch.Tensor:
    """每个模态重复相同的 121 格欧氏距离（中心为 (5,5)）。"""
    yy, xx = torch.meshgrid(
        torch.arange(11), torch.arange(11), indexing="ij"
    )
    cy, cx = 5.0, 5.0
    dist = torch.sqrt((yy.float() - cy) ** 2 + (xx.float() - cx) ** 2).reshape(-1)
    return dist.repeat(int(num_modalities))


class MultimodalClassifier(nn.Module):
    """
    空间融合：各模态保留 11×11 特征，对齐到 d_model 后经共享瓶颈，拼为 memory。
    Center query 在 cross-attention 上对 memory 位置施加 -alpha*dist 的 logit bias。
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

        lidar_hidden = int(proj_cfg.get("lidar_hidden") or 16)
        lidar_extra_blocks = int(proj_cfg.get("lidar_extra_blocks") or 0)
        lidar_feat_ch = max(32, lidar_hidden * 2)
        hsi_conv_hidden = int(proj_cfg.get("hsi_conv_hidden") or 64)
        hsi_se_ratio = int(proj_cfg.get("hsi_se_ratio") or 8)
        hsi_residual_blocks = int(proj_cfg.get("hsi_residual_blocks") or 2)
        hsi_agg_mode = str(proj_cfg.get("hsi_agg_mode") or "multi_token").strip().lower()

        self.query_tokens_per_query = 4
        qk = self.query_tokens_per_query

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

        n_spatial = int(self.SPATIAL_TOKENS)
        n_hsi = n_spatial if self.use_hsi else 0
        n_rgb = n_spatial if self.use_rgb else 0
        n_lidar = n_spatial if self.use_lidar else 0
        self.mem_len = n_hsi + n_rgb + n_lidar

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
        self.global_cls = nn.Parameter(torch.randn(1, qk, d_model))
        self.center_cls = nn.Parameter(torch.randn(1, qk, d_model))
        self.pos_embed_tgt = nn.Parameter(torch.randn(1, 2 * qk, d_model))

        self.decoder = build_spatial_fusion_decoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=n_tx,
            dim_feedforward=ff,
            dropout=tx_dropout,
            activation="gelu",
        )

        nm = sum([self.use_hsi, self.use_rgb, self.use_lidar])
        self.register_buffer("_spatial_dist_mem", _make_spatial_distance_vector(nm))

        self.global_head = ClassifierHead(d_model, head_hidden, self.num_classes)
        self.center_head = ClassifierHead(d_model, head_hidden, self.num_classes)

        self.use_supcon = bool(cls_cfg.get("use_supcon", False))
        supcon_dim = int(cls_cfg.get("supcon_proj_dim") or 128)
        if self.use_supcon:
            self.supcon_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, supcon_dim),
            )
        else:
            self.supcon_proj = None

        proj_dict = {
            "hsi": self.hsi_encoder,
            "lidar": self.lidar_encoder,
        }
        if self.use_rgb:
            proj_dict["rgb"] = self.rgb_student
        if self.use_supcon and self.supcon_proj is not None:
            proj_dict["supcon"] = self.supcon_proj
        self.projections = nn.ModuleDict(proj_dict)

        self._init_weights(
            init_type=str(cls_cfg.get("init_type") or "kaiming"),
            scale=float(cls_cfg.get("scale") or 1.0),
        )
        nn.init.normal_(self.pos_embed_mem, std=0.02)
        nn.init.normal_(self.global_cls, std=0.02)
        nn.init.normal_(self.center_cls, std=0.02)
        nn.init.normal_(self.pos_embed_tgt, std=0.02)

        self.loss_func = self._build_loss(cls_cfg)
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_bn_down(x)
        x = F.gelu(x)
        return self.spatial_bn_up(x)

    def _build_cross_attn_logit_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """(1,1,Lq,mem_len)，仅 center query 行非零。"""
        qk = self.query_tokens_per_query
        mem_len = self.mem_len
        dist = self._spatial_dist_mem.to(device=device, dtype=dtype)
        b1d = -self.center_distance_bias_alpha * dist
        m = torch.zeros(2 * qk, mem_len, device=device, dtype=dtype)
        m[qk : 2 * qk, :] = b1d.unsqueeze(0).expand(qk, -1)
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
            memory_parts.append(_spatial_flatten(h))
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
            memory_parts.append(_spatial_flatten(r))
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
            memory_parts.append(_spatial_flatten(l))
            b = l.shape[0] if b is None else b

        if b is None:
            raise RuntimeError("未启用任何模态，无法构造 memory")
        memory = torch.cat(memory_parts, dim=1)
        if memory.shape[1] != self.mem_len:
            raise RuntimeError(
                f"memory 长度 {memory.shape[1]} 与预期 {self.mem_len} 不一致"
            )
        memory = memory + self.pos_embed_mem

        qk = self.query_tokens_per_query
        g_tokens = self.global_cls.expand(b, -1, -1)
        c_tokens = self.center_cls.expand(b, -1, -1)
        tgt = torch.cat([g_tokens, c_tokens], dim=1)
        tgt = tgt + self.pos_embed_tgt

        cross_bias = self._build_cross_attn_logit_bias(tgt.device, tgt.dtype)
        out, _ = self.decoder(tgt, memory, cross_logit_bias=cross_bias, need_attn_weights=False)

        global_rep = out[:, :qk, :].mean(dim=1)
        center_rep = out[:, qk:, :].mean(dim=1)
        return global_rep, center_rep

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        g_rep, c_rep = self._forward_tokens(data_dict)
        logits_g = self.global_head(g_rep)
        logits_c = self.center_head(c_rep)
        z = None
        if return_supcon_proj:
            if self.supcon_proj is None:
                raise RuntimeError("return_supcon_proj=True 但 model_cls.use_supcon 未启用")
            z = self.supcon_proj(c_rep)

        if return_center_logits:
            if return_supcon_proj:
                return logits_g, logits_c, z
            return logits_g, logits_c
        if return_supcon_proj:
            return logits_c, z
        return logits_c


def create_multimodal_classifier(opt, diffusion=None) -> MultimodalClassifier:
    return MultimodalClassifier(opt, diffusion=diffusion)
