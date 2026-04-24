from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.train_scheduler import build_lr_scheduler

from model.rgb_student import LightweightRgbEncoder


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
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        h, w = x.shape[2], x.shape[3]
    cy, cx = h // 2, w // 2
    y0 = cy - ph // 2
    x0 = cx - pw // 2
    return x[:, :, y0 : y0 + ph, x0 : x0 + pw]


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
    """HSI 固定 11×11：每位置 1D 光谱卷积 + SE + 空间聚合 -> token(s)。agg_mode: mean | attn_pool | multi_token。"""
    _AGG_MODES = frozenset({'mean', 'attn_pool', 'multi_token'})

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv_hidden: int = 64,
        se_ratio: int = 8,
        residual_blocks: int = 2,
        agg_mode: str = 'mean',
    ):
        super().__init__()
        mode = str(agg_mode).strip().lower()
        if mode not in self._AGG_MODES:
            raise ValueError(f'hsi_agg_mode 须为 {sorted(self._AGG_MODES)}，当前 {agg_mode!r}')
        self.agg_mode = mode
        c = int(in_channels)
        h = max(32, int(conv_hidden))
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
        self.proj = nn.Linear(h, int(d_model))
        self.spatial_attn = nn.Linear(h, 1, bias=True) if self.agg_mode == 'attn_pool' else None

    @property
    def n_output_tokens(self) -> int:
        return 3 if self.agg_mode == 'multi_token' else 1

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
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

        if self.agg_mode == 'multi_token':
            # 行主序 11×11：中心 60；四角 0,10,110,120；四边中 5,115,55,65
            center = feat[:, 60]
            corner = feat[:, [0, 10, 110, 120]].mean(dim=1)
            edge = feat[:, [5, 115, 55, 65]].mean(dim=1)
            toks = torch.stack([center, corner, edge], dim=1)
            return self.proj(toks)

        if self.agg_mode == 'attn_pool':
            assert self.spatial_attn is not None
            w = F.softmax(self.spatial_attn(feat).squeeze(-1), dim=-1)
            feat = (feat * w.unsqueeze(-1)).sum(dim=1)
        else:
            feat = feat.mean(dim=1) # now use

        return self.proj(feat)


class _LidarSpatialResidualBlock(nn.Module):
    """空间 2D 卷积残差块：Conv-BN-ReLU-Conv-BN + 恒等映射（与 HSI 光谱残差块对称）。"""
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
    """小 CNN 形态学特征；stem 后可选若干空间残差块加深；输出单个 LiDAR token。"""
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

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        feat = self.extra(self.stem(lidar))
        pooled = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        return self.proj(pooled)


class MultimodalClassifier(nn.Module):
    """
    HSI：固定 11×11，每位置 1D 光谱卷积 + SE；聚合为 1 或 3 token
    RGB：轻量 CNN（LightweightRgbEncoder）从 patch 得到 1 个 token（不使用扩散模型）
    LiDAR：小 CNN（stem + 可选空间残差块）-> 1 token
    融合：两枚可学习 CLS 为 query，模态 token 为 memory，TransformerDecoder（交叉注意力）
    双头：global_head(cls[0])，center_head(cls[1])
    """
    def __init__(self, opt, diffusion=None):
        super().__init__()
        if diffusion is not None:
            warnings.warn(
                'MultimodalClassifier 已移除扩散 RGB 路径，diffusion 参数将被忽略。',
                stacklevel=2,
            )
        self.opt = opt

        ds_cfg = opt.get('dataset', {})
        cls_cfg = opt.get('model_cls', {})
        train_cfg = opt.get('train', {})
        proj_cfg = opt.get('module_cast3') or {}

        enabled_modalities = list(cls_cfg.get('enabled_modalities') or [])
        if not enabled_modalities:
            # 兼容旧配置：默认仍开启三模态
            enabled_modalities = ['hsi', 'rgb', 'lidar']
        enabled_set = set(enabled_modalities)
        self.use_hsi = 'hsi' in enabled_set
        self.use_rgb = 'rgb' in enabled_set
        self.use_lidar = 'lidar' in enabled_set
        if not (self.use_hsi or self.use_rgb or self.use_lidar):
            raise ValueError(f'enabled_modalities 不能为空：{enabled_modalities!r}')

        # RGB 仅走轻量编码器；忽略 opt 中历史字段 rgb_source / feat_scales / t
        self.rgb_source = 'student' if self.use_rgb else ''

        self.num_classes = int(cls_cfg.get('out_channels') or ds_cfg.get('n_cls') or 2)
        self.hsi_channels = int(ds_cfg.get('hsi_channels') or 32)
        self.lidar_in_ch = int(ds_cfg.get('lidar_channel') or 1)

        d_model = int(cls_cfg.get('token_dim') or 256)
        self.d_model = d_model
        nhead = int(cls_cfg.get('transformer_heads') or 4)
        if d_model % nhead != 0:
            raise ValueError(
                f'model_cls.token_dim={d_model} 必须能被 transformer_heads={nhead} 整除'
            )
        n_tx = int(cls_cfg.get('transformer_layers') or 2)
        ff = int(cls_cfg.get('transformer_ff_dim') or max(512, d_model * 2))
        tx_dropout = float(cls_cfg.get('transformer_dropout') or 0.1)
        head_hidden = int(cls_cfg.get('head_hidden') or 128)

        lidar_hidden = int(proj_cfg.get('lidar_hidden') or 16)
        lidar_extra_blocks = int(proj_cfg.get('lidar_extra_blocks') or 0)
        lidar_feat_ch = max(32, lidar_hidden * 2)
        hsi_conv_hidden = int(proj_cfg.get('hsi_conv_hidden') or 64)
        hsi_se_ratio = int(proj_cfg.get('hsi_se_ratio') or 8)
        hsi_residual_blocks = int(proj_cfg.get('hsi_residual_blocks') or 2)
        hsi_agg_mode = str(proj_cfg.get('hsi_agg_mode') or 'multi_token').strip().lower()

        self.rgb_num_tokens = 1 if self.use_rgb else 0
        self.rgb_student: Optional[LightweightRgbEncoder] = None
        if self.use_rgb:
            self.rgb_student = LightweightRgbEncoder(
                in_ch=3,
                patch_h=11,
                patch_w=11,
                d_model=d_model,
                num_tokens=self.rgb_num_tokens,
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

        raw_r2l = str(cls_cfg.get('rgb_to_lidar_guidance_mode') or 'none').strip().upper()
        if raw_r2l in ('', 'NONE', 'OFF', '0', 'FALSE'):
            self._rgb_to_lidar_guidance = 'none'
        elif raw_r2l in ('FILM', 'A'):
            self._rgb_to_lidar_guidance = 'film'
        else:
            raise ValueError(
                'model_cls.rgb_to_lidar_guidance_mode 须为 none|film|A，当前 '
                f'{raw_r2l!r}'
            )
        if self._rgb_to_lidar_guidance == 'film':
            if not (self.use_rgb and self.use_lidar):
                warnings.warn(
                    'rgb_to_lidar_guidance_mode=film 需要同时启用 rgb 与 lidar，已关闭引导。',
                    stacklevel=2,
                )
                self._rgb_to_lidar_guidance = 'none'

        self.rgb_to_lidar_film_mlp: Optional[nn.Module] = None
        if self._rgb_to_lidar_guidance == 'film':
            self.rgb_to_lidar_film_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, 2 * d_model),
            )

        n_hsi = int(self.hsi_encoder.n_output_tokens) if self.use_hsi else 0
        n_rgb = (
            self.rgb_num_tokens
            if self.use_rgb
            else 0
        )
        n_lidar = 1 if self.use_lidar else 0
        self.mem_len = n_hsi + n_rgb + n_lidar  # 启用模态 token 拼接长度
        self.seq_len = 2 + self.mem_len
        self.pos_embed_mem = nn.Parameter(torch.randn(1, self.mem_len, d_model))
        self.global_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.center_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed_tgt = nn.Parameter(torch.randn(1, 2, d_model))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=tx_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_tx)

        self.global_head = ClassifierHead(d_model, head_hidden, self.num_classes)
        self.center_head = ClassifierHead(d_model, head_hidden, self.num_classes)

        self.use_supcon = bool(cls_cfg.get('use_supcon', False))
        supcon_dim = int(cls_cfg.get('supcon_proj_dim') or 128)
        if self.use_supcon:
            self.supcon_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, supcon_dim),
            )
        else:
            self.supcon_proj = None

        proj_dict = {
            'hsi': self.hsi_encoder,
            'lidar': self.lidar_encoder,
        }
        if self.use_rgb:
            proj_dict['rgb'] = self.rgb_student
        if self.use_supcon and self.supcon_proj is not None:
            proj_dict['supcon'] = self.supcon_proj
        self.projections = nn.ModuleDict(proj_dict)

        self._init_weights(
            init_type=str(cls_cfg.get('init_type') or 'kaiming'),
            scale=float(cls_cfg.get('scale') or 1.0),
        )
        nn.init.normal_(self.pos_embed_mem, std=0.02)
        nn.init.normal_(self.global_cls, std=0.02)
        nn.init.normal_(self.center_cls, std=0.02)
        nn.init.normal_(self.pos_embed_tgt, std=0.02)

        self.loss_func = self._build_loss(cls_cfg)
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def refresh_optimizer_after_param_freeze(self) -> None:
        """在 load_state_dict 之后将某子模块设为 requires_grad=False 时调用，仅优化仍可训练参数。"""
        train_cfg = self.opt['train']
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _build_loss(self, _cls_cfg) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _build_optimizer(self, train_cfg):
        optim_cfg = train_cfg.get('optimizer', {})
        optim_type = str(optim_cfg.get('type') or 'adamw').lower()
        lr = float(optim_cfg.get('lr') or 1e-3)
        weight_decay = float(optim_cfg.get('weight_decay') or 0.0)
        betas = tuple(optim_cfg.get('betas') or (0.9, 0.999))
        params = [p for p in self.parameters() if p.requires_grad]
        if optim_type == 'adam':
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg):
        sched_cfg = train_cfg.get('scheduler', {})
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
                if init_name == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.weight.data.mul_(scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_tokens(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        memory_parts: List[torch.Tensor] = []
        b = None
        rgb_stack: Optional[torch.Tensor] = None

        if self.use_hsi:
            if 'hsi' not in data_dict:
                raise KeyError('需要 hsi 模态，但 data_dict 中缺少 hsi')
            hsi = data_dict['hsi']
            hsi_tok = self.hsi_encoder(hsi)
            if hsi_tok.dim() == 3:
                hsi_seq = hsi_tok
            else:
                hsi_seq = hsi_tok.unsqueeze(1)
            memory_parts.append(hsi_seq)
            b = hsi_seq.shape[0]

        if self.use_rgb:
            if 'rgb' not in data_dict:
                raise KeyError(
                    '需要 rgb 模态：请准备 train_rgb_patches.npy / test_rgb_patches.npy 并确保 MMDIFF_MODALITY_COMBO 含 rgb'
                )
            assert self.rgb_student is not None
            rgb_stack = self.rgb_student(data_dict['rgb'])
            memory_parts.append(rgb_stack)
            b = rgb_stack.shape[0] if b is None else b

        if self.use_lidar:
            if 'lidar' not in data_dict:
                raise KeyError('需要 lidar 模态，但 data_dict 中缺少 lidar')
            lidar_tok = self.lidar_encoder(data_dict['lidar'])
            if self._rgb_to_lidar_guidance == 'film':
                if self.rgb_to_lidar_film_mlp is None:
                    raise RuntimeError('rgb_to_lidar_guidance=film 但未初始化 rgb_to_lidar_film_mlp')
                if rgb_stack is None:
                    raise RuntimeError('RGB→LiDAR FiLM 需要本 batch 含 RGB 特征（rgb_stack）')
                rgb_ctx = rgb_stack.mean(dim=1)
                gb = self.rgb_to_lidar_film_mlp(rgb_ctx)
                gamma, beta = gb.chunk(2, dim=-1)
                lidar_tok = lidar_tok * (1.0 + torch.tanh(gamma)) + beta
            memory_parts.append(lidar_tok.unsqueeze(1))
            b = lidar_tok.shape[0] if b is None else b

        if b is None:
            raise RuntimeError('未启用任何模态，无法构造 memory')
        if not memory_parts:
            raise RuntimeError('memory_parts 为空，无法拼接 token')

        memory = torch.cat(memory_parts, dim=1)
        if memory.shape[1] != self.mem_len:
            raise RuntimeError(f'memory 长度 {memory.shape[1]} 与预期 {self.mem_len} 不一致')
        memory = memory + self.pos_embed_mem
        g_cls = self.global_cls.expand(b, -1, -1)
        c_cls = self.center_cls.expand(b, -1, -1)
        tgt = torch.cat([g_cls, c_cls], dim=1)
        tgt = tgt + self.pos_embed_tgt
        out = self.decoder(tgt, memory)
        return out[:, 0], out[:, 1]

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
                raise RuntimeError('return_supcon_proj=True 但 model_cls.use_supcon 未启用')
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
