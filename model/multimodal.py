from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.unet_hw import unet_sample_hw


def _unet_input_hw(diffusion) -> Tuple[int, int]:
    """与 get_feats 内 resize 目标一致（单源：unet_sample_hw）。"""
    return unet_sample_hw(diffusion.netG)


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



def _crop_center_3x3(x: torch.Tensor) -> torch.Tensor:
    """B,C,H,W -> B,C,3,3，不足则边缘复制 pad。"""
    _, c, h, w = x.shape
    need_y = max(0, 3 - h)
    need_x = max(0, 3 - w)
    if need_y or need_x:
        pad_top = need_y // 2
        pad_bottom = need_y - pad_top
        pad_left = need_x // 2
        pad_right = need_x - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        h, w = x.shape[2], x.shape[3]
    cy, cx = h // 2, w // 2
    y0, y1 = cy - 1, cy + 2
    x0, x1 = cx - 1, cx + 2
    return x[:, :, y0:y1, x0:x1]


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
    """
    中心 3x3：9 个像素各做 1D 光谱卷积（沿波段维），stem 后经若干光谱残差块加深，再全局池化、
    空间聚合、SE 通道门控，投影为 1 或多个 token。

    agg_mode:
      - mean: 9 位置特征算术平均 -> 1 token（原默认）
      - attn_pool: 对 9 位置学 softmax 权重再加权求和 -> 1 token（D1）
      - multi_token: 中心 / 四角均值 / 四边均值 -> 3 token（D2）
    """
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
        se_mid = max(8, h // max(1, int(se_ratio)))
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
        self.se = nn.Sequential(
            nn.Linear(h, se_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(se_mid, h, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(h, int(d_model))
        self.spatial_attn = nn.Linear(h, 1, bias=True) if self.agg_mode == 'attn_pool' else None

    @property
    def n_output_tokens(self) -> int:
        return 3 if self.agg_mode == 'multi_token' else 1

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        patch = _crop_center_3x3(hsi)
        b, c, _, _ = patch.shape
        x = patch.permute(0, 2, 3, 1).contiguous().view(b * 9, c).unsqueeze(1)
        feat = self.stem(x)
        feat = self.res_blocks(feat)
        feat = self.pool(feat).squeeze(-1)
        feat = feat.view(b, 9, -1)

        if self.agg_mode == 'multi_token':
            # 3x3 下标: 0 1 2 / 3 4 5 / 6 7 8 — 中心、四角、四边
            center = feat[:, 4]
            corner = feat[:, [0, 2, 6, 8]].mean(dim=1)
            edge = feat[:, [1, 3, 5, 7]].mean(dim=1)
            toks = torch.stack([center, corner, edge], dim=1)
            bsz, k, hd = toks.shape
            flat = toks.reshape(bsz * k, hd)
            flat = flat * self.se(flat)
            toks = flat.reshape(bsz, k, hd)
            return self.proj(toks)

        if self.agg_mode == 'attn_pool':
            assert self.spatial_attn is not None
            w = F.softmax(self.spatial_attn(feat).squeeze(-1), dim=-1)
            feat = (feat * w.unsqueeze(-1)).sum(dim=1)
        else:
            feat = feat.mean(dim=1)

        feat = feat * self.se(feat)
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
    """小 CNN 形态学特征；stem 后可选若干空间残差块加深；输出 global token 与 center token。"""
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
        self.proj_global = nn.Linear(fc, d_model)
        self.proj_center = nn.Linear(fc, d_model)

    def forward(self, lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extra(self.stem(lidar))
        g = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
        c_patch = _crop_center_3x3(feat)
        c = c_patch.mean(dim=(2, 3))
        return self.proj_global(g), self.proj_center(c)


class RGBLayerToToken(nn.Module):
    """单层扩散特征图 B,C,H,W -> B,d_model。"""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, d_model)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat).flatten(1)
        return self.fc(x)


def _probe_diffusion_layer_channels(
    diffusion,
    feat_names: List[str],
    diffusion_t: int,
) -> Dict[str, int]:
    dev = next(diffusion.netG.parameters()).device
    th, tw = _unet_input_hw(diffusion)
    dummy_rgb = torch.zeros(1, 3, th, tw, device=dev, dtype=torch.float32)
    dummy_idx = torch.zeros(1, dtype=torch.long, device=dev)
    diffusion.feed_data({'rgb': dummy_rgb, 'sample_indices': dummy_idx})
    out = diffusion.get_feats(diffusion_t, training=False)
    if not isinstance(out, dict):
        raise RuntimeError('diffusion.get_feats 应返回层名字典，请检查 feat_layers 配置')
    ch = {}
    for name in feat_names:
        if name not in out:
            raise KeyError(f'扩散特征缺少层 {name!r}，当前键: {list(out.keys())}')
        ch[name] = int(out[name].shape[1])
    return ch


class MultimodalClassifier(nn.Module):
    """
    HSI：中心 3x3，1D 光谱卷积 + SE；空间聚合可为 1 token（mean/attn_pool）或 3 token（multi_token）
    RGB：冻结 UNet，可多时间步 t，每个 t 下多尺度层各 1 token（与 FEAT_SCALES 一致）
    LiDAR：小 CNN（stem + 可选空间残差块）-> global + center 共 2 token
    融合：两枚可学习 CLS 为 query，模态 token 为 memory，TransformerDecoder（交叉注意力 + CLS 间自注意力）
    双头：global_head(cls[0])，center_head(cls[1])
    """
    def __init__(self, opt, diffusion=None):
        super().__init__()
        if diffusion is None:
            raise ValueError('MultimodalClassifier 需要注入 diffusion（StudentDiffusionWrapper）')
        self.opt = opt
        self.diffusion = diffusion

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

        self.num_classes = int(cls_cfg.get('out_channels') or ds_cfg.get('n_cls') or 2)
        self.hsi_channels = int(ds_cfg.get('hsi_channels') or 32)
        self.lidar_in_ch = int(ds_cfg.get('lidar_channel') or 1)

        ts = cls_cfg.get('t') or [50]
        self.diffusion_ts: List[int] = (
            [int(x) for x in ts] if isinstance(ts, (list, tuple)) else [int(ts)]
        )
        if not self.diffusion_ts:
            raise ValueError('model_cls.t 不能为空')
        self.diffusion_t = self.diffusion_ts[0]

        self.feat_layer_names: List[str] = list(cls_cfg.get('feat_scales') or [])
        if not self.feat_layer_names:
            raise ValueError('model_cls.feat_scales 不能为空')

        self._unet_input_hw = _unet_input_hw(diffusion)

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
        hsi_agg_mode = str(proj_cfg.get('hsi_agg_mode') or 'mean').strip().lower()

        ch_map = _probe_diffusion_layer_channels(
            diffusion, self.feat_layer_names, self.diffusion_t,
        )
        # ModuleDict 的 key 不能含 "."（与 down_blocks.1 等 UNet 子模块名冲突），用 ModuleList 与 feat_layer_names 顺序对齐
        self.rgb_projs = nn.ModuleList(
            [RGBLayerToToken(ch_map[name], d_model) for name in self.feat_layer_names]
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

        n_hsi = int(self.hsi_encoder.n_output_tokens) if self.use_hsi else 0
        n_rgb = (
            len(self.diffusion_ts) * len(self.feat_layer_names)
            if self.use_rgb
            else 0
        )
        n_lidar = 2 if self.use_lidar else 0
        self.mem_len = n_hsi + n_rgb + n_lidar  # 启用模态 token 拼接长度
        self.seq_len = 2 + self.mem_len  # 两枚 CLS + memory 长度（日志用）

        self.global_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.center_cls = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_embed_mem = nn.Parameter(torch.randn(1, self.mem_len, d_model))
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
            'rgb': self.rgb_projs,
        }
        if self.use_supcon and self.supcon_proj is not None:
            proj_dict['supcon'] = self.supcon_proj
        self.projections = nn.ModuleDict(proj_dict)

        self._init_weights(
            init_type=str(cls_cfg.get('init_type') or 'kaiming'),
            scale=float(cls_cfg.get('scale') or 1.0),
        )
        nn.init.normal_(self.global_cls, std=0.02)
        nn.init.normal_(self.center_cls, std=0.02)
        nn.init.normal_(self.pos_embed_mem, std=0.02)
        nn.init.normal_(self.pos_embed_tgt, std=0.02)

        self.loss_func = self._build_loss(cls_cfg)
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
        total_epochs = int(train_cfg.get('n_epoch') or 1)
        steps_per_epoch = int(self.opt.get('len_train_dataloader') or 1)
        total_steps = max(1, total_epochs * steps_per_epoch)
        # 续训时 opt['scheduler_lr_total_steps'] 为「首次训练」用于计算衰减边界的总 optimizer step 数；
        # 若仍用新 n_epoch 重算边界，会与 checkpoint 里 last_epoch 错位，导致 LR 从 schedule 起点跳变。
        bound_override = int(self.opt.get('scheduler_lr_total_steps') or 0)
        if bound_override > 0:
            total_steps = bound_override
        self._scheduler_lr_total_steps = int(total_steps)
        step_ratios = sched_cfg.get('step_ratios')
        gammas_multi = sched_cfg.get('gammas')

        def _two_step_lambda() -> Tuple[float, float, float, float]:
            """
            返回 (b1, b2, g1, g2)：step>=b1 乘 g1，step>=b2 再乘 g2（相对 base 的累积乘子为 g1 或 g1*g2）。
            """
            if not isinstance(step_ratios, (list, tuple)) or len(step_ratios) < 2:
                raise ValueError('scheduler.step_ratios 需为长度≥2的列表')
            ratios = sorted(float(x) for x in step_ratios[:2])
            r0, r1 = max(0.0, min(1.0, ratios[0])), max(0.0, min(1.0, ratios[1]))
            if r1 <= r0:
                raise ValueError(f'scheduler.step_ratios 需为升序，当前 {step_ratios!r}')
            g = gammas_multi if isinstance(gammas_multi, (list, tuple)) else ()
            if len(g) < 2:
                raise ValueError('scheduler.gammas 需为长度≥2，与两阶衰减对应')
            g1, g2 = float(g[0]), float(g[1])
            b1 = int(total_steps * r0)
            b2 = int(total_steps * r1)
            b2 = max(b1 + 1, b2)
            return float(b1), float(b2), g1, g2

        if isinstance(step_ratios, (list, tuple)) and len(step_ratios) >= 2:
            b1, b2, g1, g2 = _two_step_lambda()

            def lr_lambda(step: int) -> float:
                m = 1.0
                if step >= b1:
                    m *= g1
                if step >= b2:
                    m *= g2
                return m

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        constant_ratio = float(sched_cfg.get('constant_ratio') or 0.8)
        gamma = float(sched_cfg.get('gamma') or 0.1)
        step_boundary = int(total_steps * constant_ratio)

        def lr_lambda_legacy(step: int) -> float:
            return 1.0 if step < step_boundary else gamma

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda_legacy)

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

        # 先准备各模态 token（根据 enabled_modalities 动态裁剪）
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
                raise KeyError('需要 rgb 模态：请准备 train_rgb_patches.npy / test_rgb_patches.npy 并确保 MMDIFF_MODALITY_COMBO 含 rgb')
            self.diffusion.feed_data(data_dict)

            rgb_toks: List[torch.Tensor] = []
            for t in self.diffusion_ts:
                layer_feats = self.diffusion.get_feats(t, training=self.training)
                if not isinstance(layer_feats, dict):
                    raise RuntimeError(
                        '扩散特征应为 dict（多层 hook），请检查 StudentDiffusionWrapper.feat_layers'
                    )
                for name, proj in zip(self.feat_layer_names, self.rgb_projs):
                    rgb_toks.append(proj(layer_feats[name]))
            rgb_stack = torch.stack(rgb_toks, dim=1)
            memory_parts.append(rgb_stack)
            b = rgb_stack.shape[0] if b is None else b

        if self.use_lidar:
            if 'lidar' not in data_dict:
                raise KeyError('需要 lidar 模态，但 data_dict 中缺少 lidar')
            lidar = data_dict['lidar']
            lidar_g, lidar_c = self.lidar_encoder(lidar)
            memory_parts.append(lidar_g.unsqueeze(1))
            memory_parts.append(lidar_c.unsqueeze(1))
            b = lidar_g.shape[0] if b is None else b

        if b is None:
            raise RuntimeError('未启用任何模态，无法构造 memory')
        if not memory_parts:
            raise RuntimeError('memory_parts 为空，无法拼接 token')

        g_cls = self.global_cls.expand(b, -1, -1)
        c_cls = self.center_cls.expand(b, -1, -1)

        memory = torch.cat(memory_parts, dim=1)
        if memory.shape[1] != self.mem_len:
            raise RuntimeError(f'memory 长度 {memory.shape[1]} 与预期 {self.mem_len} 不一致')
        memory = memory + self.pos_embed_mem
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
