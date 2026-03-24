from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
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


class HSICenterSpectralEncoder(nn.Module):
    """中心 3x3 光谱编码 -> 1 个 token 向量 (d_model)。"""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        flat = int(in_channels) * 9
        hid = max(d_model, flat // 2)
        self.net = nn.Sequential(
            nn.Linear(flat, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, d_model),
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        patch = _crop_center_3x3(hsi)
        b = patch.shape[0]
        v = patch.reshape(b, -1)
        return self.net(v)


class LidarMorphEncoder(nn.Module):
    """小 CNN 形态学特征；输出 global token 与 center token。"""
    def __init__(self, in_ch: int, hidden: int, feat_ch: int, d_model: int):
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
        self.proj_global = nn.Linear(fc, d_model)
        self.proj_center = nn.Linear(fc, d_model)

    def forward(self, lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(lidar)
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
    image_size: int,
    diffusion_t: int,
) -> Dict[str, int]:
    dev = next(diffusion.netG.parameters()).device
    dummy_rgb = torch.zeros(1, 3, image_size, image_size, device=dev, dtype=torch.float32)
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
    HSI：中心 3x3 光谱 -> 1 token
    RGB：冻结 UNet，单层 t，多尺度层各 1 token（与 FEAT_SCALES 一致）
    LiDAR：小 CNN -> global + center 共 2 token
    融合：Transformer Encoder + global_cls / center_cls
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
        cast3 = opt.get('module_cast3', {})

        self.num_classes = int(cls_cfg.get('out_channels') or ds_cfg.get('n_cls') or 2)
        self.hsi_channels = int(ds_cfg.get('hsi_channels') or 32)
        self.lidar_in_ch = int(ds_cfg.get('lidar_channel') or 1)

        ts = cls_cfg.get('t') or [50]
        self.diffusion_t = int(ts[0] if isinstance(ts, (list, tuple)) else ts)

        self.feat_layer_names: List[str] = list(cls_cfg.get('feat_scales') or [])
        if not self.feat_layer_names:
            raise ValueError('model_cls.feat_scales 不能为空')

        img_size = int(opt.get('model', {}).get('image_size') or 32)
        self._image_size = img_size

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

        lidar_hidden = int(cast3.get('lidar_hidden') or 16)
        lidar_feat_ch = max(32, lidar_hidden * 2)

        ch_map = _probe_diffusion_layer_channels(
            diffusion, self.feat_layer_names, img_size, self.diffusion_t,
        )
        # ModuleDict 的 key 不能含 "."（与 down_blocks.1 等 UNet 子模块名冲突），用 ModuleList 与 feat_layer_names 顺序对齐
        self.rgb_projs = nn.ModuleList(
            [RGBLayerToToken(ch_map[name], d_model) for name in self.feat_layer_names]
        )

        self.hsi_encoder = HSICenterSpectralEncoder(self.hsi_channels, d_model)
        self.lidar_encoder = LidarMorphEncoder(
            self.lidar_in_ch, lidar_hidden, lidar_feat_ch, d_model,
        )

        n_rgb = len(self.feat_layer_names)
        self.seq_len = 2 + 1 + n_rgb + 2
        self.global_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.center_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=tx_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_tx)

        self.global_head = ClassifierHead(d_model, head_hidden, self.num_classes)
        self.center_head = ClassifierHead(d_model, head_hidden, self.num_classes)

        self.projections = nn.ModuleDict(
            {
                'hsi': self.hsi_encoder,
                'lidar': self.lidar_encoder,
                'rgb': self.rgb_projs,
            }
        )

        self._init_weights(
            init_type=str(cls_cfg.get('init_type') or 'kaiming'),
            scale=float(cls_cfg.get('scale') or 1.0),
        )
        nn.init.normal_(self.global_cls, std=0.02)
        nn.init.normal_(self.center_cls, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

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
            return None
        total_epochs = int(train_cfg.get('n_epoch') or 1)
        steps_per_epoch = int(self.opt.get('len_train_dataloader') or 1)
        total_steps = max(1, total_epochs * steps_per_epoch)
        constant_ratio = float(sched_cfg.get('constant_ratio') or 0.8)
        gamma = float(sched_cfg.get('gamma') or 0.1)
        step_boundary = int(total_steps * constant_ratio)

        def lr_lambda(step: int) -> float:
            return 1.0 if step < step_boundary else gamma

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _init_weights(self, init_type: str, scale: float) -> None:
        init_name = init_type.lower()
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if init_name == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                module.weight.data.mul_(scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_tokens(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if 'rgb' not in data_dict:
            raise KeyError('需要 rgb 模态：请准备 train_rgb_patches.npy / test_rgb_patches.npy 并启用 USE_RGB_PATCHES')
        hsi = data_dict['hsi']
        lidar = data_dict['lidar']
        self.diffusion.feed_data(data_dict)
        layer_feats = self.diffusion.get_feats(self.diffusion_t, training=self.training)
        if not isinstance(layer_feats, dict):
            raise RuntimeError('扩散特征应为 dict（多层 hook），请检查 StudentDiffusionWrapper.feat_layers')

        hsi_tok = self.hsi_encoder(hsi)
        lidar_g, lidar_c = self.lidar_encoder(lidar)

        rgb_toks = []
        for name, proj in zip(self.feat_layer_names, self.rgb_projs):
            rgb_toks.append(proj(layer_feats[name]))
        rgb_stack = torch.stack(rgb_toks, dim=1)

        b = hsi_tok.shape[0]
        g_cls = self.global_cls.expand(b, -1, -1)
        c_cls = self.center_cls.expand(b, -1, -1)

        seq = torch.cat(
            [
                g_cls,
                c_cls,
                hsi_tok.unsqueeze(1),
                rgb_stack,
                lidar_g.unsqueeze(1),
                lidar_c.unsqueeze(1),
            ],
            dim=1,
        )
        if seq.shape[1] != self.seq_len:
            raise RuntimeError(f'seq 长度 {seq.shape[1]} 与预期 {self.seq_len} 不一致')
        seq = seq + self.pos_embed
        encoded = self.transformer(seq)
        return encoded[:, 0], encoded[:, 1]

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
    ):
        g_rep, c_rep = self._forward_tokens(data_dict)
        logits_g = self.global_head(g_rep)
        logits_c = self.center_head(c_rep)
        if return_center_logits:
            return logits_g, logits_c
        return logits_c


def create_multimodal_classifier(opt, diffusion=None) -> MultimodalClassifier:
    return MultimodalClassifier(opt, diffusion=diffusion)
