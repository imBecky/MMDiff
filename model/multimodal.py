from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        focal = (1.0 - prob).pow(self.gamma)
        return F.nll_loss(focal * log_prob, targets)


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


class MultimodalClassifier(nn.Module):
    """Minimal multimodal classifier with global and center heads."""

    def __init__(self, opt, diffusion=None):
        super().__init__()
        self.opt = opt
        self.diffusion = diffusion

        ds_cfg = opt.get('dataset', {})
        cls_cfg = opt.get('model_cls', {})
        train_cfg = opt.get('train', {})
        self.modalities = list(ds_cfg.get('modalities') or ['hsi', 'lidar'])
        self.output_cm_size = int(cls_cfg.get('output_cm_size') or 3)
        self.num_classes = int(cls_cfg.get('out_channels') or ds_cfg.get('n_cls') or 2)

        hidden_channels = 32
        fusion_channels = 64
        head_hidden = 128

        channel_map = {
            'hsi': int(ds_cfg.get('hsi_channels') or 32),
            'lidar': int(ds_cfg.get('lidar_channel') or 1),
            'rgb': 3,
        }
        self.projections = nn.ModuleDict({
            name: ConvStem(channel_map[name], hidden_channels)
            for name in self.modalities
        })
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * len(self.modalities), fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )
        self.global_head = ClassifierHead(fusion_channels, head_hidden, self.num_classes)
        self.center_head = ClassifierHead(fusion_channels, head_hidden, self.num_classes)

        self._init_weights(
            init_type=str(cls_cfg.get('init_type') or 'kaiming'),
            scale=float(cls_cfg.get('scale') or 1.0),
        )
        self.loss_func = self._build_loss(cls_cfg)
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _build_loss(self, cls_cfg) -> nn.Module:
        loss_type = str(cls_cfg.get('loss_type') or 'ce').lower()
        if loss_type == 'focal':
            gamma = float(cls_cfg.get('focal_gamma') or 2.0)
            return FocalLoss(gamma=gamma)
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

    def _encode_modalities(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for name in self.modalities:
            if name not in data_dict:
                raise KeyError(f'missing modality "{name}" in data_dict')
            feats.append(self.projections[name](data_dict[name]))
        return self.fusion(torch.cat(feats, dim=1))

    def _extract_center_feature(self, feat_map: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feat_map.shape
        crop = max(1, min(self.output_cm_size, height, width))
        if crop % 2 == 0:
            crop = max(1, crop - 1)
        cy, cx = height // 2, width // 2
        half = crop // 2
        y0, y1 = max(0, cy - half), min(height, cy + half + 1)
        x0, x1 = max(0, cx - half), min(width, cx + half + 1)
        center_feat = feat_map[:, :, y0:y1, x0:x1]
        return center_feat.mean(dim=(2, 3))

    def forward(self, data_dict: Dict[str, torch.Tensor], return_center_logits: bool = False):
        feat_map = self._encode_modalities(data_dict)
        global_feat = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
        center_feat = self._extract_center_feature(feat_map)

        logits_g = self.global_head(global_feat)
        logits_c = self.center_head(center_feat)

        if return_center_logits:
            return logits_g, logits_c
        return logits_c


def create_multimodal_classifier(opt, diffusion=None) -> MultimodalClassifier:
    return MultimodalClassifier(opt, diffusion=diffusion)
