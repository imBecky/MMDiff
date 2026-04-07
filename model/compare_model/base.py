"""对比模型基类：与 pipeline 契约一致（loss_func / optimizer / exp_lr_scheduler）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from param import HSI_CHANNELS

from pipeline.train_scheduler import build_lr_scheduler


class CompareClassifierBase(nn.Module):
    """
    对比实验用分类器基类。
    - 仅使用 HSI + LiDAR（忽略 RGB / 扩散，diffusion 参数保留以匹配 create_classifier 签名）。
    - forward 支持 return_center_logits / return_supcon_proj（对比跑法下 pipeline 不会启用）。
    """

    def __init__(self, opt: Any, diffusion: Any = None):
        super().__init__()
        self.opt = opt
        self.diffusion = diffusion
        ds_cfg = opt.get('dataset', {})
        self.num_classes = int(ds_cfg.get('n_cls') or opt.get('model_cls', {}).get('out_channels') or 20)
        self.hsi_channels = int(ds_cfg.get('hsi_channels') or HSI_CHANNELS)
        self.lidar_channels = int(ds_cfg.get('lidar_channel') or 1)

        # 优化器须在子类注册完所有 Parameter 后再建，否则 parameters() 为空会报错
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = None  # type: ignore
        self.exp_lr_scheduler = None

    def _init_optimizer_and_scheduler(self) -> None:
        """子类在定义完全部卷积/线性层后调用。"""
        train_cfg = self.opt.get('train', {})
        self.optimizer = self._build_optimizer(train_cfg)
        self.exp_lr_scheduler = self._build_scheduler(train_cfg)

    def _build_optimizer(self, train_cfg: Dict) -> torch.optim.Optimizer:
        optim_cfg = train_cfg.get('optimizer', {})
        optim_type = str(optim_cfg.get('type') or 'adamw').lower()
        lr = float(optim_cfg.get('lr') or 1e-3)
        weight_decay = float(optim_cfg.get('weight_decay') or 0.0)
        betas = tuple(optim_cfg.get('betas') or (0.9, 0.999))
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                '优化器参数列表为空：请确认子类在 _init_optimizer_and_scheduler() 之前已注册全部可训练层'
            )
        if optim_type == 'adam':
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg: Dict):
        sched_cfg = train_cfg.get('scheduler', {})
        if not sched_cfg:
            self._scheduler_lr_total_steps = 0
            return None
        sch, ts = build_lr_scheduler(self.optimizer, train_cfg, self.opt)
        self._scheduler_lr_total_steps = int(ts)
        return sch

    def _hsi_lidar(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return data_dict['hsi'], data_dict['lidar']

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError
