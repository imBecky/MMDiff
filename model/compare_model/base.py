"""对比模型基类：与 pipeline 契约一致（loss_func / optimizer / exp_lr_scheduler）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


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
        self.hsi_channels = int(ds_cfg.get('hsi_channels') or 50)
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
        total_epochs = int(train_cfg.get('n_epoch') or 1)
        steps_per_epoch = int(self.opt.get('len_train_dataloader') or 1)
        total_steps = max(1, total_epochs * steps_per_epoch)
        bound_override = int(self.opt.get('scheduler_lr_total_steps') or 0)
        if bound_override > 0:
            total_steps = bound_override
        self._scheduler_lr_total_steps = int(total_steps)
        step_ratios = sched_cfg.get('step_ratios')
        gammas_multi = sched_cfg.get('gammas')

        if isinstance(step_ratios, (list, tuple)) and len(step_ratios) >= 2:
            ratios = sorted(float(x) for x in step_ratios[:2])
            r0, r1 = max(0.0, min(1.0, ratios[0])), max(0.0, min(1.0, ratios[1]))
            if r1 <= r0:
                raise ValueError(f'scheduler.step_ratios 需为升序，当前 {step_ratios!r}')
            g = gammas_multi if isinstance(gammas_multi, (list, tuple)) else ()
            if len(g) < 2:
                raise ValueError('scheduler.gammas 需为长度≥2')
            g1, g2 = float(g[0]), float(g[1])
            b1 = int(total_steps * r0)
            b2 = int(total_steps * r1)
            b2 = max(b1 + 1, b2)

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

    def _hsi_lidar(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return data_dict['hsi'], data_dict['lidar']

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        return_center_logits: bool = False,
        return_supcon_proj: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError
