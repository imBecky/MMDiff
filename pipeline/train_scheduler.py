"""分类训练学习率调度（LambdaLR，按 optimizer step）。"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict[str, Any],
    opt: Dict[str, Any],
) -> Tuple[Optional[torch.optim.lr_scheduler.LambdaLR], int]:
    sched_cfg = train_cfg.get('scheduler') or {}
    if not sched_cfg:
        return None, 0

    te = int(train_cfg.get('n_epoch') or 1)
    spe = int(opt.get('len_train_dataloader') or 1)
    total_steps = max(1, te * spe)
    bo = int(opt.get('scheduler_lr_total_steps') or 0)
    if bo > 0:
        total_steps = bo
    total_steps = int(total_steps)

    name = str(sched_cfg.get('name') or 'piecewise_two_step').lower().strip()

    if name in ('cosine', 'cosine_annealing'):
        eta = float(sched_cfg.get('eta_min_ratio', 0.01))
        eta = max(0.0, min(1.0, eta))
        wr = float(sched_cfg.get('warmup_ratio', 0.0))
        wr = max(0.0, min(1.0, wr))
        wsteps = int(sched_cfg.get('warmup_steps') or 0)
        if wsteps > 0:
            warmup_steps = min(wsteps, total_steps)
        else:
            warmup_steps = max(0, min(int(round(total_steps * wr)), total_steps))

        def lr_lambda(step: int) -> float:
            if warmup_steps <= 0:
                denom = max(float(total_steps - 1), 1.0)
                t = min(float(step), float(total_steps - 1)) / denom
                return eta + (1.0 - eta) * 0.5 * (1.0 + math.cos(math.pi * t))
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            n_cos = total_steps - warmup_steps
            if n_cos <= 1:
                return float(eta)
            rel = step - warmup_steps
            t = min(float(rel), float(n_cos - 1)) / float(max(n_cos - 1, 1))
            return eta + (1.0 - eta) * 0.5 * (1.0 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), total_steps

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
        b2 = max(b1 + 1, int(total_steps * r1))

        def lr_lambda_pw(step: int) -> float:
            m = 1.0
            if step >= b1:
                m *= g1
            if step >= b2:
                m *= g2
            return m

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_pw), total_steps

    cr = float(sched_cfg.get('constant_ratio') or 0.8)
    gm = float(sched_cfg.get('gamma') or 0.1)
    boundary = int(total_steps * cr)

    def lr_lambda_legacy(step: int) -> float:
        return 1.0 if step < boundary else gm

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_legacy), total_steps
