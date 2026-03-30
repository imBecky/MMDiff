"""对比模型注册表：名称 -> 类。仅包含有公开官方实现的方法之 PyTorch 对照。"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Type

import torch.nn as nn

from .architectures import (
    CoupledCNNClassifier,
    DCMNetClassifier,
    ExViTClassifier,
    FusAtNetClassifier,
    HCTClassifier,
    MACNClassifier,
    MSFMambaClassifier,
    SSMAEClassifier,
)

CompareFactory = Callable[[Any, Any], nn.Module]

REGISTRY: Dict[str, Type[nn.Module]] = {
    'coupled_cnn': CoupledCNNClassifier,
    'fusatnet': FusAtNetClassifier,
    'macn': MACNClassifier,
    'hct': HCTClassifier,
    'exvit': ExViTClassifier,
    'ss_mae': SSMAEClassifier,
    'ss-mae': SSMAEClassifier,
    'msfmamba': MSFMambaClassifier,
    'dcmnet': DCMNetClassifier,
}


def resolve_compare_model_name() -> str:
    raw = (os.environ.get('MMDIFF_COMPARE_MODEL') or '').strip().lower()
    return raw.replace('-', '_')


def create_compare_classifier(opt: Any, diffusion: Any = None) -> nn.Module:
    name = resolve_compare_model_name()
    if not name:
        raise ValueError(
            '未指定对比模型：请设环境变量 MMDIFF_COMPARE_MODEL 或使用 main_compare.py --model'
        )
    if name not in REGISTRY:
        raise ValueError(
            f'未知对比模型 {name!r}。可选: {", ".join(sorted(set(REGISTRY.keys())))}'
        )
    cls = REGISTRY[name]
    return cls(opt, diffusion=diffusion)
