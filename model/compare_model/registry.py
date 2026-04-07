"""对比模型注册表：名称 -> 类。"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Type

import torch.nn as nn

from .architectures import (
    DFINetClassifier,
    ExViTClassifier,
    FGCNNClassifier,
    FusAtNetClassifier,
    MACNClassifier,
    TwoBranchCNNClassifier,
)

CompareFactory = Callable[[Any, Any], nn.Module]

REGISTRY: Dict[str, Type[nn.Module]] = {
    'fgcnn': FGCNNClassifier,
    'f_gcnn': FGCNNClassifier,
    # 旧名兼容
    'fgcn': FGCNNClassifier,
    'f_gcn': FGCNNClassifier,
    'fusatnet': FusAtNetClassifier,
    'fus_at_net': FusAtNetClassifier,
    'exvit': ExViTClassifier,
    'mvit': ExViTClassifier,
    'ex_vit': ExViTClassifier,
    # Xu et al. 2017 双分支 CNN（BUCT Keras 仓库 PyTorch 复现）
    'two_branch_cnn': TwoBranchCNNClassifier,
    'two_branch': TwoBranchCNNClassifier,
    'twobranch_cnn': TwoBranchCNNClassifier,
    'xu2017_ms': TwoBranchCNNClassifier,
    # Gao et al. 2022 DFINet（HSI+MSI；MSI→LiDAR patch）
    # 官方：https://github.com/formango/HSI_MSI_Multisource_Classification
    'dfinet': DFINetClassifier,
    'dfi_net': DFINetClassifier,
    'dfi': DFINetClassifier,
    'formango_dfinet': DFINetClassifier,
    'hsi_msi_multisource': DFINetClassifier,
    # Li et al. 2023 MACN（like413/MACN）
    'macn': MACNClassifier,
}

# 须走 pipeline/dfinet_protocol.py（联合损失+SGD）的注册名 = 所有指向 DFINetClassifier 的键
DFINET_PROTOCOL_COMPARE_NAMES = frozenset(
    n for n, cls in REGISTRY.items() if cls is DFINetClassifier
)


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
