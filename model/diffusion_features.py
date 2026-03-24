# -*- coding: utf-8 -*-
"""与 ../GFDiff/train_distill 及 GFDiff/model/diffusion_features 一致：按 UNet 子模块名抓取中间特征。"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiLayerFeatureExtractor:
    """通过 register_forward_hook 同时捕获模型多个中间层的输出。"""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.features: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        module_dict = dict(model.named_modules())
        for name in layer_names:
            if name not in module_dict:
                available = [n for n, _ in model.named_modules() if n]
                raise ValueError(f"未找到层 '{name}'。可用（节选前 40 个）: {available[:40]}")
            hook = module_dict[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def fn(_module, _inp, out):
            self.features[name] = out[0] if isinstance(out, tuple) else out

        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def probe_align_layer_channels(
    model: nn.Module,
    layer_names: list[str],
    img_size: int,
    in_channels: int,
    device: torch.device,
) -> list[int]:
    """一次 dummy 前向，得到各对齐层输出通道数。"""
    extractor = MultiLayerFeatureExtractor(model, layer_names)
    dummy_x = torch.randn(1, in_channels, img_size, img_size, device=device)
    dummy_t = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        model(dummy_x, dummy_t)
    channels = [int(extractor.features[n].shape[1]) for n in layer_names]
    extractor.remove()
    return channels
