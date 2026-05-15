#!/usr/bin/env python3
"""轻量冒烟：MultimodalClassifier 在各 MMDIFF_MEMORY_COMPRESS_MODE 下 forward 维度一致。

须在「首次 import param」之前设置 `MMDIFF_MODALITY_COMBO`（及可选其它 param 级变量）。

  python utils/smoke_memory_compress.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_memory_env(mode: str, *, center_token: bool, grid_size: int = 4, k: int = 16) -> None:
    os.environ["MMDIFF_MEMORY_COMPRESS_MODE"] = mode
    os.environ["MMDIFF_MEMORY_KEEP_CENTER_TOKEN"] = "1" if center_token else "0"
    os.environ["MMDIFF_COUPLING_HIDDEN_FACTOR"] = "2"
    if mode == "grid":
        os.environ["MMDIFF_MEMORY_GRID_SIZE"] = str(grid_size)
        os.environ.pop("MMDIFF_MEMORY_COMPRESS_TOKENS", None)
    elif mode in ("linear", "latent"):
        os.environ["MMDIFF_MEMORY_COMPRESS_TOKENS"] = str(k)
        os.environ.pop("MMDIFF_MEMORY_GRID_SIZE", None)
    else:
        os.environ.pop("MMDIFF_MEMORY_GRID_SIZE", None)
        os.environ.pop("MMDIFF_MEMORY_COMPRESS_TOKENS", None)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("SKIP smoke_memory_compress: 未安装 torch（请在训练环境运行）")
        return 0

    os.environ.setdefault("MMDIFF_MODALITY_COMBO", "hsi+rgb+lidar")

    cases = [
        ("none", False),
        ("grid", False),
        ("linear", False),
        ("latent", False),
        ("grid", True),
        ("linear", True),
        ("latent", True),
    ]

    import param
    from model.multimodal import MultimodalClassifier

    opt = param.opt
    B = 2
    ds = opt["dataset"]
    hsi_c = int(ds.get("hsi_channels") or 48)
    n_cls = int(ds.get("n_cls") or 20)
    dd = {
        "hsi": torch.randn(B, hsi_c, 11, 11),
        "rgb": torch.randn(B, 3, 11, 11),
        "lidar": torch.randn(B, 1, 11, 11),
    }

    for mode, ct in cases:
        _configure_memory_env(mode, center_token=ct)
        model = MultimodalClassifier(opt).eval()
        with torch.no_grad():
            logits_c = model(dd)
            logits_g, logits_cc = model(dd, return_center_logits=True)
        assert logits_c.shape == (B, n_cls), (mode, ct, logits_c.shape)
        assert logits_g.shape == (B, n_cls)
        assert logits_cc.shape == (B, n_cls)
        bias = model._build_cross_attn_logit_bias(dd["hsi"].device, dd["hsi"].dtype)
        assert bias.shape[-1] == model.mem_len, (mode, ct, bias.shape, model.mem_len)
        print(f"OK mode={mode} ct={int(ct)} mem_len={model.mem_len}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
