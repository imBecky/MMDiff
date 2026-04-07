#!/usr/bin/env python3
"""
HSICenterSpectralEncoder（gate-before-pool）随机输入前向 + 反传自检：
输出范数、SE gate 分布、各子模块梯度范数。若 gate 塌缩到常数或 stem 梯度近零，可辅助判断「结构更聪明」是否带来优化问题。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from param import HSI_CHANNELS

from model.multimodal import HSICenterSpectralEncoder, _crop_center_3x3


def _gate_stats(enc: HSICenterSpectralEncoder, hsi: torch.Tensor) -> dict[str, float]:
    # 仅统计 gate 分布，不参与反传，避免 float(.) 对带 grad 张量报警
    with torch.no_grad():
        patch = _crop_center_3x3(hsi)
        b, c, _, _ = patch.shape
        x = patch.permute(0, 2, 3, 1).contiguous().view(b * 9, c).unsqueeze(1)
        feat = enc.stem(x)
        feat = enc.res_blocks(feat)
        gate = enc.se(feat.mean(dim=2))
    return {
        'gate_mean': float(gate.mean()),
        'gate_std': float(gate.std()),
        'gate_min': float(gate.min()),
        'gate_max': float(gate.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description='HSI 分支 gate-before-pool 自检')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--channels', type=int, default=HSI_CHANNELS)
    p.add_argument('--patch', type=int, default=11)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--conv-hidden', type=int, default=96)
    p.add_argument('--se-ratio', type=int, default=16)
    p.add_argument('--res-blocks', type=int, default=6)
    p.add_argument('--agg-mode', type=str, default='attn_pool')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    enc = HSICenterSpectralEncoder(
        in_channels=args.channels,
        d_model=args.d_model,
        conv_hidden=args.conv_hidden,
        se_ratio=args.se_ratio,
        residual_blocks=args.res_blocks,
        agg_mode=args.agg_mode,
    )
    hsi = torch.randn(args.batch, args.channels, args.patch, args.patch, requires_grad=True)
    out = enc(hsi)
    loss = out.float().sum()
    loss.backward()

    gs = _gate_stats(enc, hsi.detach())
    print('HSI branch sanity (random input)')
    print(f"  out shape={tuple(out.shape)} out_mean={out.detach().mean():.6f} out_std={out.detach().std():.6f}")
    print(
        f"  gate: mean={gs['gate_mean']:.4f} std={gs['gate_std']:.4f} "
        f"min={gs['gate_min']:.4f} max={gs['gate_max']:.4f}"
    )

    norms: list[tuple[str, float]] = []
    for name, param in enc.named_parameters():
        if param.grad is None:
            continue
        tag = 'stem' if name.startswith('stem') else 'se' if name.startswith('se') else 'res' if name.startswith('res_blocks') else 'proj' if name.startswith('proj') else 'spatial_attn' if name.startswith('spatial_attn') else 'other'
        norms.append((tag, float(param.grad.norm())))

    by_tag: dict[str, list[float]] = {}
    for tag, n in norms:
        by_tag.setdefault(tag, []).append(n)
    print('  grad L2 norm (sum over params by prefix):')
    for tag in sorted(by_tag.keys()):
        s = sum(by_tag[tag])
        print(f"    {tag}: {s:.6f}")

    # 若 SE gate 在随机输入上 std≈0，说明门控可能过平；真实数据上再对照一次更可靠。
    if gs['gate_std'] < 1e-4:
        print('  [hint] gate std 极小：随机输入下门控近似常数，可对照真实 batch 再观察。')


if __name__ == '__main__':
    main()
