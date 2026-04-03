#!/usr/bin/env python3
"""
离线预计算用于蒸馏的RGB扩散模型中间特征（与 MultimodalClassifier 中 RGBLayerToToken 后一致），
写入 npy + meta.json。训练集形状 (N_train, 4, num_tokens, d_model)；测试集 (N_test, 1, ...)，仅 rot_k=0。

用法（在仓库根目录，已激活 conda 环境）:
  python utils/precompute_rgb_teacher_tokens.py --split train --batch-size 32
  python utils/precompute_rgb_teacher_tokens.py --split test --batch-size 32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.format import open_memmap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from param import (  # noqa: E402
    CLS_DIFFUSION_TIMESTEPS,
    CLS_TOKEN_DIM,
    DATA_DIR,
    FEAT_SCALES,
    PATCH_WINDOW_SIZE,
    RANDOM_SEED,
    STUDENT_CHECKPOINT,
    STUDENT_NUM_TRAIN_TIMESTEPS,
    TRAIN_LABELS_PATH,
    TRAIN_RGB_PATCHES_PATH,
    TEST_LABELS_PATH,
    DIFFUSION_NOISE_MODE,
    DIFFUSION_NORMALIZE_INPUT,
)
from model.multimodal import RGBLayerToToken, _probe_diffusion_layer_channels  # noqa: E402
from pipeline.data import _apply_rot_k, _crop_patch_hwc  # noqa: E402
from pipeline.rgb_teacher_cache import default_meta, save_meta  # noqa: E402
from pipeline.student_diffusion import StudentDiffusionWrapper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='预计算 RGB teacher token 缓存')
    p.add_argument('--split', choices=('train', 'test'), default='train')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--out-train', type=str, default='', help='覆盖默认 train 输出路径')
    p.add_argument('--out-test', type=str, default='', help='覆盖默认 test 输出路径')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def _load_rgb_volume(split: str) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    if split == 'train':
        labels = np.load(TRAIN_LABELS_PATH).astype(np.int64, copy=True)
        rgb = np.load(TRAIN_RGB_PATCHES_PATH, mmap_mode='r')
        out_npy = Path(DATA_DIR) / 'rgb_teacher_tokens_train.npy'
        out_meta = Path(DATA_DIR) / 'rgb_teacher_tokens_train.meta.json'
    else:
        labels = np.load(TEST_LABELS_PATH).astype(np.int64, copy=True)
        rgb = np.load(TRAIN_RGB_PATCHES_PATH, mmap_mode='r')
        out_npy = Path(DATA_DIR) / 'rgb_teacher_tokens_test.npy'
        out_meta = Path(DATA_DIR) / 'rgb_teacher_tokens_test.meta.json'
    return rgb, labels, out_npy, out_meta


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    split = args.split
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rgb_vol, labels, out_npy, out_meta = _load_rgb_volume(split)
    if split == 'train' and args.out_train:
        out_npy = Path(args.out_train).with_suffix('.npy')
        out_meta = Path(args.out_train).with_suffix('.meta.json')
    if split == 'test' and args.out_test:
        out_npy = Path(args.out_test).with_suffix('.npy')
        out_meta = Path(args.out_test).with_suffix('.meta.json')

    n = len(labels)
    diffusion_ts = list(CLS_DIFFUSION_TIMESTEPS)
    feat_names = list(FEAT_SCALES)
    num_tokens = len(diffusion_ts) * len(feat_names)
    d_model = int(CLS_TOKEN_DIM)

    diffusion = StudentDiffusionWrapper(
        STUDENT_CHECKPOINT,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        noise_mode=DIFFUSION_NOISE_MODE,
        noise_seed_base=RANDOM_SEED,
        normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
        feat_layers=feat_names,
    )
    t0 = int(diffusion_ts[0])
    ch_map = _probe_diffusion_layer_channels(diffusion, feat_names, t0)
    rgb_projs = nn.ModuleList(
        [RGBLayerToToken(ch_map[name], d_model) for name in feat_names]
    ).to(device)
    rgb_projs.eval()

    if split == 'train':
        n_rot = 4
        out_shape = (n, n_rot, num_tokens, d_model)
    else:
        n_rot = 1
        out_shape = (n, n_rot, num_tokens, d_model)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    # 须为标准 .npy（含 magic），否则 np.load/mmap_tokens 会误解析
    fp = open_memmap(str(out_npy), mode='w+', dtype=np.float32, shape=out_shape)

    w = int(PATCH_WINDOW_SIZE)
    bs = max(1, int(args.batch_size))

    def run_batch(
        rows: list[int],
        cols: list[int],
        global_rows: list[int],
        rot_ks: list[int],
    ) -> np.ndarray:
        if not rows:
            raise ValueError('empty batch')
        tensors = []
        for row, col, rk in zip(rows, cols, rot_ks):
            rp = _crop_patch_hwc(rgb_vol, row, col, w)
            if rk:
                rp = _apply_rot_k(rp, rk)
            t = np.transpose(rp, (2, 0, 1)).astype(np.float32, copy=False)
            tensors.append(t)
        rgb = torch.from_numpy(np.stack(tensors, axis=0)).to(device)
        sample_indices = torch.tensor(global_rows, dtype=torch.long, device=device)
        data_dict = {'rgb': rgb, 'sample_indices': sample_indices}
        diffusion.feed_data(data_dict)
        toks: list[torch.Tensor] = []
        for t in diffusion_ts:
            layer_feats = diffusion.get_feats(t, training=False)
            if not isinstance(layer_feats, dict):
                raise RuntimeError('get_feats 应返回 dict')
            for name, proj in zip(feat_names, rgb_projs):
                toks.append(proj(layer_feats[name]))
        stack = torch.stack(toks, dim=1).cpu().numpy().astype(np.float32)
        return stack

    if split == 'train':
        for rk in range(4):
            for start in tqdm(
                range(0, n, bs),
                desc=f'precompute train rot_k={rk}',
                unit='batch',
                leave=True,
            ):
                end = min(start + bs, n)
                rows = [int(labels[i, 1]) for i in range(start, end)]
                cols = [int(labels[i, 2]) for i in range(start, end)]
                grs = list(range(start, end))
                rks = [rk] * (end - start)
                stack = run_batch(rows, cols, grs, rks)
                fp[start:end, rk, :, :] = stack
            fp.flush()
    else:
        for start in tqdm(
            range(0, n, bs),
            desc='precompute test rot_k=0',
            unit='batch',
            leave=True,
        ):
            end = min(start + bs, n)
            rows = [int(labels[i, 1]) for i in range(start, end)]
            cols = [int(labels[i, 2]) for i in range(start, end)]
            grs = list(range(start, end))
            rks = [0] * (end - start)
            stack = run_batch(rows, cols, grs, rks)
            fp[start:end, 0, :, :] = stack
        fp.flush()

    del fp
    meta = default_meta(
        n_rows=n,
        n_rot=n_rot,
        num_tokens=num_tokens,
        d_model=d_model,
        feat_scales=feat_names,
        diffusion_ts=diffusion_ts,
        student_checkpoint=str(STUDENT_CHECKPOINT),
    )
    meta['split'] = split
    save_meta(out_meta, meta)
    print(f'写入 {out_npy} shape={out_shape}')
    print(f'meta -> {out_meta}')


if __name__ == '__main__':
    main()
