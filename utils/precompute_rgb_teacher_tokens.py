#!/usr/bin/env python3
"""
离线预计算 RGB 扩散教师 token（与 MultimodalClassifier 中 RGBLayerToToken 后一致），
写入 npy + meta.json。输入为 HR 严格视野裁块（rgb_hr.npy），双线性到 HR_TEACHER_INPUT_SIZE 再喂冻结 UNet。

默认提取层（偏纹理：浅层 down + 解码 up）::
  down_blocks.0, down_blocks.1, up_blocks.1

训练时使用 cached_teacher 须与 param.FEAT_SCALES（及 token维）一致，否则需重算缓存或改 --feat-layers。

用法（在仓库根目录）:
  python utils/precompute_rgb_teacher_tokens.py --split train --batch-size 32
  python utils/precompute_rgb_teacher_tokens.py --split test --batch-size 32
  python utils/precompute_rgb_teacher_tokens.py --split train --feat-layers down_blocks.0,down_blocks.1,up_blocks.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DEFAULT_FEAT_SCALES = (
    'down_blocks.0',
    'down_blocks.1',
    'up_blocks.1',
)

from param import (  # noqa: E402
    CLS_DIFFUSION_TIMESTEPS,
    CLS_TOKEN_DIM,
    DATA_DIR,
    HR_TEACHER_INPUT_SIZE,
    PATCH_WINDOW_SIZE,
    RANDOM_SEED,
    RGB_DIFFUSION_TEACHER_CHECKPOINT,
    STUDENT_NUM_TRAIN_TIMESTEPS,
    TRAIN_LABELS_PATH,
    TRAIN_RGB_HR_PATH,
    TEST_LABELS_PATH,
    DIFFUSION_NOISE_MODE,
    DIFFUSION_NORMALIZE_INPUT,
)
from model.multimodal import RGBLayerToToken, _probe_diffusion_layer_channels  # noqa: E402
from pipeline.data import (  # noqa: E402
    _apply_rot_k,
    _crop_hr_strict_hwc,
    load_rgb_hr_meta,
)
from pipeline.rgb_teacher_cache import default_meta, save_meta  # noqa: E402
from pipeline.student_diffusion import StudentDiffusionWrapper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='预计算 RGB 扩散教师 token 缓存（HR 严格视野）')
    p.add_argument('--split', choices=('train', 'test'), default='train')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--out-train', type=str, default='', help='覆盖默认 train 输出路径')
    p.add_argument('--out-test', type=str, default='', help='覆盖默认 test 输出路径')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument(
        '--feat-layers',
        type=str,
        default=','.join(_DEFAULT_FEAT_SCALES),
        help='逗号分隔 UNet 子模块名（与 diffusers UNet2DModel 一致），默认 down_blocks.0,down_blocks.1,up_blocks.1',
    )
    return p.parse_args()


def _load_rgb_volume(split: str) -> tuple[np.ndarray, np.ndarray, Path, Path, int, int]:
    if split == 'train':
        labels = np.load(TRAIN_LABELS_PATH).astype(np.int64, copy=True)
        out_npy = Path(DATA_DIR) / 'rgb_teacher_tokens_train_strict.npy'
    else:
        labels = np.load(TEST_LABELS_PATH).astype(np.int64, copy=True)
        out_npy = Path(DATA_DIR) / 'rgb_teacher_tokens_test_strict.npy'
    out_meta = out_npy.with_suffix('.meta.json')
    if not TRAIN_RGB_HR_PATH.is_file():
        raise FileNotFoundError(f'需要 {TRAIN_RGB_HR_PATH}（请先运行 data_prepare.py）')
    rgb = np.load(TRAIN_RGB_HR_PATH, mmap_mode='r')
    meta = load_rgb_hr_meta()
    rh = int(meta['rh'])
    rw = int(meta['rw'])
    if rh != rw:
        raise ValueError(f'严格视野预计算要求 rh==rw，当前 rh={rh} rw={rw}')
    return rgb, labels, out_npy, out_meta, rh, rw


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    split = args.split
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rgb_vol, labels, out_npy, out_meta, rh, rw = _load_rgb_volume(split)
    if split == 'train' and args.out_train:
        out_npy = Path(args.out_train).with_suffix('.npy')
        out_meta = Path(args.out_train).with_suffix('.meta.json')
    if split == 'test' and args.out_test:
        out_npy = Path(args.out_test).with_suffix('.npy')
        out_meta = Path(args.out_test).with_suffix('.meta.json')

    n = len(labels)
    diffusion_ts = list(CLS_DIFFUSION_TIMESTEPS)
    feat_names = [x.strip() for x in (args.feat_layers or '').split(',') if x.strip()]
    if not feat_names:
        raise ValueError('--feat-layers 解析为空，请传入逗号分隔层名，例如 down_blocks.0,down_blocks.1,up_blocks.1')
    num_tokens = len(diffusion_ts) * len(feat_names)
    d_model = int(CLS_TOKEN_DIM)

    diffusion = StudentDiffusionWrapper(
        RGB_DIFFUSION_TEACHER_CHECKPOINT,
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
    fp = open_memmap(str(out_npy), mode='w+', dtype=np.float32, shape=out_shape)

    w = int(PATCH_WINDOW_SIZE)
    bs = max(1, int(args.batch_size))
    th = tw = int(HR_TEACHER_INPUT_SIZE)

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
            rp = _crop_hr_strict_hwc(rgb_vol, row, col, w, rh, rw)
            if rk:
                rp = _apply_rot_k(rp, rk)
            t = np.transpose(rp, (2, 0, 1)).astype(np.float32, copy=False)
            tensors.append(t)
        rgb = torch.from_numpy(np.stack(tensors, axis=0)).to(device)
        rgb = F.interpolate(rgb, size=(th, tw), mode='bilinear', align_corners=False)
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
        diffusion_teacher_checkpoint=str(RGB_DIFFUSION_TEACHER_CHECKPOINT),
    )
    meta['split'] = split
    meta['strict_hr'] = True
    meta['hr_teacher_input_size'] = int(HR_TEACHER_INPUT_SIZE)
    meta['rgb_hr_path'] = str(TRAIN_RGB_HR_PATH)
    save_meta(out_meta, meta)
    print(f'写入 {out_npy} shape={out_shape}')
    print(f'meta -> {out_meta}')


if __name__ == '__main__':
    main()
