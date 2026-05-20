"""
从融合 .mat 或 SZUTree 分文件目录生成「整幅归一化影像 + 像素索引」，显著小于逐 patch .npy。

两种方式：
1）**融合包**：单个 .mat（含键 hsi, lidar, rgb, train, test），用法见下方 `DATA_PATH`。
2）**SZUTree 多分文件**：同目录放置 `HSI.mat`（HDF5 / 98 波段）、`LiDAR.mat`、`RGB.mat`、`label.mat`，与
   `utils/extract_szutree_dataset.py` 约定一致（标签 2×2 众数下采样、按类抽样 train/test 等）。
   ```
   python data_prepare.py --szutree-dir "D:\\path\\to\\SZUTreeData_R1_2.0"
   ```
   默认写出到 `<szutree-dir>/prepared/`；也可用 `--prepared-dir` 指定。
   （可选：`python utils/extract_szutree_dataset.py --export --r1-dir ...` 先生成融合 .mat，
   再改 `DATA_PATH` 跑一次本脚本——与 --szutree-dir 二选一即可。）

单 .mat 时 HSI 通道数：**路径名含 `szutree`** 则按 98 通道； Houston 默认 48。若你把 SZUTree 导出到不带该字样的路径，
请设置 **`MMDIFF_HSI_CHANNELS=98`**。

输出（文件名与旧版 / param 一致；其中 rgb_hr* 仅为历史命名）：
  train_patches.npy      (H,W,HSI+LiDAR) float16
  train_rgb_patches.npy  (H,W,3) float16  与 HSI 同格的 LR RGB
  rgb_hr.npy             (**文件名历史遗留**) 严格视野用的「对齐裁切 RGB」整幅
  rgb_hr.meta.json       rh/rw、LR 与裁切记空间、`strict_*` 块像素尺寸等
  train_labels.npy       (N,3) int32: [原始标签, row, col]
  test_labels.npy        (M,3) int32
  label_shift.npy        标量 int64

旋转增强改在训练时在线做（param.TRAIN_ROT_AUGMENT_FACTOR）。

用法：
  python data_prepare.py
  python data_prepare.py --szutree-dir /path/to/folder/with/mats
"""
from __future__ import annotations

import argparse
import json
import os
import scipy.io as sio
from os import makedirs
from pathlib import Path
from scipy import sparse
import numpy as np

from param import (
    LIDAR_CHANNELS,
    NUM_CLASSES,
    PATCH_WINDOW_SIZE,
    RGB_CHANNELS,
)

DATA_PATH = '../../autodl-fs/houston2018/houston2018.mat'


def _infer_hsi_channels(single_mat_path: str) -> int:
    e = os.environ.get('MMDIFF_HSI_CHANNELS', '').strip()
    if e.isdigit():
        return int(e)
    lp = single_mat_path.lower()
    if 'szutree' in lp:
        return 98
    return 48


def _infer_data_dir(mat_path_str: str) -> Path:
    lp = mat_path_str.lower()
    if 'szutree' in lp:
        return Path('../../autodl-fs/szutree/prepared')
    return Path('../../autodl-fs/houston2018/prepared')


def _build_output_paths(data_dir: Path) -> dict[str, Path]:
    d = Path(data_dir)
    return {
        'data_dir': d,
        'train_patches': d / 'train_patches.npy',
        'train_rgb': d / 'train_rgb_patches.npy',
        'train_rgb_hr': d / 'rgb_hr.npy',
        'rgb_meta': d / 'rgb_hr.meta.json',
        'train_labels': d / 'train_labels.npy',
        'test_labels': d / 'test_labels.npy',
        'label_shift': d / 'label_shift.npy',
    }


SAVE_NORMALIZED_FEATURES = False
PATCH_DTYPE = np.float16


def build_sorted_index(y):
    coords = np.argwhere(y > 0)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    labels = y[coords[:, 0], coords[:, 1]].astype(np.int32, copy=False)
    height, width = y.shape
    order_key = (
        labels.astype(np.int64) * (height * width)
        + coords[:, 0].astype(np.int64) * width
        + coords[:, 1].astype(np.int64)
    )
    order = np.argsort(order_key)

    return np.column_stack([labels[order], coords[order]]).astype(np.int32, copy=False)


def normalize_features(feats):
    feats = feats.astype(np.float32, copy=False)
    feats_min = feats.min(axis=(0, 1), keepdims=True)
    shifted = feats - feats_min
    feats_max = shifted.max(axis=(0, 1), keepdims=True)

    return np.divide(
        shifted,
        feats_max,
        out=np.zeros_like(shifted, dtype=np.float32),
        where=feats_max != 0,
    )


def ensure_hsi_channel_dim(hsi, hsi_channels: int):
    if hsi.ndim != 3:
        raise ValueError(f'Unexpected HSI shape: {hsi.shape}')

    if hsi.shape[-1] == hsi_channels:
        return hsi
    if hsi.shape[0] == hsi_channels:
        return np.transpose(hsi, (1, 2, 0))
    raise ValueError(f'Expected {hsi_channels} HSI channels, but got shape {hsi.shape}')


def ensure_lidar_channel_dim(lidar):
    if lidar.ndim == 2:
        lidar = np.expand_dims(lidar, axis=2)
    elif lidar.ndim != 3:
        raise ValueError(f'Unexpected LiDAR shape: {lidar.shape}')

    if lidar.shape[-1] == LIDAR_CHANNELS:
        return lidar
    if lidar.shape[0] == LIDAR_CHANNELS:
        return np.transpose(lidar, (1, 2, 0))
    raise ValueError(f'Expected {LIDAR_CHANNELS} LiDAR channel(s), but got shape {lidar.shape}')


def ensure_rgb_channel_dim(rgb):
    if rgb.ndim != 3:
        raise ValueError(f'Unexpected RGB shape: {rgb.shape}')
    if rgb.shape[-1] == RGB_CHANNELS:
        return rgb
    if rgb.shape[0] == RGB_CHANNELS:
        return np.transpose(rgb, (1, 2, 0))
    raise ValueError(f'Expected {RGB_CHANNELS} RGB channels, but got shape {rgb.shape}')


def downsample_rgb_to_match(rgb_hwc, target_h, target_w):
    h, w, c = rgb_hwc.shape
    if c != RGB_CHANNELS:
        raise ValueError(f'Unexpected rgb channel count: {c}')
    if h == target_h and w == target_w:
        return rgb_hwc
    rh = h // target_h
    rw = w // target_w
    h_crop = target_h * rh
    w_crop = target_w * rw
    rgb_cropped = rgb_hwc[:h_crop, :w_crop, :]
    return rgb_cropped.reshape(target_h, rh, target_w, rw, c).mean(axis=(1, 3))


def ensure_label_array(labels, name):
    if sparse.issparse(labels):
        print(f'{name} is sparse, converting to dense array.')
        labels = labels.toarray()
    else:
        labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError(f'Expected {name} labels to be 2D, but got shape {labels.shape}')

    return labels


def run_prepare_pipeline(
    data,
    *,
    hsi_channels: int,
    prepared_dir: Path,
    log_shapes_keys: tuple[str, ...],
) -> None:
    paths = _build_output_paths(Path(prepared_dir))
    data_dir = paths['data_dir']
    makedirs(data_dir, exist_ok=True)

    hsi = ensure_hsi_channel_dim(data['hsi'], hsi_channels)
    lidar = ensure_lidar_channel_dim(data['lidar'])
    rgb_source_hwc = ensure_rgb_channel_dim(data['rgb'])
    h_lr, w_lr = int(hsi.shape[0]), int(hsi.shape[1])
    hh, ww, _ = rgb_source_hwc.shape
    rh = hh // h_lr
    rw = ww // w_lr
    h_crop = h_lr * rh
    w_crop = w_lr * rw
    rgb_aligned_crop_hwc = rgb_source_hwc[:h_crop, :w_crop, :]
    rgb_lr_hwc = downsample_rgb_to_match(rgb_source_hwc, target_h=h_lr, target_w=w_lr)

    feats = np.concatenate([hsi, lidar], axis=2)

    tc = hsi_channels + LIDAR_CHANNELS
    if feats.shape[2] != tc:
        raise ValueError(f'Expected {tc} total channels, but got shape {feats.shape}')

    for e in log_shapes_keys:
        if e in data:
            v = data[e]
            shape = getattr(v, 'shape', 'n/a')
            print(f'the shape of {e} is {shape}')
    print('Normalizing features...')
    feats_norm = normalize_features(feats)
    print('Normalizing RGB features...')
    rgb_norm = normalize_features(rgb_lr_hwc)
    print('Normalizing aligned-crop RGB for strict-view volume (saved as rgb_hr.npy, legacy filename)...')
    rgb_strict_view_norm = normalize_features(rgb_aligned_crop_hwc)

    train = ensure_label_array(data['train'], 'train')
    test = ensure_label_array(data['test'], 'test')

    print('Building train/test pixel indices...')
    train_idx = build_sorted_index(train)
    test_idx = build_sorted_index(test)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError('Empty train or test index')

    label_shift = int(min(int(train_idx[:, 0].min()), int(test_idx[:, 0].min())))
    g = np.array(label_shift, dtype=np.int64)

    train_patches_p = paths['train_patches']
    train_rgb_p = paths['train_rgb']
    train_rgb_hr_p = paths['train_rgb_hr']
    rgb_meta_p = paths['rgb_meta']
    train_labels_p = paths['train_labels']
    test_labels_p = paths['test_labels']
    label_shift_p = paths['label_shift']

    print(f'Saving train_patches -> {train_patches_p} ({PATCH_DTYPE})...')
    np.save(train_patches_p, feats_norm.astype(PATCH_DTYPE))
    print(f'Saving train_rgb_patches -> {train_rgb_p} ({PATCH_DTYPE})...')
    np.save(train_rgb_p, rgb_norm.astype(PATCH_DTYPE))
    print(
        f'Saving strict-view RGB volume -> {train_rgb_hr_p} ({PATCH_DTYPE}) '
        '(on-disk name rgb_hr.npy)'
    )
    np.save(train_rgb_hr_p, rgb_strict_view_norm.astype(PATCH_DTYPE))
    # hr_h/hr_w 键名为历史沿用：语义是「对齐到 LR×(rh,rw) 整格后的 RGB 体积」之高宽；可与 lr 相同
    hr_meta = {
        'rh': int(rh),
        'rw': int(rw),
        'lr_h': int(h_lr),
        'lr_w': int(w_lr),
        'hr_h': int(h_crop),
        'hr_w': int(w_crop),
        'patch_window_size': int(PATCH_WINDOW_SIZE),
        'strict_hr_patch_h': int(PATCH_WINDOW_SIZE * rh),
        'strict_hr_patch_w': int(PATCH_WINDOW_SIZE * rw),
    }
    rgb_meta_p.write_text(
        json.dumps(hr_meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'Saving strict-view RGB meta -> {rgb_meta_p} (legacy filename rgb_hr.meta.json)')
    print(f'Saving train_labels -> {train_labels_p} (N={train_idx.shape[0]})...')
    np.save(train_labels_p, train_idx.astype(np.int32))
    print(f'Saving test_labels -> {test_labels_p} (M={test_idx.shape[0]})...')
    np.save(test_labels_p, test_idx.astype(np.int32))
    print(f'Saving label_shift={label_shift} -> {label_shift_p}')
    np.save(label_shift_p, g)

    if SAVE_NORMALIZED_FEATURES:
        print(f'Saving extra hsi/lidar slices to {data_dir}...')
        np.save(data_dir / 'hsi_only.npy', feats_norm[:, :, :hsi_channels])
        np.save(data_dir / 'lidar_only.npy', feats_norm[:, :, hsi_channels:])
        np.save(data_dir / 'rgb_only.npy', rgb_norm)

    print('Done.')
    print(
        f'  PATCH_WINDOW_SIZE={PATCH_WINDOW_SIZE}（与 param 一致）；训练时旋转见 param.TRAIN_ROT_AUGMENT_FACTOR'
    )
    print(
        f'  标签平移量 label_shift={label_shift}，平移后类别应在 0..{NUM_CLASSES - 1}'
    )


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='mat / SZUTree 分文件 → prepared/*.npy')
    p.add_argument(
        '--szutree-dir',
        type=Path,
        default=None,
        help='SZUTree 多分文件目录（HSI.mat, LiDAR.mat, RGB.mat, label.mat）',
    )
    p.add_argument(
        '--prepared-dir',
        type=Path,
        default=None,
        help='输出 prepared 目录（SZUTree 分文件时默认 <szutree-dir>/prepared；融合 .mat 时默认按路径推断，见源码）',
    )
    p.add_argument(
        '--mat',
        type=str,
        default=None,
        help='覆盖融合 .mat 路径（默认脚本内 DATA_PATH）',
    )
    p.add_argument(
        '--train-percent-per-class',
        type=float,
        default=1.0,
        help='--szutree-dir 时每类训练像素百分比（与同目录 extract_szutree 默认一致，如 1.0 表示 1%%)',
    )
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-classes', type=int, default=20)
    p.add_argument(
        '--no-progress',
        action='store_true',
        help='--szutree-dir 读取时关闭 tqdm（由 extract_szutree 模块使用）',
    )
    return p.parse_args()


def main():
    args = _parse_cli()
    log_keys = ('hsi', 'lidar', 'rgb', 'train', 'test')

    if args.szutree_dir is not None:
        r1 = Path(args.szutree_dir).expanduser().resolve()
        if not r1.is_dir():
            raise FileNotFoundError(f'--szutree-dir 不是目录: {r1}')
        prepared = (
            Path(args.prepared_dir).expanduser().resolve()
            if args.prepared_dir is not None
            else (r1 / 'prepared')
        )
        from utils.extract_szutree_dataset import assemble_szutree_split_mats_payload

        print(f'SZUTree 分文件目录: {r1}')
        print(f'写入 prepared: {prepared}')
        fusion = assemble_szutree_split_mats_payload(
            r1,
            train_percent=float(args.train_percent_per_class),
            seed=int(args.seed),
            num_classes=int(args.num_classes),
            save_train_test_dense=False,
            show_progress=not args.no_progress,
        )
        run_prepare_pipeline(
            fusion,
            hsi_channels=98,
            prepared_dir=prepared,
            log_shapes_keys=log_keys,
        )
        print(
            '提示：训练前请确认 `param.py` 中 DATA_DIR 指向上述 prepared，且 HSI 为 98 波段（MMDIFF_HSI_CHANNELS=98）。',
        )
        return

    mat_path = str(Path(args.mat or DATA_PATH).expanduser())
    ch = _infer_hsi_channels(mat_path)
    out_dir = (
        Path(args.prepared_dir).expanduser().resolve()
        if args.prepared_dir is not None
        else _infer_data_dir(mat_path)
    )

    print(f'Fusion .mat: {mat_path}')
    print(f'HSI_CHANNELS={ch} （可用 MMDIFF_HSI_CHANNELS 覆盖；路径含 szutree 时为 98）')
    print(f'输出目录: {out_dir}')

    data = sio.loadmat(mat_path)
    run_prepare_pipeline(
        data,
        hsi_channels=ch,
        prepared_dir=out_dir,
        log_shapes_keys=log_keys,
    )


if __name__ == '__main__':
    main()
