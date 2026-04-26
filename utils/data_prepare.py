"""
从 .mat 生成「整幅归一化影像 + 像素索引」，显著小于逐 patch .npy。

输入为融合包（一次读入）：`szutree_r1.mat`（MAT v5 有单块约 4GB 上限）或由
`extract_szutree_dataset.py --export` 在超大体积时改写的同茎 `szutree_r1.npz`。
须能解析出 hsi, lidar, rgb, train, test（train/test 可为 CSR 分块）。

输出（文件名与旧版一致，见 param）：
  train_patches.npy      (H,W,HSI+LiDAR) float16
  train_rgb_patches.npy  (H,W,3) float16
  rgb_hr.npy             (H_hr,W_hr,3) float16  与 LR 对齐的 HR 裁切、归一化（严格视野蒸馏用）
  rgb_hr.meta.json       rh/rw、LR/HR 形状、严格视野 HR patch 尺寸等
  train_labels.npy       (N,3) int32: [原始标签, row, col]
  test_labels.npy        (M,3) int32
  label_shift.npy        标量 int64，训练时与 pipeline/data 内平移一致

旋转增强改在训练时在线做（param.TRAIN_ROT_AUGMENT_FACTOR）。

空间对齐：推荐由 `extract_szutree_dataset.py` 在导出时完成，本脚本默认**仅校验** hsi/lidar/rgb
与 train 标签图同 (H,W)，不静默重采样。若需兼容旧数据并允许在 data_prepare 中重采样影像，
请设置: MMDIFF_DATA_PREPARE_ALLOW_RESIZE=1

**输出目录**由 `param.DATA_PREPARE_INPUT_MAT` 路径子串含 `szu` / `houston` 决定 `DATA_DIR`（见
`param.py`）。`DATA_PREPARE_INPUT_MAT` 可**始终**指向约定文件名如 `szutree_r1.mat`：若大体积
导出只生成了同茎的 `szutree_r1.npz`，本脚本会**自动**改读该 `.npz`，**无需**改 `param`。
训练阶段只读 `param` 下 `.../prepared/*.npy`，与 .mat/.npz 无关。

用法：python data_prepare.py
"""
from __future__ import annotations

import os
import sys
from os import makedirs
from pathlib import Path
import json
import scipy.io as sio
from scipy import sparse
from scipy.ndimage import zoom
import numpy as np

from param import (
    DATA_DIR,
    DATA_PREPARE_INPUT_MAT,
    HSI_CHANNELS,
    LABEL_SHIFT_PATH,
    LIDAR_CHANNELS,
    NUM_CLASSES,
    PATCH_WINDOW_SIZE,
    RGB_CHANNELS,
    RGB_HR_META_PATH,
    TEST_LABELS_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_HR_PATH,
    TRAIN_RGB_PATCHES_PATH,
)

SAVE_NORMALIZED_FEATURES = False
PATCH_DTYPE = np.float16

makedirs(DATA_DIR, exist_ok=True)


def _resolve_fusion_input_path() -> str:
    """
    按 param 配置路径读取；若文件不存在，则尝试同目录同主文件名的 .mat <-> .npz。
    这样大导出只落 .npz 时不必改 `DATA_PREPARE_INPUT_MAT`。
    """
    p = Path(DATA_PREPARE_INPUT_MAT)
    if p.is_file():
        return str(p.resolve())
    suf = p.suffix.lower()
    if suf == '.mat':
        alt = p.with_suffix('.npz')
    elif suf == '.npz':
        alt = p.with_suffix('.mat')
    else:
        alt = p
    if alt != p and alt.is_file():
        print(
            f'data_prepare: 未找到 {p}，改用 {alt}',
            file=sys.stderr,
        )
        return str(alt.resolve())
    msg = f'未找到融合包: {p}'
    if alt != p:
        msg += f' 与 {alt}'
    raise FileNotFoundError(msg)


def load_fusion_data(path: str) -> dict:
    """
    与 extract_szutree 导出一致：.mat 用 loadmat；.npz 为 hsi/lidar/rgb
    与 train、test（稠密 或 CSR 分块）。
    """
    p = path.lower()
    if p.endswith('.npz'):
        z = np.load(path, allow_pickle=False)
        out: dict = {
            'hsi': z['hsi'],
            'lidar': z['lidar'],
            'rgb': z['rgb'],
        }
        if 'train_data' in z.files:
            th, tw = int(z['train_shape'][0]), int(z['train_shape'][1])
            out['train'] = sparse.csr_matrix(
                (z['train_data'], z['train_indices'], z['train_indptr']),
                shape=(th, tw),
            )
            t2h, t2w = int(z['test_shape'][0]), int(z['test_shape'][1])
            out['test'] = sparse.csr_matrix(
                (z['test_data'], z['test_indices'], z['test_indptr']),
                shape=(t2h, t2w),
            )
        else:
            out['train'] = z['train']
            out['test'] = z['test']
        return out
    return sio.loadmat(path)


def _allow_data_prepare_resize() -> bool:
    return (os.environ.get('MMDIFF_DATA_PREPARE_ALLOW_RESIZE') or '').strip().lower() in (
        '1',
        'true',
        'yes',
    )


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


def ensure_hsi_channel_dim(hsi):
    if hsi.ndim != 3:
        raise ValueError(f'Unexpected HSI shape: {hsi.shape}')

    if hsi.shape[-1] == HSI_CHANNELS:
        return hsi
    if hsi.shape[0] == HSI_CHANNELS:
        return np.transpose(hsi, (1, 2, 0))
    raise ValueError(f'Expected {HSI_CHANNELS} HSI channels, but got shape {hsi.shape}')


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


def _resize_spatial_hwc(vol_hwc: np.ndarray, out_h: int, out_w: int, order: int = 1) -> np.ndarray:
    """
    将 (H,W,C) 影像重采样到 (out_h,out_w)。标签栅格不改动，只用于对齐 HSI/LiDAR/RGB。
    order=1 双线性（按轴分离），适合多通道强度图。
    """
    in_h, in_w, c = vol_hwc.shape[0], vol_hwc.shape[1], vol_hwc.shape[2]
    if in_h == out_h and in_w == out_w:
        return vol_hwc
    z = np.ascontiguousarray(vol_hwc.astype(np.float32, copy=False))
    zh = out_h / float(in_h)
    zw = out_w / float(in_w)
    y = zoom(z, (zh, zw, 1.0), order=order, mode='nearest')
    y = y[:out_h, :out_w, :]
    if y.shape[0] < out_h or y.shape[1] < out_w or y.shape[2] != c:
        out = np.zeros((out_h, out_w, c), dtype=np.float32)
        h0, w0 = min(y.shape[0], out_h), min(y.shape[1], out_w)
        out[:h0, :w0, :] = y[:h0, :w0, :].astype(np.float32, copy=False)
        y = out
    return y.astype(np.float32, copy=False)


def _align_hsi_lidar_to_label_grid(
    hsi: np.ndarray,
    lidar: np.ndarray,
    ref_h: int,
    ref_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    if hsi.shape[0] != lidar.shape[0] or hsi.shape[1] != lidar.shape[1]:
        print(
            f'[warn] HSI 空间 {hsi.shape[:2]} 与 LiDAR 空间 {lidar.shape[:2]} 不一致，'
            f'将分别重采样到标签栅格 ({ref_h},{ref_w})。',
        )
    out_hsi = hsi
    if (out_hsi.shape[0], out_hsi.shape[1]) != (ref_h, ref_w):
        print(
            f'[data_prepare] 将 HSI 从 {out_hsi.shape[:2]} 重采样到标签尺寸 ({ref_h},{ref_w})',
        )
        out_hsi = _resize_spatial_hwc(out_hsi, ref_h, ref_w, order=1)
    out_lidar = lidar
    if (out_lidar.shape[0], out_lidar.shape[1]) != (ref_h, ref_w):
        print(
            f'[data_prepare] 将 LiDAR 从 {out_lidar.shape[:2]} 重采样到标签尺寸 ({ref_h},{ref_w})',
        )
        out_lidar = _resize_spatial_hwc(out_lidar, ref_h, ref_w, order=1)
    return out_hsi, out_lidar


def _align_rgb_to_lr_for_hr(
    rgb_raw: np.ndarray,
    h_lr: int,
    w_lr: int,
) -> tuple[np.ndarray, int, int, int, int, np.ndarray]:
    """
    在 LR 为 (h_lr,w_lr) 时准备 HR 裁切块与下采样 LR，与未改前逻辑一致 but 在 RGB
    小于 LR 时先上采样、非整比时先重采样，使 block mean 可执行。
    返回: rgb 下采样 (h_lr,w_lr,3), rh, rw, h_crop, w_crop, rgb_hr_cropped
    """
    hh, ww, c = int(rgb_raw.shape[0]), int(rgb_raw.shape[1]), int(rgb_raw.shape[2])
    r = np.ascontiguousarray(rgb_raw.astype(np.float32, copy=False))

    if hh < h_lr or ww < w_lr:
        th = max(hh, h_lr)
        tw = max(ww, w_lr)
        print(f'[data_prepare] RGB {hh}x{ww} 小于 LR {h_lr}x{w_lr}，上采样到 {th}x{tw}')
        r = _resize_spatial_hwc(r, th, tw, order=1)
        hh, ww = th, tw

    rh = max(1, hh // h_lr)
    rw = max(1, ww // w_lr)
    h_crop = h_lr * rh
    w_crop = w_lr * rw

    if h_crop > hh or w_crop > ww:
        th = max(hh, h_crop)
        tw = max(ww, w_crop)
        print(f'[data_prepare] 将 RGB 重采样以覆盖 HR 块 {h_crop}x{w_crop}（自 {hh}x{ww}）')
        r = _resize_spatial_hwc(r, th, tw, order=1)

    r = r[:h_crop, :w_crop, :]
    rgb_hr_cropped = r
    rh = h_crop // h_lr
    rw = w_crop // w_lr
    rgb_lr = downsample_rgb_to_match(rgb_hr_cropped, target_h=h_lr, target_w=w_lr)
    return rgb_lr, int(rh), int(rw), int(h_crop), int(w_crop), rgb_hr_cropped


def main():
    data = load_fusion_data(_resolve_fusion_input_path())
    allow_resize = _allow_data_prepare_resize()
    if not allow_resize:
        print('data_prepare: 严格模式（默认）：影像须已与标签同尺寸。设置 MMDIFF_DATA_PREPARE_ALLOW_RESIZE=1 可恢复旧版重采样。')

    train = ensure_label_array(data['train'], 'train')
    test = ensure_label_array(data['test'], 'test')
    if train.shape != test.shape:
        raise ValueError(
            f'train 与 test 标签图空间尺寸须一致: train.shape={train.shape} test.shape={test.shape}'
        )
    ref_h, ref_w = int(train.shape[0]), int(train.shape[1])

    hsi = ensure_hsi_channel_dim(data['hsi'])
    lidar = ensure_lidar_channel_dim(data['lidar'])
    if allow_resize:
        hsi, lidar = _align_hsi_lidar_to_label_grid(hsi, lidar, ref_h, ref_w)
    else:
        if (hsi.shape[0], hsi.shape[1]) != (ref_h, ref_w) or (lidar.shape[0], lidar.shape[1]) != (
            ref_h,
            ref_w,
        ):
            raise ValueError(
                f'严格模式: HSI {hsi.shape[:2]} 与 LiDAR {lidar.shape[:2]} 须与标签 {ref_h}x{ref_w} 一致；'
                f'请先用 extract_szutree_dataset 导出，或设 MMDIFF_DATA_PREPARE_ALLOW_RESIZE=1'
            )

    rgb_raw = ensure_rgb_channel_dim(data['rgb'])
    h_lr, w_lr = ref_h, ref_w
    if not allow_resize and (rgb_raw.shape[0], rgb_raw.shape[1]) != (ref_h, ref_w):
        raise ValueError(
            f'严格模式: RGB(LR) {rgb_raw.shape[:2]} 须与标签 {ref_h}x{ref_w} 一致；'
            f'或设 MMDIFF_DATA_PREPARE_ALLOW_RESIZE=1 允许在 data_prepare 中重采样'
        )
    rgb, rh, rw, h_crop, w_crop, rgb_hr_cropped = _align_rgb_to_lr_for_hr(
        rgb_raw, h_lr, w_lr
    )

    feats = np.concatenate([hsi, lidar], axis=2)

    tc = HSI_CHANNELS + LIDAR_CHANNELS
    if feats.shape[2] != tc:
        raise ValueError(f'Expected {tc} total channels, but got shape {feats.shape}')

    for e in data.keys():
        if e in {'hsi', 'lidar', 'rgb', 'test', 'train'}:
            print(f'the shape of {e} is {data[e].shape}')
    print('Normalizing features...')
    feats_norm = normalize_features(feats)
    print('Normalizing RGB features...')
    rgb_norm = normalize_features(rgb)
    print('Normalizing HR RGB (aligned crop, strict-view distill)...')
    rgb_hr_norm = normalize_features(rgb_hr_cropped)

    print('Building train/test pixel indices...')
    train_idx = build_sorted_index(train)
    test_idx = build_sorted_index(test)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError('Empty train or test index')

    label_shift = int(min(int(train_idx[:, 0].min()), int(test_idx[:, 0].min())))
    g = np.array(label_shift, dtype=np.int64)

    print(f'Saving train_patches -> {TRAIN_PATCHES_PATH} ({PATCH_DTYPE})...')
    np.save(TRAIN_PATCHES_PATH, feats_norm.astype(PATCH_DTYPE))
    print(f'Saving train_rgb_patches -> {TRAIN_RGB_PATCHES_PATH} ({PATCH_DTYPE})...')
    np.save(TRAIN_RGB_PATCHES_PATH, rgb_norm.astype(PATCH_DTYPE))
    print(f'Saving rgb_hr -> {TRAIN_RGB_HR_PATH} ({PATCH_DTYPE})...')
    np.save(TRAIN_RGB_HR_PATH, rgb_hr_norm.astype(PATCH_DTYPE))
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
    RGB_HR_META_PATH.write_text(
        json.dumps(hr_meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'Saving rgb_hr meta -> {RGB_HR_META_PATH}')
    print(f'Saving train_labels -> {TRAIN_LABELS_PATH} (N={train_idx.shape[0]})...')
    np.save(TRAIN_LABELS_PATH, train_idx.astype(np.int32))
    print(f'Saving test_labels -> {TEST_LABELS_PATH} (M={test_idx.shape[0]})...')
    np.save(TEST_LABELS_PATH, test_idx.astype(np.int32))
    print(f'Saving label_shift={label_shift} -> {LABEL_SHIFT_PATH}')
    np.save(LABEL_SHIFT_PATH, g)

    if SAVE_NORMALIZED_FEATURES:
        out_dir = DATA_DIR
        print(f'Saving extra hsi/lidar slices to {out_dir}...')
        np.save(out_dir / 'hsi_only.npy', feats_norm[:, :, :HSI_CHANNELS])
        np.save(out_dir / 'lidar_only.npy', feats_norm[:, :, HSI_CHANNELS:])
        np.save(out_dir / 'rgb_only.npy', rgb_norm)

    print('Done.')
    print(
        f'  PATCH_WINDOW_SIZE={PATCH_WINDOW_SIZE}（与 param 一致）；训练时旋转见 param.TRAIN_ROT_AUGMENT_FACTOR'
    )
    print(
        f'  标签平移量 label_shift={label_shift}，平移后类别应在 0..{NUM_CLASSES - 1}'
    )


if __name__ == '__main__':
    main()
