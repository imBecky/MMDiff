"""
从 houston2018.mat 生成「整幅归一化影像 + 像素索引」，显著小于逐 patch .npy。

输出（文件名与旧版一致，见 param）：
  train_patches.npy      (H,W,HSI+LiDAR) float16
  train_rgb_patches.npy  (H,W,3) float16
  train_labels.npy       (N,3) int32: [原始标签, row, col]
  test_labels.npy        (M,3) int32
  label_shift.npy        标量 int64，训练时与 pipeline/data 内平移一致

旋转增强改在训练时在线做（param.TRAIN_ROT_AUGMENT_FACTOR）。

用法：python data_prepare.py
"""
from __future__ import annotations

from os import makedirs
import scipy.io as sio
from scipy import sparse
import numpy as np

from param import (
    DATA_DIR,
    HSI_CHANNELS,
    LABEL_SHIFT_PATH,
    LIDAR_CHANNELS,
    NUM_CLASSES,
    PATCH_WINDOW_SIZE,
    RGB_CHANNELS,
    TEST_LABELS_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_PATCHES_PATH,
)

DATA_PATH = '../../autodl-fs/houston2018/houston2018.mat'
SAVE_NORMALIZED_FEATURES = False
PATCH_DTYPE = np.float16

makedirs(DATA_DIR, exist_ok=True)


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


def main():
    data = sio.loadmat(DATA_PATH)

    hsi = ensure_hsi_channel_dim(data['hsi'])
    lidar = ensure_lidar_channel_dim(data['lidar'])
    rgb = ensure_rgb_channel_dim(data['rgb'])
    rgb = downsample_rgb_to_match(rgb, target_h=hsi.shape[0], target_w=hsi.shape[1])

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

    train = ensure_label_array(data['train'], 'train')
    test = ensure_label_array(data['test'], 'test')

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
