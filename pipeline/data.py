from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from param import (
    HSI_CHANNELS,
    LABEL_SHIFT_PATH,
    LIDAR_CHANNELS,
    NUM_CLASSES,
    NUM_WORKERS,
    PATCH_WINDOW_SIZE,
    TEST_LABELS_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_PATCHES_PATH,
    TRAIN_ROT_AUGMENT_FACTOR,
    USE_RGB_PATCHES,
)


def _patch_array_to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    if x.dtype == np.float16:
        return x.astype(np.float32, copy=False)
    if x.dtype == np.float64:
        return x.astype(np.float32, copy=False)
    raise ValueError(f'patch 数组应为浮点类型，当前 dtype={x.dtype}')


def _require_prepared_data_files():
    req = [
        TRAIN_PATCHES_PATH,
        TRAIN_LABELS_PATH,
        TEST_LABELS_PATH,
        LABEL_SHIFT_PATH,
    ]
    missing = [p for p in req if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            '缺少数据文件（请先运行 data_prepare.py）：\n'
            + '\n'.join(str(p) for p in missing)
        )
    if USE_RGB_PATCHES and not TRAIN_RGB_PATCHES_PATH.is_file():
        raise FileNotFoundError(
            f'USE_RGB_PATCHES 为真但缺少 {TRAIN_RGB_PATCHES_PATH}，请运行 data_prepare 生成 train_rgb_patches.npy'
        )


def _crop_patch_hwc(vol_hwc: np.ndarray, row: int, col: int, window_size: int) -> np.ndarray:
    m = window_size // 2
    h, w, c = vol_hwc.shape
    r0, r1 = row - m, row + m + 1
    c0, c1 = col - m, col + m + 1
    pad_top = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left = max(0, -c0)
    pad_right = max(0, c1 - w)
    r0c, r1c = max(0, r0), min(h, r1)
    c0c, c1c = max(0, c0), min(w, c1)
    patch = vol_hwc[r0c:r1c, c0c:c1c, :].astype(np.float32, copy=False)
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0.0,
        )
    assert patch.shape[0] == window_size and patch.shape[1] == window_size
    return patch


def _random_rot_k(factor: int) -> int:
    if factor == 1:
        return 0
    if factor == 2:
        return 0 if np.random.randint(0, 2) == 0 else 2
    return int(np.random.randint(0, 4))


def _apply_rot_k(patch_hwc: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return patch_hwc
    return np.rot90(patch_hwc, k=k, axes=(0, 1)).copy()


class PatchDataset(torch.utils.data.Dataset):
    """
    feats: (H,W,C) HSI+LiDAR；rgb: (H,W,3) 或 None。
    indices: (N,3) [label, row, col]，label 已为 0..NUM_CLASSES-1。
    """

    def __init__(
        self,
        feats_vol: np.ndarray,
        rgb_vol: Optional[np.ndarray],
        indices: np.ndarray,
        *,
        window_size: int,
        training: bool,
        rot_factor: int = 1,
    ):
        super().__init__()
        self.feats = feats_vol
        self.rgb = rgb_vol
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window_size = int(window_size)
        self.training = training
        self.rot_factor = int(rot_factor)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        lab = int(self.indices[index, 0])
        row = int(self.indices[index, 1])
        col = int(self.indices[index, 2])
        w = self.window_size
        fp = _crop_patch_hwc(self.feats, row, col, w)
        rk = _random_rot_k(self.rot_factor) if self.training and self.rot_factor > 1 else 0
        fp = _apply_rot_k(fp, rk)
        hsi = _patch_array_to_float32(np.transpose(fp[:, :, :HSI_CHANNELS], (2, 0, 1)))
        lidar = _patch_array_to_float32(
            np.transpose(fp[:, :, HSI_CHANNELS : HSI_CHANNELS + LIDAR_CHANNELS], (2, 0, 1))
        )
        if self.rgb is not None:
            rp = _crop_patch_hwc(self.rgb, row, col, w)
            rp = _apply_rot_k(rp, rk)
            rgb = _patch_array_to_float32(np.transpose(rp, (2, 0, 1)))
            return (
                torch.from_numpy(hsi),
                torch.from_numpy(lidar),
                torch.from_numpy(rgb),
                torch.tensor(lab, dtype=torch.long),
                torch.tensor(index, dtype=torch.long),
            )
        return (
            torch.from_numpy(hsi),
            torch.from_numpy(lidar),
            torch.tensor(lab, dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def load_train_bundle():
    """
    mmap 整幅 HSI+LiDAR / RGB + 训练索引；label 已按 label_shift 平移。
    """
    _require_prepared_data_files()
    feats = np.load(TRAIN_PATCHES_PATH, mmap_mode='r')
    rgb = np.load(TRAIN_RGB_PATCHES_PATH, mmap_mode='r') if USE_RGB_PATCHES else None
    train_indices = np.load(TRAIN_LABELS_PATH).astype(np.int64, copy=True)
    label_shift = int(np.load(LABEL_SHIFT_PATH))
    train_indices[:, 0] = train_indices[:, 0] - label_shift
    if int(train_indices[:, 0].max()) >= NUM_CLASSES or int(train_indices[:, 0].min()) < 0:
        raise ValueError(
            f'训练标签越界: min={int(train_indices[:, 0].min())} max={int(train_indices[:, 0].max())} '
            f'NUM_CLASSES={NUM_CLASSES}'
        )
    return feats, rgb, train_indices, label_shift


def load_test_indices_shifted(label_shift: int) -> np.ndarray:
    if not TEST_LABELS_PATH.is_file():
        raise FileNotFoundError(f'缺少 {TEST_LABELS_PATH}')
    test_indices = np.load(TEST_LABELS_PATH).astype(np.int64, copy=True)
    test_indices[:, 0] = test_indices[:, 0] - int(label_shift)
    if int(test_indices[:, 0].max()) >= NUM_CLASSES or int(test_indices[:, 0].min()) < 0:
        raise ValueError(
            f'测试标签越界: min={int(test_indices[:, 0].min())} max={int(test_indices[:, 0].max())}'
        )
    return test_indices


def subset_train_indices_balanced(
    train_indices: np.ndarray,
    samples_per_class: int,
    seed: int,
    num_classes: int,
) -> np.ndarray:
    labels = train_indices[:, 0]
    rng = np.random.RandomState(seed)
    parts = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        n_take = min(int(samples_per_class), len(idx))
        chosen = rng.choice(idx, size=n_take, replace=False)
        parts.append(chosen)
    if not parts:
        return train_indices
    all_idx = np.concatenate(parts)
    rng.shuffle(all_idx)
    return train_indices[all_idx]


def split_train_val_indices(
    train_indices: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if val_ratio <= 0 or val_ratio >= 1.0:
        return train_indices, None
    labels = train_indices[:, 0]
    idx = np.arange(len(train_indices))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_indices[train_idx], train_indices[val_idx]


def build_test_loader(
    feats_vol: np.ndarray,
    rgb_vol: Optional[np.ndarray],
    test_indices: np.ndarray,
    batch_size: int,
) -> DataLoader:
    ds = PatchDataset(
        feats_vol,
        rgb_vol,
        test_indices,
        window_size=PATCH_WINDOW_SIZE,
        training=False,
        rot_factor=1,
    )
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )


def build_dataloaders(
    feats_vol: np.ndarray,
    rgb_vol: Optional[np.ndarray],
    tr_idx: np.ndarray,
    va_idx: Optional[np.ndarray],
    test_idx: Optional[np.ndarray],
    batch_size: int,
    *,
    defer_test: bool,
):
    pin_memory = torch.cuda.is_available()
    rot = int(TRAIN_ROT_AUGMENT_FACTOR)
    if rot not in (1, 2, 4):
        rot = 1

    train_ds = PatchDataset(
        feats_vol,
        rgb_vol,
        tr_idx,
        window_size=PATCH_WINDOW_SIZE,
        training=True,
        rot_factor=rot,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = None
    if va_idx is not None and len(va_idx) > 0:
        val_ds = PatchDataset(
            feats_vol,
            rgb_vol,
            va_idx,
            window_size=PATCH_WINDOW_SIZE,
            training=False,
            rot_factor=1,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    test_loader = None
    if not defer_test and test_idx is not None and len(test_idx) > 0:
        test_loader = build_test_loader(feats_vol, rgb_vol, test_idx, batch_size)

    return train_loader, val_loader, test_loader


def batch_to_dict(batch, device, use_rgb_patches: bool):
    if use_rgb_patches:
        hsi, lidar, rgb, labels, sample_indices = batch
        hsi = hsi.to(device=device, dtype=torch.float32)
        lidar = lidar.to(device=device, dtype=torch.float32)
        rgb = rgb.to(device=device, dtype=torch.float32)
        labels = labels.to(device).long()
        sample_indices = sample_indices.to(device=device, dtype=torch.long)
        return {'hsi': hsi, 'lidar': lidar, 'rgb': rgb, 'sample_indices': sample_indices}, labels
    hsi, lidar, labels, sample_indices = batch
    hsi = hsi.to(device=device, dtype=torch.float32)
    lidar = lidar.to(device=device, dtype=torch.float32)
    labels = labels.to(device).long()
    sample_indices = sample_indices.to(device=device, dtype=torch.long)
    return {'hsi': hsi, 'lidar': lidar, 'sample_indices': sample_indices}, labels
