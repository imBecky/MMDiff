from __future__ import annotations

import json
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
    RGB_HR_META_PATH,
    TEST_LABELS_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_HR_PATH,
    TRAIN_RGB_PATCHES_PATH,
    TRAIN_ROT_AUGMENT_FACTOR,
    USE_RGB_PATCHES,
    USE_SUPCON,
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
    if USE_RGB_PATCHES:
        if not TRAIN_RGB_HR_PATH.is_file():
            raise FileNotFoundError(
                f'启用 rgb 需要 {TRAIN_RGB_HR_PATH}，请运行 data_prepare 生成 rgb_hr.npy'
            )
        if not RGB_HR_META_PATH.is_file():
            raise FileNotFoundError(
                f'启用 rgb 需要 {RGB_HR_META_PATH}，请运行 data_prepare 生成 rgb_hr.meta.json'
            )


def load_rgb_hr_meta() -> dict:
    """读取 rgb_hr.meta.json（rh/rw、LR/HR 形状、严格视野块尺寸）。"""
    if not RGB_HR_META_PATH.is_file():
        raise FileNotFoundError(f'缺少 {RGB_HR_META_PATH}，请先运行 data_prepare.py')
    with open(RGB_HR_META_PATH, encoding='utf-8') as f:
        return json.load(f)


def load_rgb_hr_volume():
    """mmap 整幅归一化 HR RGB（与 train_rgb_patches 栅格对齐的裁切区域）。"""
    if not TRAIN_RGB_HR_PATH.is_file():
        raise FileNotFoundError(f'缺少 {TRAIN_RGB_HR_PATH}，请先运行 data_prepare.py')
    return np.load(TRAIN_RGB_HR_PATH, mmap_mode='r')


def _crop_hr_strict_hwc(
    vol_hr_hwc: np.ndarray,
    row_lr: int,
    col_lr: int,
    window_lr: int,
    rh: int,
    rw: int,
) -> np.ndarray:
    """
    与 LR patch (row_lr,col_lr,window_lr) 空间对齐的 HR 裁块，形状
    (window_lr*rh, window_lr*rw, C)。边界零填充与 _crop_patch_hwc 一致。
    """
    m = window_lr // 2
    h, w, c = vol_hr_hwc.shape
    r0, r1 = row_lr - m, row_lr + m + 1
    c0, c1 = col_lr - m, col_lr + m + 1
    hr_r0, hr_r1 = r0 * rh, r1 * rh
    hr_c0, hr_c1 = c0 * rw, c1 * rw
    pad_top = max(0, -hr_r0)
    pad_bottom = max(0, hr_r1 - h)
    pad_left = max(0, -hr_c0)
    pad_right = max(0, hr_c1 - w)
    hr_r0c, hr_r1c = max(0, hr_r0), min(h, hr_r1)
    hr_c0c, hr_c1c = max(0, hr_c0), min(w, hr_c1)
    patch = vol_hr_hwc[hr_r0c:hr_r1c, hr_c0c:hr_c1c, :].astype(np.float32, copy=False)
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = np.pad(
            patch,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0.0,
        )
    exp_h, exp_w = window_lr * rh, window_lr * rw
    if patch.shape[0] != exp_h or patch.shape[1] != exp_w:
        raise ValueError(
            f'HR strict crop 形状 {patch.shape[:2]} 与期望 ({exp_h},{exp_w}) 不一致'
        )
    return patch


def hr_crop_downsample_to_lr_patch(
    hr_patch_hwc: np.ndarray,
    rh: int,
    rw: int,
    window_lr: int,
) -> np.ndarray:
    """HR 严格视野块按 rh×rw 非重叠块均值下采样为 (window_lr, window_lr, C)。"""
    h, w, c = hr_patch_hwc.shape
    exp_h, exp_w = window_lr * rh, window_lr * rw
    if h != exp_h or w != exp_w:
        raise ValueError(f'HR patch 形状 {h}x{w} 与期望 {exp_h}x{exp_w} 不一致')
    return hr_patch_hwc.reshape(window_lr, rh, window_lr, rw, c).mean(axis=(1, 3))


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


def _tensorize_view_from_crops(
    fp_hwc: np.ndarray,
    rgb_hwc: Optional[np.ndarray],
    *,
    rot_factor: int,
    training: bool,
):
    """同一空间 patch 上独立随机旋转一次，得到单视图张量。"""
    rk = _random_rot_k(rot_factor) if training and rot_factor > 1 else 0
    fp = _apply_rot_k(fp_hwc, rk)
    hsi = _patch_array_to_float32(np.transpose(fp[:, :, :HSI_CHANNELS], (2, 0, 1)))
    lidar = _patch_array_to_float32(
        np.transpose(fp[:, :, HSI_CHANNELS : HSI_CHANNELS + LIDAR_CHANNELS], (2, 0, 1))
    )
    if rgb_hwc is not None:
        rp = _apply_rot_k(rgb_hwc, rk)
        rgb = _patch_array_to_float32(np.transpose(rp, (2, 0, 1)))
        return torch.from_numpy(hsi), torch.from_numpy(lidar), torch.from_numpy(rgb), int(rk)
    return torch.from_numpy(hsi), torch.from_numpy(lidar), int(rk)


class PatchDataset(torch.utils.data.Dataset):
    """
    feats: (H,W,C) HSI+LiDAR；rgb: (H,W,3) 或 None。
    indices: (N,3) [label, row, col]，label 已为 0..NUM_CLASSES-1。
    global_row_indices: 与 indices 等长，第 i 条样本在对应 labels 表中的行号（train_labels.npy 或 test_labels.npy），
用于 teacher 特征缓存与可复现噪声；若省略则沿用 __getitem__ 的局部 index（兼容旧行为）。
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
        supcon_dual_view: bool = False,
        global_row_indices: Optional[np.ndarray] = None,
        rgb_hr_vol: Optional[np.ndarray] = None,
        hr_rh: int = 1,
        hr_rw: int = 1,
        rgb_strict_view: bool = False,
    ):
        super().__init__()
        self.feats = feats_vol
        self.rgb = rgb_vol
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window_size = int(window_size)
        self.training = training
        self.rot_factor = int(rot_factor)
        self.supcon_dual_view = bool(supcon_dual_view)
        self.rgb_hr_vol = rgb_hr_vol
        self.hr_rh = int(hr_rh)
        self.hr_rw = int(hr_rw)
        self.rgb_strict_view = bool(rgb_strict_view)
        if self.rgb_strict_view:
            if self.rgb_hr_vol is None:
                raise ValueError('rgb_strict_view=True 时需要 rgb_hr_vol（rgb_hr.npy mmap）')
            if self.hr_rh <= 0 or self.hr_rw <= 0:
                raise ValueError('hr_rh/hr_rw 须为正整数')
            if self.hr_rh != self.hr_rw:
                raise ValueError(
                    '严格视野 HR 旋转与 teacher 对齐需 rh==rw（当前 HR 块非正方形）'
                )
        if global_row_indices is not None:
            gr = np.asarray(global_row_indices, dtype=np.int64).reshape(-1)
            if gr.shape[0] != len(self.indices):
                raise ValueError(
                    f'global_row_indices 长度 {gr.shape[0]} 与 indices 行数 {len(self.indices)} 不一致'
                )
            self.global_row_indices = gr
        else:
            self.global_row_indices = None

    def _rgb_patch_base_hwc(self, row: int, col: int, w: int) -> np.ndarray:
        """未旋转的 RGB patch（LR 下采样栅格；strict 时由 HR 块均值得到）。"""
        if self.rgb_strict_view and self.rgb_hr_vol is not None:
            hp = _crop_hr_strict_hwc(self.rgb_hr_vol, row, col, w, self.hr_rh, self.hr_rw)
            return hr_crop_downsample_to_lr_patch(hp, self.hr_rh, self.hr_rw, w)
        if self.rgb is None:
            raise RuntimeError('需要 rgb_vol 或 rgb_hr_vol+strict')
        return _crop_patch_hwc(self.rgb, row, col, w)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        lab = int(self.indices[index, 0])
        row = int(self.indices[index, 1])
        col = int(self.indices[index, 2])
        w = self.window_size
        fp = _crop_patch_hwc(self.feats, row, col, w)
        global_row = int(self.global_row_indices[index]) if self.global_row_indices is not None else int(index)
        lab_t = torch.tensor(lab, dtype=torch.long)
        gr_t = torch.tensor(global_row, dtype=torch.long)

        if self.training and self.supcon_dual_view:
            if self.rgb is not None or self.rgb_strict_view:
                rp = self._rgb_patch_base_hwc(row, col, w)
                h1, l1, r1, rk1 = _tensorize_view_from_crops(
                    fp, rp, rot_factor=self.rot_factor, training=self.training,
                )
                h2, l2, r2, rk2 = _tensorize_view_from_crops(
                    fp, rp, rot_factor=self.rot_factor, training=self.training,
                )
                rk1_t = torch.tensor(rk1, dtype=torch.long)
                rk2_t = torch.tensor(rk2, dtype=torch.long)
                return h1, l1, r1, h2, l2, r2, lab_t, gr_t, rk1_t, rk2_t
            h1, l1, rk1 = _tensorize_view_from_crops(
                fp, None, rot_factor=self.rot_factor, training=self.training,
            )
            h2, l2, rk2 = _tensorize_view_from_crops(
                fp, None, rot_factor=self.rot_factor, training=self.training,
            )
            rk1_t = torch.tensor(rk1, dtype=torch.long)
            rk2_t = torch.tensor(rk2, dtype=torch.long)
            return h1, l1, h2, l2, lab_t, gr_t, rk1_t, rk2_t

        rk = _random_rot_k(self.rot_factor) if self.training and self.rot_factor > 1 else 0
        fp = _apply_rot_k(fp, rk)
        hsi = _patch_array_to_float32(np.transpose(fp[:, :, :HSI_CHANNELS], (2, 0, 1)))
        lidar = _patch_array_to_float32(
            np.transpose(fp[:, :, HSI_CHANNELS : HSI_CHANNELS + LIDAR_CHANNELS], (2, 0, 1))
        )
        rk_t = torch.tensor(int(rk), dtype=torch.long)
        if self.rgb is not None or self.rgb_strict_view:
            rp = self._rgb_patch_base_hwc(row, col, w)
            rp = _apply_rot_k(rp, rk)
            rgb = _patch_array_to_float32(np.transpose(rp, (2, 0, 1)))
            return (
                torch.from_numpy(hsi),
                torch.from_numpy(lidar),
                torch.from_numpy(rgb),
                lab_t,
                gr_t,
                rk_t,
            )
        return (
            torch.from_numpy(hsi),
            torch.from_numpy(lidar),
            lab_t,
            gr_t,
            rk_t,
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
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    返回 (train_rows, val_rows, train_positions, val_positions)。
    train_positions[i] / val_positions[j] 为样本在原始 train_indices（即 train_labels.npy 行序）中的行号，用于 teacher 缓存与稳定噪声。
    """
    n = len(train_indices)
    all_pos = np.arange(n, dtype=np.int64)
    if val_ratio <= 0 or val_ratio >= 1.0:
        return train_indices, None, all_pos, None
    labels = train_indices[:, 0]
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_indices[train_idx], train_indices[val_idx], train_idx.astype(np.int64), val_idx.astype(np.int64)


def build_test_loader(
    feats_vol: np.ndarray,
    rgb_vol: Optional[np.ndarray],
    test_indices: np.ndarray,
    batch_size: int,
    *,
    global_row_indices: Optional[np.ndarray] = None,
    rgb_strict_view: bool = False,
    rgb_hr_vol: Optional[np.ndarray] = None,
    hr_rh: int = 1,
    hr_rw: int = 1,
) -> DataLoader:
    if global_row_indices is None:
        global_row_indices = np.arange(len(test_indices), dtype=np.int64)
    ds = PatchDataset(
        feats_vol,
        rgb_vol,
        test_indices,
        window_size=PATCH_WINDOW_SIZE,
        training=False,
        rot_factor=1,
        supcon_dual_view=False,
        global_row_indices=global_row_indices,
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
        rgb_strict_view=rgb_strict_view,
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
    train_global_rows: Optional[np.ndarray] = None,
    val_global_rows: Optional[np.ndarray] = None,
    test_global_rows: Optional[np.ndarray] = None,
    rgb_strict_view: bool = False,
    rgb_hr_vol: Optional[np.ndarray] = None,
    hr_rh: int = 1,
    hr_rw: int = 1,
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
        supcon_dual_view=USE_SUPCON,
        global_row_indices=train_global_rows,
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
        rgb_strict_view=rgb_strict_view,
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
            supcon_dual_view=False,
            global_row_indices=val_global_rows,
            rgb_hr_vol=rgb_hr_vol,
            hr_rh=hr_rh,
            hr_rw=hr_rw,
            rgb_strict_view=rgb_strict_view,
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
        tgr = test_global_rows
        if tgr is None:
            tgr = np.arange(len(test_idx), dtype=np.int64)
        test_loader = build_test_loader(
            feats_vol,
            rgb_vol,
            test_idx,
            batch_size,
            global_row_indices=tgr,
            rgb_strict_view=rgb_strict_view,
            rgb_hr_vol=rgb_hr_vol,
            hr_rh=hr_rh,
            hr_rw=hr_rw,
        )

    return train_loader, val_loader, test_loader


def batch_to_dict(batch, device, use_rgb_patches: bool, use_supcon: bool = False):
    """将 DataLoader batch 转为模型输入 dict；sample_indices 使用全局行号（与 train_labels/test_labels 对齐）。"""
    if use_supcon:
        if use_rgb_patches:
            hsi1, lidar1, rgb1, hsi2, lidar2, rgb2, labels, gr, rk1, rk2 = batch
            hsi = torch.cat([hsi1, hsi2], dim=0).to(device=device, dtype=torch.float32)
            lidar = torch.cat([lidar1, lidar2], dim=0).to(device=device, dtype=torch.float32)
            rgb = torch.cat([rgb1, rgb2], dim=0).to(device=device, dtype=torch.float32)
            labels = labels.to(device).long()
            labels = torch.cat([labels, labels], dim=0)
            gr = gr.to(device=device, dtype=torch.long)
            gr = torch.cat([gr, gr], dim=0)
            rk = torch.cat(
                [rk1.to(device=device, dtype=torch.long), rk2.to(device=device, dtype=torch.long)],
                dim=0,
            )
            return {
                'hsi': hsi,
                'lidar': lidar,
                'rgb': rgb,
                'sample_indices': gr,
                'global_row': gr,
                'rot_k': rk,
            }, labels
        hsi1, lidar1, hsi2, lidar2, labels, gr, rk1, rk2 = batch
        hsi = torch.cat([hsi1, hsi2], dim=0).to(device=device, dtype=torch.float32)
        lidar = torch.cat([lidar1, lidar2], dim=0).to(device=device, dtype=torch.float32)
        labels = labels.to(device).long()
        labels = torch.cat([labels, labels], dim=0)
        gr = gr.to(device=device, dtype=torch.long)
        gr = torch.cat([gr, gr], dim=0)
        rk = torch.cat(
            [rk1.to(device=device, dtype=torch.long), rk2.to(device=device, dtype=torch.long)],
            dim=0,
        )
        return {
            'hsi': hsi,
            'lidar': lidar,
            'sample_indices': gr,
            'global_row': gr,
            'rot_k': rk,
        }, labels
    if use_rgb_patches:
        hsi, lidar, rgb, labels, gr, rk = batch
        hsi = hsi.to(device=device, dtype=torch.float32)
        lidar = lidar.to(device=device, dtype=torch.float32)
        rgb = rgb.to(device=device, dtype=torch.float32)
        labels = labels.to(device).long()
        gr = gr.to(device=device, dtype=torch.long)
        rk = rk.to(device=device, dtype=torch.long)
        return {
            'hsi': hsi,
            'lidar': lidar,
            'rgb': rgb,
            'sample_indices': gr,
            'global_row': gr,
            'rot_k': rk,
        }, labels
    hsi, lidar, labels, gr, rk = batch
    hsi = hsi.to(device=device, dtype=torch.float32)
    lidar = lidar.to(device=device, dtype=torch.float32)
    labels = labels.to(device).long()
    gr = gr.to(device=device, dtype=torch.long)
    rk = rk.to(device=device, dtype=torch.long)
    return {
        'hsi': hsi,
        'lidar': lidar,
        'sample_indices': gr,
        'global_row': gr,
        'rot_k': rk,
    }, labels
