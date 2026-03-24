import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from param import (
    HSI_CHANNELS,
    LIDAR_CHANNELS,
    NUM_CLASSES,
    RGB_CHANNELS,
    TEST_LABELS_PATH,
    TEST_PATCHES_PATH,
    TEST_RGB_PATCHES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_PATCHES_PATH,
    USE_RGB_PATCHES,
)


def load_data():
    train_patches = np.load(TRAIN_PATCHES_PATH)
    test_patches = np.load(TEST_PATCHES_PATH)

    train_labels = np.load(TRAIN_LABELS_PATH)
    test_labels = np.load(TEST_LABELS_PATH)

    expected_channels = HSI_CHANNELS + LIDAR_CHANNELS
    if train_patches.shape[-1] != expected_channels or test_patches.shape[-1] != expected_channels:
        raise ValueError(
            f'Expected patch channels to be {expected_channels}, '
            f'but got train={train_patches.shape[-1]}, test={test_patches.shape[-1]}'
        )

    train_hsi = np.transpose(train_patches[:, :, :, :HSI_CHANNELS], (0, 3, 1, 2))
    train_lidar = np.transpose(train_patches[:, :, :, HSI_CHANNELS:HSI_CHANNELS + LIDAR_CHANNELS], (0, 3, 1, 2))

    test_hsi = np.transpose(test_patches[:, :, :, :HSI_CHANNELS], (0, 3, 1, 2))
    test_lidar = np.transpose(test_patches[:, :, :, HSI_CHANNELS:HSI_CHANNELS + LIDAR_CHANNELS], (0, 3, 1, 2))

    train_rgb, test_rgb = None, None
    if USE_RGB_PATCHES:
        train_rgb_arr = np.load(TRAIN_RGB_PATCHES_PATH)
        test_rgb_arr = np.load(TEST_RGB_PATCHES_PATH)
        if train_rgb_arr.shape[-1] != RGB_CHANNELS or test_rgb_arr.shape[-1] != RGB_CHANNELS:
            raise ValueError(
                f'RGB patches last dim must be RGB_CHANNELS={RGB_CHANNELS}, '
                f'got train={train_rgb_arr.shape[-1]}, test={test_rgb_arr.shape[-1]}'
            )
        train_rgb = np.transpose(train_rgb_arr, (0, 3, 1, 2))
        test_rgb = np.transpose(test_rgb_arr, (0, 3, 1, 2))
        for name, a, ref in (('train', train_rgb, train_hsi), ('test', test_rgb, test_hsi)):
            if a.shape[0] != ref.shape[0]:
                raise ValueError(f'{name} RGB N mismatch: rgb={a.shape[0]} vs hsi={ref.shape[0]}')
            if a.shape[2:] != ref.shape[2:]:
                raise ValueError(f'{name} RGB spatial shape mismatch: rgb={a.shape[2:]} vs hsi={ref.shape[2:]}')

    # CrossEntropyLoss expects 0-based class indices；train/test 用同一全局偏移，避免 split 后类别编号错位
    global_min = int(min(train_labels.min(), test_labels.min()))
    train_labels = train_labels - global_min
    test_labels = test_labels - global_min
    if int(train_labels.max()) >= NUM_CLASSES or int(test_labels.max()) >= NUM_CLASSES:
        raise ValueError(
            f'Label range exceeds NUM_CLASSES={NUM_CLASSES} after shift by {global_min}: '
            f'train max={int(train_labels.max())}, test max={int(test_labels.max())}'
        )

    return train_hsi, train_lidar, train_rgb, train_labels, test_hsi, test_lidar, test_rgb, test_labels


def subset_train_balanced_per_class(
    train_hsi,
    train_lidar,
    train_rgb,
    train_labels,
    samples_per_class: int,
    seed: int,
    num_classes: int,
):
    """
    从训练集中每类随机至多取 samples_per_class 个样本（分层），用于小数据快速验证。
    labels 须已为 0..num_classes-1。
    """
    y = np.asarray(train_labels)
    rng = np.random.RandomState(seed)
    parts = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n_take = min(int(samples_per_class), len(idx))
        chosen = rng.choice(idx, size=n_take, replace=False)
        parts.append(chosen)
    if not parts:
        return train_hsi, train_lidar, train_rgb, train_labels
    all_idx = np.concatenate(parts)
    rng.shuffle(all_idx)
    tr_h = train_hsi[all_idx]
    tr_l = train_lidar[all_idx]
    tr_y = y[all_idx]
    tr_rgb = train_rgb[all_idx] if train_rgb is not None else None
    return tr_h, tr_l, tr_rgb, tr_y


def split_train_val(
    train_hsi,
    train_lidar,
    train_rgb,
    train_labels,
    val_ratio,
    seed,
):
    """从训练 patch 中划分验证集（分层），test 集保持独立仅用于最终评估。"""
    if val_ratio <= 0 or val_ratio >= 1.0:
        return (
            train_hsi,
            train_lidar,
            train_rgb,
            train_labels,
            None,
            None,
            None,
            None,
        )
    n = len(train_labels)
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
    )
    tr_h = train_hsi[train_idx]
    va_h = train_hsi[val_idx]
    tr_l = train_lidar[train_idx]
    va_l = train_lidar[val_idx]
    tr_y = train_labels[train_idx]
    va_y = train_labels[val_idx]
    tr_rgb, va_rgb = None, None
    if train_rgb is not None:
        tr_rgb = train_rgb[train_idx]
        va_rgb = train_rgb[val_idx]
    return tr_h, tr_l, tr_rgb, tr_y, va_h, va_l, va_rgb, va_y


class IndexedTensorDataset(torch.utils.data.Dataset):
    """TensorDataset + 样本下标，供扩散特征确定性噪声使用。"""

    def __init__(self, *tensors):
        self.tensors = tensors
        assert all(len(tensors[0]) == len(t) for t in tensors)

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors) + (index,)

    def __len__(self):
        return len(self.tensors[0])


def build_dataloaders(
    train_hsi,
    train_lidar,
    train_rgb,
    train_labels,
    val_hsi,
    val_lidar,
    val_rgb,
    val_labels,
    test_hsi,
    test_lidar,
    test_rgb,
    test_labels,
    batch_size,
):
    if train_rgb is not None:
        train_dataset = IndexedTensorDataset(
            torch.from_numpy(train_hsi),
            torch.from_numpy(train_lidar),
            torch.from_numpy(train_rgb),
            torch.from_numpy(train_labels),
        )
        test_dataset = IndexedTensorDataset(
            torch.from_numpy(test_hsi),
            torch.from_numpy(test_lidar),
            torch.from_numpy(test_rgb),
            torch.from_numpy(test_labels),
        )
    else:
        train_dataset = IndexedTensorDataset(
            torch.from_numpy(train_hsi),
            torch.from_numpy(train_lidar),
            torch.from_numpy(train_labels),
        )
        test_dataset = IndexedTensorDataset(
            torch.from_numpy(test_hsi),
            torch.from_numpy(test_lidar),
            torch.from_numpy(test_labels),
        )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=18,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=18,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_hsi is not None and len(val_hsi) > 0:
        if val_rgb is not None:
            val_dataset = IndexedTensorDataset(
                torch.from_numpy(val_hsi),
                torch.from_numpy(val_lidar),
                torch.from_numpy(val_rgb),
                torch.from_numpy(val_labels),
            )
        else:
            val_dataset = IndexedTensorDataset(
                torch.from_numpy(val_hsi),
                torch.from_numpy(val_lidar),
                torch.from_numpy(val_labels),
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=18,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, test_loader


def batch_to_dict(batch, device, use_rgb_patches: bool):
    """与 opt['dataset']['modalities'] 一致，构造分类器所需的 data_dict（键名需与模态名相同）。"""
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
