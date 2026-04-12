"""
从 SZUTree R1 的 RGB.mat 读取高分辨率 RGB，切成 patch_size×patch_size 的 PNG。

默认：stride=patch_size（互不重叠）；若边缘不足一块则不再向右/向下延伸（与整除网格一致）。

用法：
  python export_rgb_hr_patches.py
  python export_rgb_hr_patches.py --stride 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm


def load_rgb_chw(mat_path: Path) -> np.ndarray:
    d = sio.loadmat(mat_path)
    if "data" not in d:
        raise KeyError(f"{mat_path} 中缺少变量 data")
    x = np.asarray(d["data"])
    if x.ndim != 3:
        raise ValueError(f"RGB 期望 3D，得到 shape={x.shape}")
    if x.shape[0] == 3:
        return x
    if x.shape[-1] == 3:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"无法识别 RGB 布局: shape={x.shape}")


def iter_patch_toplefts(h: int, w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    if h < patch or w < patch:
        raise ValueError(f"图像 {h}x{w} 小于 patch {patch}x{patch}")
    tops: list[int] = []
    lefts: list[int] = []
    t = 0
    while t + patch <= h:
        tops.append(t)
        t += stride
    l = 0
    while l + patch <= w:
        lefts.append(l)
        l += stride
    return [(r, c) for r in tops for c in lefts]


def main() -> None:
    p = argparse.ArgumentParser(description="RGB.mat → 24×24 PNG patches")
    p.add_argument(
        "--mat",
        type=Path,
        default=Path(r"E:\autodl-fs\SZUTreeData2.0\SZUTreeData_R1_2.0\RGB.mat"),
        help="RGB.mat 路径",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\autodl-fs\SZUTreeData2.0\rgb_hr"),
        help="输出 PNG 目录",
    )
    p.add_argument("--patch", type=int, default=24, help="patch 边长（像素）")
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="步长；默认等于 patch（互不重叠）",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="rgb",
        help="文件名前缀：{prefix}_r{row}_c{col}.png",
    )
    args = p.parse_args()
    patch = int(args.patch)
    stride = int(args.stride) if args.stride is not None else patch
    if stride < 1 or patch < 1:
        raise SystemExit("patch 与 stride 须为正整数")

    print(f"读取 {args.mat} …")
    chw = load_rgb_chw(args.mat)
    _, h, w = chw.shape
    print(f"RGB shape (C,H,W) = {chw.shape}, dtype={chw.dtype}")

    positions = iter_patch_toplefts(h, w, patch, stride)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"共 {len(positions)} 个 patch → {args.out.resolve()}")

    for r, c in tqdm(positions, desc="写入 PNG", dynamic_ncols=True):
        tile = chw[:, r : r + patch, c : c + patch]
        hwc = np.transpose(tile, (1, 2, 0))
        if hwc.dtype != np.uint8:
            hwc = np.clip(hwc, 0, 255).astype(np.uint8)
        fn = args.out / f"{args.prefix}_r{r:05d}_c{c:05d}.png"
        Image.fromarray(hwc, mode="RGB").save(fn)

    print("完成。")


if __name__ == "__main__":
    main()
