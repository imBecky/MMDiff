"""
从 SZUTreeData_R1_2.0 多个 .mat 生成与 Houston 流程兼容的总 houston2018.mat（无 PCA，HSI 保留 98 波段）。

必需键：hsi, lidar, rgb, train, test
- hsi: float32，形状 (H_lr, W_lr, 98)，与 LiDAR/标签低分辨率网格一致
- lidar: float32，形状 (H_lr, W_lr, 1)
- rgb: uint8，形状 (3, H_hr, W_hr)
- train / test: 二维稀疏或稠密标签图，非零为类别 id（1..20），与 hsi 同空间尺寸

标签：5cm label.mat 通过 2×2 块众数下采样到 10cm，与 HSI/LiDAR 对齐。

用法：
  python extract_szutree_dataset.py --inspect
  set MMDIFF_HSI_CHANNELS=98
  python extract_szutree_dataset.py --export
  python data_prepare.py

Windows PowerShell:
  $env:MMDIFF_HSI_CHANNELS=98
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.stats import mode as scipy_mode
from tqdm import tqdm


def _default_r1_dir() -> Path:
    return Path(r"E:\autodl-fs\SZUTreeData2.0\SZUTreeData_R1_2.0")


def _default_out_mat() -> Path:
    here = Path(__file__).resolve().parent
    return (here / "../../autodl-fs/houston2018/houston2018.mat").resolve()


def load_hsi_h5(path: Path, *, show_progress: bool = True, band_chunk: int = 8) -> np.ndarray:
    """返回 float32，形状 (H, W, 98)，空间维与 LiDAR (3085, 2405) 一致。"""
    with h5py.File(path, "r") as f:
        if "hyperspectral_data_98bands" not in f:
            raise KeyError(f"{path} 中缺少 hyperspectral_data_98bands")
        ds = f["hyperspectral_data_98bands"]
        shp = ds.shape
        # 沿波段维分块读，便于大文件时显示进度
        if show_progress and len(shp) == 3 and shp[0] == 98:
            parts: list[np.ndarray] = []
            it = range(0, 98, band_chunk)
            for i in tqdm(
                it,
                desc="读取 HSI 波段",
                unit="块",
                leave=False,
                dynamic_ncols=True,
            ):
                sl = slice(i, min(i + band_chunk, 98))
                parts.append(np.asarray(ds[sl], dtype=np.float32))
            x = np.concatenate(parts, axis=0)
        else:
            x = np.asarray(ds[:], dtype=np.float32)
    # HDF5: (98, 3085, 2405) -> (3085, 2405, 98)
    if x.shape[0] == 98:
        x = np.transpose(x, (1, 2, 0))
    elif x.shape[-1] == 98:
        pass
    else:
        raise ValueError(f"无法识别 HSI 形状: {x.shape}")
    return x


def load_lidar_h5(path: Path, *, show_progress: bool = True) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "chm" not in f:
            raise KeyError(f"{path} 中缺少 chm")
        ds = f["chm"]
        if show_progress:
            with tqdm(total=1, desc="读取 LiDAR", leave=False, dynamic_ncols=True) as pbar:
                z = np.asarray(ds[:], dtype=np.float32)
                pbar.update(1)
        else:
            z = np.asarray(ds[:], dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"LiDAR 期望 2D，得到 {z.shape}")
    return z[..., np.newaxis]


def load_rgb_mat(path: Path, *, show_progress: bool = True) -> np.ndarray:
    if show_progress:
        with tqdm(total=1, desc="读取 RGB.mat", leave=False, dynamic_ncols=True) as pbar:
            d = sio.loadmat(path)
            pbar.update(1)
    else:
        d = sio.loadmat(path)
    if "data" not in d:
        raise KeyError(f"{path} 中缺少 data")
    return np.asarray(d["data"])


def load_label_mat(path: Path, *, show_progress: bool = True) -> np.ndarray:
    if show_progress:
        with tqdm(total=1, desc="读取 label.mat", leave=False, dynamic_ncols=True) as pbar:
            d = sio.loadmat(path)
            pbar.update(1)
    else:
        d = sio.loadmat(path)
    if "data" not in d:
        raise KeyError(f"{path} 中缺少 data")
    return np.asarray(d["data"])


def downsample_label_hr_to_lr(
    label_hr: np.ndarray, *, show_progress: bool = True
) -> np.ndarray:
    """HR (2*H, 2*W) -> LR (H,W)，2×2 众数。"""
    if show_progress:
        tqdm.write("标签 HR→LR (2×2 众数)…", file=sys.stderr)
    h_hr, w_hr = label_hr.shape
    if h_hr % 2 != 0 or w_hr % 2 != 0:
        raise ValueError(f"HR 标签高宽须为偶数，当前 {label_hr.shape}")
    h_lr, w_lr = h_hr // 2, w_hr // 2
    blocks = label_hr.reshape(h_lr, 2, w_lr, 2).transpose(0, 2, 1, 3)
    flat = blocks.reshape(h_lr, w_lr, 4)
    lr = scipy_mode(flat, axis=2, keepdims=False).mode
    if lr.ndim > 2:
        lr = np.squeeze(lr, axis=-1)
    return lr.astype(np.uint8, copy=False)


def align_lr_label_to_hsi_spatial(
    lr: np.ndarray, hsi_hw: tuple[int, int]
) -> np.ndarray:
    """
    将下采样后的 LR 标签与 HSI 空间维对齐。
    SZUTree 中 label 与 HSI 的 H/W 轴约定可能互为转置，此时仅对标签转置即可。
    """
    h, w = int(hsi_hw[0]), int(hsi_hw[1])
    if lr.shape == (h, w):
        return lr
    if lr.shape == (w, h):
        print(
            "[INFO] LR 标签与 HSI 空间维为转置关系，已对标签转置以与 HSI/LiDAR 对齐。",
            file=sys.stderr,
        )
        return np.ascontiguousarray(lr.T)
    print(
        f"[WARN] LR 标签 {lr.shape} 与 HSI {(h, w)} 仍不一致，保留原标签；若训练异常请检查坐标系。",
        file=sys.stderr,
    )
    return lr


def build_train_test_sparse(
    label_lr: np.ndarray,
    train_percent: float,
    seed: int,
    num_classes: int,
    *,
    show_progress: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    对每个类别 c，从该类像素中随机抽取约 train_percent% 进入 train，其余进入 test。
    train_percent 为「百分数」，例如 1.0 表示 1%。
    """
    rng = np.random.default_rng(seed)
    h, w = label_lr.shape
    train_r: list[np.ndarray] = []
    train_c: list[np.ndarray] = []
    train_v: list[np.ndarray] = []
    test_r: list[np.ndarray] = []
    test_c: list[np.ndarray] = []
    test_v: list[np.ndarray] = []

    class_iter = range(1, num_classes + 1)
    if show_progress:
        class_iter = tqdm(
            class_iter,
            desc="划分 train/test（按类）",
            unit="类",
            dynamic_ncols=True,
            leave=False,
        )
    for c in class_iter:
        coords = np.argwhere(label_lr == c)
        n = coords.shape[0]
        if n == 0:
            continue
        rng.shuffle(coords, axis=0)
        n_tr = int(round(n * train_percent / 100.0))
        n_tr = max(1, min(n, n_tr)) if n > 0 else 0
        tr = coords[:n_tr]
        te = coords[n_tr:]
        if tr.size:
            train_r.append(tr[:, 0].astype(np.int32))
            train_c.append(tr[:, 1].astype(np.int32))
            train_v.append(np.full(tr.shape[0], c, dtype=np.int32))
        if te.size:
            test_r.append(te[:, 0].astype(np.int32))
            test_c.append(te[:, 1].astype(np.int32))
            test_v.append(np.full(te.shape[0], c, dtype=np.int32))

    if not train_r:
        raise RuntimeError("train 为空，请检查标签与 train_percent")
    tr_rows = np.concatenate(train_r)
    tr_cols = np.concatenate(train_c)
    tr_data = np.concatenate(train_v)
    train_mat = sparse.coo_matrix(
        (tr_data, (tr_rows, tr_cols)), shape=(h, w), dtype=np.int32
    ).tocsr()

    if not test_r:
        test_mat = sparse.csr_matrix((h, w), dtype=np.int32)
    else:
        te_rows = np.concatenate(test_r)
        te_cols = np.concatenate(test_c)
        te_data = np.concatenate(test_v)
        test_mat = sparse.coo_matrix(
            (te_data, (te_rows, te_cols)), shape=(h, w), dtype=np.int32
        ).tocsr()

    return train_mat, test_mat


def run_inspect(r1_dir: Path, detail: bool = False) -> None:
    hsi_p = r1_dir / "HSI.mat"
    lidar_p = r1_dir / "LiDAR.mat"
    rgb_p = r1_dir / "RGB.mat"
    lab_p = r1_dir / "label.mat"

    print("=== SZUTreeData_R1_2.0 检查 ===")
    print(f"目录: {r1_dir.resolve()}")
    for p, name in [
        (hsi_p, "HSI.mat"),
        (lidar_p, "LiDAR.mat"),
        (rgb_p, "RGB.mat"),
        (lab_p, "label.mat"),
    ]:
        print(f"\n[{name}] exists={p.is_file()} path={p}")
        if not p.is_file():
            continue
        if name == "HSI.mat":
            try:
                with h5py.File(p, "r") as f:
                    print("  keys:", list(f.keys()))
                    ds = f["hyperspectral_data_98bands"]
                    print("  hyperspectral_data_98bands shape:", ds.shape, ds.dtype)
            except Exception as e:
                print("  h5py 读取失败:", e)
        elif name == "LiDAR.mat":
            try:
                with h5py.File(p, "r") as f:
                    print("  keys:", list(f.keys()))
                    ds = f["chm"]
                    print("  chm shape:", ds.shape, ds.dtype)
            except Exception as e:
                print("  h5py 读取失败:", e)
        else:
            try:
                info = sio.whosmat(str(p))
                print("  whosmat:", info)
            except Exception as e:
                print("  whosmat 失败:", e)

    if lab_p.is_file() and not detail:
        print("\n提示: 加 --inspect-detail 可加载整幅 label 并统计 LR（较慢）")
    elif lab_p.is_file() and detail:
        y = load_label_mat(lab_p, show_progress=False)
        u, _cnt = np.unique(y, return_counts=True)
        print("\n[label.mat] 详细统计")
        print("  HR shape:", y.shape)
        print("  唯一值数量:", len(u), "min/max:", int(u.min()), int(u.max()))
        lr = downsample_label_hr_to_lr(y, show_progress=False)
        print("  LR (2×2 众数) shape:", lr.shape)
        print("  LR 前景类像素 (label>0):", int((lr > 0).sum()))


def assemble_szutree_split_mats_payload(
    r1_dir: Path,
    train_percent: float,
    seed: int,
    num_classes: int,
    save_train_test_dense: bool,
    *,
    show_progress: bool = True,
) -> dict:
    """
    与同目录下的 HSI.mat / LiDAR.mat / RGB.mat / label.mat 读出并拼装为 fusion 字典（键与 houston2018 兼容：hsi,lidar,rgb,train,test）。
    train/test 默认可为 CSR；也可保存前转稠密。供 `extract_szutree_dataset.run_export` 与根目录 `data_prepare.py --szutree-dir` 复用。
    """
    hsi = load_hsi_h5(r1_dir / "HSI.mat", show_progress=show_progress)
    lidar = load_lidar_h5(r1_dir / "LiDAR.mat", show_progress=show_progress)
    rgb = load_rgb_mat(r1_dir / "RGB.mat", show_progress=show_progress)
    label_hr = load_label_mat(r1_dir / "label.mat", show_progress=show_progress)

    if lidar.shape[:2] != hsi.shape[:2]:
        raise ValueError(f"HSI 与 LiDAR 空间尺寸不一致: hsi {hsi.shape[:2]} vs lidar {lidar.shape[:2]}")
    lr_ds = downsample_label_hr_to_lr(label_hr, show_progress=show_progress)
    lr_before = lr_ds.shape
    lr = align_lr_label_to_hsi_spatial(lr_ds, hsi.shape[:2])
    if lr.shape != hsi.shape[:2]:
        raise ValueError(
            f"LR 标签 {lr_before} 对齐后 {lr.shape}，仍与 HSI {hsi.shape[:2]} 不一致"
            "（需为同一形状或互为转置）"
        )
    # 仅当 LR 标签下采样后与 HSI 互为转置、经 align 后已对齐时，对 RGB 做 (3,H,W)->(3,W,H)
    if lr_before != hsi.shape[:2] and lr.shape == hsi.shape[:2]:
        rgb = np.ascontiguousarray(np.transpose(rgb, (0, 2, 1)))
        print(
            "[INFO] 已对 RGB 作空间维转置，使 HR 行/列 = 2×HSI 行/列。",
            file=sys.stderr,
        )

    train_sp, test_sp = build_train_test_sparse(
        lr,
        train_percent,
        seed,
        num_classes,
        show_progress=show_progress,
    )

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"RGB 期望 (3,H,W)，得到 {rgb.shape}")
    rh, rw = int(rgb.shape[1]), int(rgb.shape[2])
    if rh != lr.shape[0] * 2 or rw != lr.shape[1] * 2:
        print(
            f"[WARN] RGB {rh}x{rw} 与 LR×2 {lr.shape[0]*2}x{lr.shape[1]*2} 不完全一致，"
            "data_prepare 会按块均值对齐到 LR。",
            file=sys.stderr,
        )

    common = {
        "hsi": hsi.astype(np.float32, copy=False),
        "lidar": lidar.astype(np.float32, copy=False),
        "rgb": rgb.astype(np.uint8, copy=False),
    }
    if save_train_test_dense:
        if show_progress:
            tqdm.write(
                "train/test 稀疏 → 稠密（可能较慢、占内存）…",
                file=sys.stderr,
            )
        return {
            **common,
            "train": train_sp.toarray(),
            "test": test_sp.toarray(),
        }
    return {**common, "train": train_sp, "test": test_sp}


def run_export(
    r1_dir: Path,
    out_mat: Path,
    train_percent: float,
    seed: int,
    num_classes: int,
    save_train_test_dense: bool,
    *,
    show_progress: bool = True,
) -> None:
    payload = assemble_szutree_split_mats_payload(
        r1_dir,
        train_percent,
        seed,
        num_classes,
        save_train_test_dense,
        show_progress=show_progress,
    )
    train_sp = payload["train"]
    if hasattr(train_sp, "nnz"):
        train_nnz, test_nnz = int(train_sp.nnz), int(payload["test"].nnz)
    else:
        train_nnz = int(np.count_nonzero(payload["train"]))
        test_nnz = int(np.count_nonzero(payload["test"]))

    out_mat.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        with tqdm(
            total=1,
            desc=f"写入 {out_mat.name}",
            dynamic_ncols=True,
        ) as pbar:
            sio.savemat(str(out_mat), payload, do_compression=True)
            pbar.update(1)
    else:
        print(f"写入 {out_mat} ...")
        sio.savemat(str(out_mat), payload, do_compression=True)
    print(
        "完成。HSI 为 98 波段、无 PCA。"
        " 运行 data_prepare 与训练前请设置: MMDIFF_HSI_CHANNELS=98"
    )
    print(f"  train 非零: {train_nnz}, test 非零: {test_nnz}")


def main() -> None:
    p = argparse.ArgumentParser(description="SZUTree R1 -> houston2018 兼容 .mat（无 PCA）")
    p.add_argument(
        "--r1-dir",
        type=Path,
        default=_default_r1_dir(),
        help="SZUTreeData_R1_2.0 目录",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 houston2018.mat 路径（默认 autodl-fs/houston2018/houston2018.mat）",
    )
    p.add_argument("--inspect", action="store_true", help="仅检查并打印信息（快速，不加载整幅标签）")
    p.add_argument(
        "--inspect-detail",
        action="store_true",
        help="与 --inspect 合用：加载 label 并做 LR 统计（较慢）",
    )
    p.add_argument("--export", action="store_true", help="导出总 .mat")
    p.add_argument(
        "--train-percent-per-class",
        type=float,
        default=1.0,
        help="每类抽取训练像素占该类总像素的百分比，例如 1.0 表示 1%%",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument(
        "--train-test-dense",
        action="store_true",
        help="train/test 以稠密矩阵保存（文件更大；默认稀疏）",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 tqdm 进度条（日志/重定向时更干净）",
    )
    args = p.parse_args()
    out_path = args.out if args.out is not None else _default_out_mat()

    if args.inspect:
        run_inspect(args.r1_dir, detail=args.inspect_detail)
        return
    if args.export:
        run_export(
            args.r1_dir,
            out_path,
            train_percent=args.train_percent_per_class,
            seed=args.seed,
            num_classes=args.num_classes,
            save_train_test_dense=args.train_test_dense,
            show_progress=not args.no_progress,
        )
        return

    p.print_help()
    print("\n请指定 --inspect 或 --export", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
