"""
从 SZUTreeData_R1_2.0 生成与 Houston/本仓库兼容的 `szutree_r1.mat`（无 PCA，HSI 默认 98 波段）；
体积极大时 `savemat` 可能因 MAT v5 单块 ≤4GB 而失败，此时自动改存同茎名 `szutree_r1.npz` 并在 meta 中注明。

Houston 兼容约定（同 `utils/extrac_dataset.py` 中最终 `houston2018.mat` 键）:
  hsi, lidar, rgb, train, test
- hsi: float32, (H,W, D_hsi)
- lidar: float32, (H, W, 1)
- rgb: uint8, (3, H, W)
- train / test: 二维稀疏或稠密，非零为类别 id；与 hsi 同 (H,W)

论文级数据协议:
- `label.mat` 的 `data` 为唯一监督栅格：不转置、不插值、不众数、不裁切；仅按类划分得到 train/test 稀疏图。
- HSI、LiDAR、RGB 若与标签空间尺寸不同，仅对影像重采样到 label 的 (H,W)，并记录 `szutree_r1.meta.json`。

环境变量:
  MMDIFF_HSI_CHANNELS=98  （与 param / data_prepare 一致）

用法:
  python extract_szutree_dataset.py --inspect
  python extract_szutree_dataset.py --inspect --inspect-detail
  set MMDIFF_HSI_CHANNELS=98
  python extract_szutree_dataset.py --export
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.ndimage import zoom
from tqdm import tqdm


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _default_r1_dir() -> Path:
    return Path(r"../../autodl-fs/szutree")


def _default_out_mat() -> Path:
    here = Path(__file__).resolve().parent
    return (here / "../../autodl-fs/szutree/szutree_r1.mat").resolve()


def _resize_spatial_hwc(
    vol_hwc: np.ndarray, out_h: int, out_w: int, *, order: int = 1
) -> np.ndarray:
    in_h, in_w, c = vol_hwc.shape[0], vol_hwc.shape[1], vol_hwc.shape[2]
    if in_h == out_h and in_w == out_w:
        return vol_hwc
    z = np.ascontiguousarray(vol_hwc.astype(np.float32, copy=False))
    zh, zw = out_h / float(in_h), out_w / float(in_w)
    y = zoom(z, (zh, zw, 1.0), order=order, mode="nearest")
    y = y[:out_h, :out_w, :]
    if y.shape[0] < out_h or y.shape[1] < out_w or y.shape[2] != c:
        out = np.zeros((out_h, out_w, c), dtype=np.float32)
        h0, w0 = min(y.shape[0], out_h), min(y.shape[1], out_w)
        out[:h0, :w0, :] = y[:h0, :w0, :].astype(np.float32, copy=False)
        y = out
    return y.astype(np.float32, copy=False)


def load_hsi_h5(
    path: Path, *, hsi_bands: int, show_progress: bool = True, band_chunk: int = 8
) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "hyperspectral_data_98bands" not in f:
            raise KeyError(f"{path} 中缺少 hyperspectral_data_98bands")
        ds = f["hyperspectral_data_98bands"]
        shp = ds.shape
        if show_progress and len(shp) == 3 and shp[0] == hsi_bands:
            parts: list[np.ndarray] = []
            it = range(0, hsi_bands, band_chunk)
            for i in tqdm(
                it,
                desc="读取 HSI 波段",
                unit="块",
                leave=False,
                dynamic_ncols=True,
            ):
                sl = slice(i, min(i + band_chunk, hsi_bands))
                parts.append(np.asarray(ds[sl], dtype=np.float32))
            x = np.concatenate(parts, axis=0)
        else:
            x = np.asarray(ds[:], dtype=np.float32)
    if x.shape[0] == hsi_bands:
        x = np.transpose(x, (1, 2, 0))
    elif x.shape[-1] == hsi_bands:
        pass
    else:
        raise ValueError(
            f"无法识别 HSI 形状: {x.shape}（期望首维或末维为 {hsi_bands} 个波段）"
        )
    return x


def load_lidar_h5(path: Path, *, show_progress: bool = True) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "chm" not in f:
            raise KeyError(f"{path} 中缺少 chm")
        ds = f["chm"]
        if show_progress:
            with tqdm(
                total=1, desc="读取 LiDAR", leave=False, dynamic_ncols=True
            ) as pbar:
                z = np.asarray(ds[:], dtype=np.float32)
                pbar.update(1)
        else:
            z = np.asarray(ds[:], dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"LiDAR 期望 2D，得到 {z.shape}")
    return z[..., np.newaxis]


def load_rgb_mat(path: Path, *, show_progress: bool = True) -> np.ndarray:
    if show_progress:
        with tqdm(
            total=1, desc="读取 RGB.mat", leave=False, dynamic_ncols=True
        ) as pbar:
            d = sio.loadmat(path)
            pbar.update(1)
    else:
        d = sio.loadmat(path)
    if "data" not in d:
        raise KeyError(f"{path} 中缺少 data")
    return np.asarray(d["data"])


def load_label_mat(path: Path, *, show_progress: bool = True) -> np.ndarray:
    if show_progress:
        with tqdm(
            total=1, desc="读取 label.mat", leave=False, dynamic_ncols=True
        ) as pbar:
            d = sio.loadmat(path)
            pbar.update(1)
    else:
        d = sio.loadmat(path)
    if "data" not in d:
        raise KeyError(f"{path} 中缺少 data")
    return np.asarray(d["data"])


def _rgb_to_hwc(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3:
        raise ValueError(f"RGB 期望 3D，得到 {rgb.shape}")
    if rgb.shape[0] == 3:
        return np.ascontiguousarray(np.transpose(rgb, (1, 2, 0)))
    if rgb.shape[-1] == 3:
        return np.ascontiguousarray(rgb)
    raise ValueError(
        f"无法识别 RGB 布局: {rgb.shape}，期望 (3,H,W) 或 (H,W,3)"
    )


def _rgb_hwc_to_chw_u8(hw: np.ndarray) -> np.ndarray:
    x = hw.astype(np.float32, copy=False)
    if x.size and float(np.nanmax(x)) <= 1.5:
        x = np.clip(x * 255.0, 0.0, 255.0)
    x = np.clip(np.rint(x), 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))


def _validate_label_values(
    label_2d: np.ndarray, num_classes: int, *, fail_on_unseen: bool
) -> list[int]:
    """返回前景中出现且不在 1..num_classes 的类别值（去重）。"""
    u = np.unique(label_2d)
    u_int = u.astype(np.int64, copy=False)
    bad: list[int] = []
    for v in u_int.tolist():
        if v == 0:
            continue
        if v < 1 or v > num_classes:
            bad.append(int(v))
    if bad and fail_on_unseen:
        raise ValueError(
            f"label 中出现前景类别 {bad!r}，与 --num-classes={num_classes} 不一致"
        )
    if bad:
        print(
            f"[WARN] label 中出现 num_classes 范围外的前景值 {bad!r}；"
            f"这些像素仍保留，但 build_train_test 仅遍历 1..{num_classes}。",
            file=sys.stderr,
        )
    return bad


def build_train_test_sparse(
    label_lr: np.ndarray,
    train_percent: float,
    seed: int,
    num_classes: int,
    *,
    show_progress: bool = True,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
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
        raise RuntimeError("train 为空，请检查标签与 train_percent / num_classes")

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


def _per_class_counts(
    label_2d: np.ndarray, train_sp: sparse.csr_matrix, test_sp: sparse.csr_matrix, num_classes: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in range(1, num_classes + 1):
        n_full = int(np.sum(label_2d == c))
        n_tr = int(np.sum(train_sp.data == c)) if train_sp.nnz else 0
        n_te = int(np.sum(test_sp.data == c)) if test_sp.nnz else 0
        out[str(c)] = {
            "label_pixels": n_full,
            "train_sparse_pixels": n_tr,
            "test_sparse_pixels": n_te,
        }
    return out


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
            except OSError as e:
                print("  h5py 读取失败:", e)
        elif name == "LiDAR.mat":
            try:
                with h5py.File(p, "r") as f:
                    print("  keys:", list(f.keys()))
                    ds = f["chm"]
                    print("  chm shape:", ds.shape, ds.dtype)
            except OSError as e:
                print("  h5py 读取失败:", e)
        else:
            try:
                info = sio.whosmat(str(p))
                print("  whosmat:", info)
            except OSError as e:
                print("  whosmat 失败:", e)

    if lab_p.is_file() and not detail:
        print("\n提示: 加 --inspect-detail 可加载整幅 label 并统计（较慢）")
    elif lab_p.is_file() and detail:
        y = load_label_mat(lab_p, show_progress=False)
        u, _cnt = np.unique(y, return_counts=True)
        print("\n[label.mat] 详细统计")
        print("  shape:", y.shape, y.dtype)
        print("  唯一值数量:", len(u), "min/max:", int(u.min()), int(u.max()))
        print("  前景像素 (label>0):", int((y > 0).sum()))


def run_export(
    r1_dir: Path,
    out_mat: Path,
    train_percent: float,
    seed: int,
    num_classes: int,
    hsi_bands: int,
    save_train_test_dense: bool,
    fail_on_unseen_label_class: bool,
    *,
    show_progress: bool = True,
) -> None:
    label_raw = load_label_mat(r1_dir / "label.mat", show_progress=show_progress)
    if label_raw.ndim != 2:
        raise ValueError(f"label 须为 2D，得到 shape={label_raw.shape}")
    if not np.isfinite(label_raw).all():
        raise ValueError("label 含 NaN/Inf")
    label_2d = label_raw.astype(np.int32, copy=False)
    ref_h, ref_w = int(label_2d.shape[0]), int(label_2d.shape[1])

    hsi_0 = load_hsi_h5(
        r1_dir / "HSI.mat", hsi_bands=hsi_bands, show_progress=show_progress
    )
    if hsi_0.shape[2] != hsi_bands:
        raise ValueError(f"HSI 波段数 {hsi_0.shape[2]} 与 hsi_bands={hsi_bands} 不一致")
    lidar_0 = load_lidar_h5(r1_dir / "LiDAR.mat", show_progress=show_progress)
    rgb_0 = load_rgb_mat(r1_dir / "RGB.mat", show_progress=show_progress)
    rgb_in_layout = "CHW" if rgb_0.ndim == 3 and rgb_0.shape[0] == 3 else "HWC"
    rgb_hw_0 = _rgb_to_hwc(rgb_0)

    meta: dict[str, Any] = {
        "schema": "szutree_r1",
        "interpolation": "scipy.ndimage.zoom order=1, mode=nearest; labels untouched",
        "label_shape": [ref_h, ref_w],
        "hsi": {
            "in_shape": list(hsi_0.shape),
            "out_shape": [ref_h, ref_w, hsi_bands],
            "resampled": [hsi_0.shape[0], hsi_0.shape[1]] != [ref_h, ref_w],
        },
        "lidar": {
            "in_shape": list(lidar_0.shape),
            "out_shape": [ref_h, ref_w, 1],
            "resampled": [lidar_0.shape[0], lidar_0.shape[1]] != [ref_h, ref_w],
        },
        "rgb": {
            "in_shape": list(rgb_0.shape),
            "in_layout": rgb_in_layout,
            "out_shape": [3, ref_h, ref_w],
            "resampled": [rgb_hw_0.shape[0], rgb_hw_0.shape[1]] != [ref_h, ref_w],
        },
    }

    _validate_label_values(
        label_2d, num_classes, fail_on_unseen=fail_on_unseen_label_class
    )

    hsi = _resize_spatial_hwc(
        hsi_0, ref_h, ref_w, order=1
    ).astype(np.float32, copy=False)
    lidar = _resize_spatial_hwc(
        lidar_0, ref_h, ref_w, order=1
    ).astype(np.float32, copy=False)
    rgb_hw = _resize_spatial_hwc(
        rgb_hw_0, ref_h, ref_w, order=1
    )
    rgb_out = _rgb_hwc_to_chw_u8(rgb_hw)

    train_sp, test_sp = build_train_test_sparse(
        label_2d,
        train_percent,
        seed,
        num_classes,
        show_progress=show_progress,
    )

    meta["train_test_split"] = {
        "seed": int(seed),
        "train_percent_per_class": float(train_percent),
        "per_class": _per_class_counts(label_2d, train_sp, test_sp, num_classes),
    }
    meta["train_nnz"] = int(train_sp.nnz)
    meta["test_nnz"] = int(test_sp.nnz)

    out_mat.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_mat.with_name(out_mat.stem + ".meta.json")

    if save_train_test_dense:
        if show_progress:
            tqdm.write(
                "train/test 稀疏 → 稠密（可能较慢、占内存）…",
                file=sys.stderr,
            )
        train_arr = train_sp.toarray()
        test_arr = test_sp.toarray()
        payload: dict[str, Any] = {
            "hsi": hsi,
            "lidar": lidar,
            "rgb": rgb_out,
            "train": train_arr,
            "test": test_arr,
        }
    else:
        train_arr = None
        test_arr = None
        payload = {
            "hsi": hsi,
            "lidar": lidar,
            "rgb": rgb_out,
            "train": train_sp,
            "test": test_sp,
        }

    if save_train_test_dense and train_arr is not None and test_arr is not None:
        est_b = int(
            hsi.nbytes
            + lidar.nbytes
            + rgb_out.nbytes
            + train_arr.nbytes
            + test_arr.nbytes
        )
    else:
        est_b = int(
            hsi.nbytes
            + lidar.nbytes
            + rgb_out.nbytes
            + train_sp.data.nbytes
            + train_sp.indices.nbytes
            + train_sp.indptr.nbytes
            + test_sp.data.nbytes
            + test_sp.indices.nbytes
            + test_sp.indptr.nbytes
        )
    meta["est_payload_bytes"] = est_b

    out_written: Path
    try:
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
        meta["storage"] = "mat_v5"
        meta["output_file"] = str(out_mat.resolve())
        out_written = out_mat
    except OverflowError:
        out_npz = out_mat.with_suffix(".npz")
        if show_progress:
            tqdm.write(
                f"MAT v5 单块 ≤4GB 限制，改存 {out_npz.name}（NumPy 压缩包）…",
                file=sys.stderr,
            )
        else:
            print(
                f"savemat 超出 MAT v5 单块限制，改存 {out_npz} …",
                file=sys.stderr,
            )
        if save_train_test_dense and train_arr is not None and test_arr is not None:
            np.savez_compressed(
                out_npz,
                hsi=hsi,
                lidar=lidar,
                rgb=rgb_out,
                train=train_arr,
                test=test_arr,
            )
        else:
            np.savez_compressed(
                out_npz,
                hsi=hsi,
                lidar=lidar,
                rgb=rgb_out,
                train_data=train_sp.data,
                train_indices=train_sp.indices,
                train_indptr=train_sp.indptr,
                train_shape=np.asarray(train_sp.shape, dtype=np.int64),
                test_data=test_sp.data,
                test_indices=test_sp.indices,
                test_indptr=test_sp.indptr,
                test_shape=np.asarray(test_sp.shape, dtype=np.int64),
            )
        meta["storage"] = "npz"
        meta["output_file"] = str(out_npz.resolve())
        out_written = out_npz
        if out_mat.exists():
            out_mat.unlink()

    meta["output_mat"] = str(out_mat.resolve())
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"完成: {out_written}\n  meta: {meta_path}\n"
        f"  HSI 波段数={hsi_bands}，请同步 param / data_prepare 的 HSI_CHANNELS。\n"
        f"  train 非零: {train_sp.nnz}, test 非零: {test_sp.nnz}"
    )
    print(
        f"  最终: hsi {hsi.shape} lidar {lidar.shape} rgb {rgb_out.shape} "
        f"label {label_2d.shape}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="SZUTree R1 -> szutree_r1.mat（标签栅格不动，仅重采样影像）"
    )
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
        help="输出 .mat 路径（默认 ../../autodl-fs/szutree/szutree_r1.mat）",
    )
    p.add_argument("--inspect", action="store_true", help="仅检查并打印信息")
    p.add_argument(
        "--inspect-detail",
        action="store_true",
        help="与 --inspect 合用：加载整幅 label 并统计",
    )
    p.add_argument("--export", action="store_true", help="导出总 .mat + .meta.json")
    p.add_argument(
        "--train-percent-per-class",
        type=float,
        default=1.0,
        help="每类训练像素占该类总像素的百分比，例如 1.0 表示 1%%",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-classes", type=int, default=20)
    p.add_argument(
        "--hsi-bands",
        type=int,
        default=None,
        help="HSI 波段数；默认 MMDIFF_HSI_CHANNELS 或 98",
    )
    p.add_argument(
        "--fail-on-unseen-label",
        action="store_true",
        help="若 label 前景值不在 1..num_classes 则直接报错",
    )
    p.add_argument(
        "--train-test-dense",
        action="store_true",
        help="train/test 以稠密保存（大；默认稀疏）",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 tqdm 进度条",
    )
    args = p.parse_args()
    out_path = args.out if args.out is not None else _default_out_mat()
    hsi_b = args.hsi_bands if args.hsi_bands is not None else _int_env("MMDIFF_HSI_CHANNELS", 98)

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
            hsi_bands=hsi_b,
            save_train_test_dense=args.train_test_dense,
            fail_on_unseen_label_class=bool(args.fail_on_unseen_label),
            show_progress=not args.no_progress,
        )
        return

    p.print_help()
    print("\n请指定 --inspect 或 --export", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
