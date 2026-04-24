#!/usr/bin/env python3
"""
对 prepared 数据做快速体检：形状/类型/范围/标签分布，并输出预览图。

默认将报告与图保存到与 param 中 autodl 路径风格一致的 ../../autodl-tmp/data_report_prepared。

用法（仓库根目录）:
  python utils/verify_prepared_data.py
  python utils/verify_prepared_data.py --out ../../autodl-tmp/my_report
  python utils/verify_prepared_data.py --data-dir /path/to/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _default_out_dir() -> Path:
    # 与 param 中 ../../autodl-tmp 同层级：从仓库根向上两级再进 autodl-tmp
    return (_REPO_ROOT / ".." / ".." / "autodl-tmp" / "data_report_prepared").resolve()


def _import_param(data_dir: Optional[Path] = None):
    if data_dir is not None:
        os.environ["MMDIFF_DATA_DIR"] = str(data_dir.resolve())
    import param

    if data_dir is not None:
        d = data_dir
        param.DATA_DIR = d
        param.TRAIN_PATCHES_PATH = d / "train_patches.npy"
        param.TRAIN_RGB_PATCHES_PATH = d / "train_rgb_patches.npy"
        param.TRAIN_LABELS_PATH = d / "train_labels.npy"
        param.TEST_LABELS_PATH = d / "test_labels.npy"
        param.LABEL_SHIFT_PATH = d / "label_shift.npy"
        param.TRAIN_RGB_HR_PATH = d / "rgb_hr.npy"
        param.RGB_HR_META_PATH = d / "rgb_hr.meta.json"
    return param


def _safe_stat(name: str, arr: np.ndarray, max_sample: int) -> Dict[str, Any]:
    flat = arr.reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "empty": True}
    if n > max_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(max_sample, n), replace=False)
        sample = np.asarray(flat[idx], dtype=np.float64)
    else:
        sample = np.asarray(flat, dtype=np.float64)
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_elements": n,
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
        "subsampled": n > max_sample,
    }


def _label_report(labels: np.ndarray, num_classes: int, name: str) -> Dict[str, Any]:
    if labels.size == 0:
        return {"name": name, "n": 0}
    raw = labels[:, 0].astype(np.int64)
    uniq, cnt = np.unique(raw, return_counts=True)
    return {
        "name": name,
        "n": int(len(labels)),
        "unique_labels": [int(x) for x in uniq.tolist()],
        "per_class_count": {int(k): int(v) for k, v in zip(uniq, cnt)},
        "num_classes_in_param": int(num_classes),
        "row_range": [int(labels[:, 1].min()), int(labels[:, 1].max())],
        "col_range": [int(labels[:, 2].min()), int(labels[:, 2].max())],
    }


def _to_uint8_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-12:
        if x.ndim == 2:
            return np.zeros(x.shape, dtype=np.uint8)
        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    if y.ndim == 2:
        return (y * 255.0 + 0.5).astype(np.uint8)
    return (y * 255.0 + 0.5).astype(np.uint8)


def _fig_overview(
    feats_hwc: np.ndarray,
    rgb_hwc: np.ndarray,
    hsi_ch: int,
    lidar_ch: int,
    band_triplet: Tuple[int, int, int],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hsi = feats_hwc[:, :, :hsi_ch]
    lidar = feats_hwc[:, :, hsi_ch : hsi_ch + lidar_ch]
    b0, b1, b2 = [min(max(0, b), hsi_ch - 1) for b in band_triplet]
    hsi_rgb = np.stack([hsi[:, :, b2], hsi[:, :, b1], hsi[:, :, b0]], axis=2)
    hsi_u8 = _to_uint8_image(hsi_rgb)
    rgb_u8 = _to_uint8_image(rgb_hwc)
    lidar_u8 = _to_uint8_image(lidar[:, :, 0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(hsi_u8)
    axes[0].set_title(f"HSI 假彩 (R,G,B <- band {b0},{b1},{b2} 与 stack 序)")
    axes[0].axis("off")
    axes[1].imshow(lidar_u8, cmap="gray")
    axes[1].set_title("LiDAR")
    axes[1].axis("off")
    axes[2].imshow(rgb_u8)
    axes[2].set_title("train_rgb_patches (LR 栅格)")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_patches(
    feats_hwc: np.ndarray,
    train_idx: np.ndarray,
    hsi_ch: int,
    window: int,
    n_show: int,
    seed: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n = min(n_show, len(train_idx))
    if n == 0:
        return
    pick = rng.choice(len(train_idx), size=n, replace=False)
    m = int(np.ceil(np.sqrt(n)))
    half = window // 2
    h, w, _ = feats_hwc.shape

    fig, axes = plt.subplots(m, m, figsize=(2.2 * m, 2.2 * m))
    axes = np.atleast_2d(axes)
    flat = axes.ravel()
    t0, t1, t2 = hsi_ch // 4, hsi_ch // 2, (3 * hsi_ch) // 4

    for k in range(m * m):
        ax = flat[k]
        if k >= n:
            ax.axis("off")
            continue
        ii = int(pick[k])
        row, col = int(train_idx[ii, 1]), int(train_idx[ii, 2])
        lab = int(train_idx[ii, 0])
        r0, r1 = max(0, row - half), min(h, row + half + 1)
        c0, c1 = max(0, col - half), min(w, col + half + 1)
        ph = feats_hwc[r0:r1, c0:c1, :hsi_ch]
        patch = np.stack([ph[:, :, t2], ph[:, :, t1], ph[:, :, t0]], axis=2)
        pu = _to_uint8_image(patch)
        pad = np.zeros((window, window, 3), dtype=np.uint8)
        pad[: pu.shape[0], : pu.shape[1], :] = pu
        ax.imshow(pad)
        ax.set_title(f"cls {lab} @({row},{col})", fontsize=7)
        ax.axis("off")
    plt.suptitle(f"随机 {n} 个像元: HSI 假彩 {window}×{window} patch", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_rgb_hr(hr_vol: np.ndarray, out_path: Path, downsample: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w, c = hr_vol.shape
    hh = hr_vol[::downsample, ::downsample, :]
    u8 = _to_uint8_image(hh)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * h / max(w, 1)))
    ax.imshow(u8)
    ax.set_title(f"rgb_hr (stride {downsample}) 原 {h}×{w}×{c}")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="prepared 数据体检 + 报告")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="prepared 目录；默认同 param.DATA_DIR 或环境变量 MMDIFF_DATA_DIR",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=f"输出目录；默认 {_default_out_dir()}",
    )
    parser.add_argument("--max-pixels", type=int, default=2_000_000, help="每数组统计最大采样元素数")
    parser.add_argument("--no-figures", action="store_true", help="不生成 png")
    parser.add_argument(
        "--hsi-bands",
        type=str,
        default="29,19,9",
        help="整幅 HSI 假彩三波段索引，逗号分隔，须在 [0, HSI-1]",
    )
    args = parser.parse_args()

    data_dir: Optional[Path] = None
    s = (args.data_dir or "").strip()
    if s:
        data_dir = Path(s).expanduser().resolve()
    else:
        es = (os.environ.get("MMDIFF_DATA_DIR") or "").strip()
        if es:
            data_dir = Path(es).expanduser().resolve()

    param = _import_param(data_dir)
    hsi_ch, lidar_ch = int(param.HSI_CHANNELS), int(param.LIDAR_CHANNELS)
    try:
        band_triplet = tuple(int(x.strip()) for x in args.hsi_bands.split(","))
        if len(band_triplet) != 3:
            raise ValueError
    except (ValueError, TypeError):
        print("错误: --hsi-bands 需要三个整数，如 29,19,9", file=sys.stderr)
        return 2

    out = Path((args.out or "").strip() or str(_default_out_dir()))
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "data_dir": str(param.DATA_DIR),
        "output_dir": str(out),
        "hsi_channels": hsi_ch,
        "lidar_channels": lidar_ch,
        "num_classes": int(param.NUM_CLASSES),
        "patch_window_size": int(param.PATCH_WINDOW_SIZE),
    }
    text: List[str] = [
        "=== prepared 数据报告 ===",
        f"DATA_DIR: {param.DATA_DIR}",
        f"本报告输出: {out}",
        "",
    ]

    def open_mmap(p: Path):
        return np.load(str(p), mmap_mode="r")

    tp = param.TRAIN_PATCHES_PATH
    if not tp.is_file():
        text.append(f"[缺失] {tp}")
    else:
        mm = open_mmap(tp)
        report["train_patches"] = _safe_stat("train_patches", mm, args.max_pixels)
        text.append(f"[OK] train_patches: shape={mm.shape} dtype={mm.dtype}")
        text.append(
            f"     采样 min/max/mean~ {report['train_patches']['min']:.4g} / "
            f"{report['train_patches']['max']:.4g} / {report['train_patches']['mean']:.4g}"
        )

    tr = param.TRAIN_RGB_PATCHES_PATH
    rgb = None
    if not tr.is_file():
        text.append(f"[缺失] {tr}")
    else:
        rgb = open_mmap(tr)
        report["train_rgb_patches"] = _safe_stat("train_rgb", rgb, args.max_pixels)
        text.append(f"[OK] train_rgb_patches: shape={rgb.shape}")
        text.append(
            f"     采样 min/max/mean~ {report['train_rgb_patches']['min']:.4g} / "
            f"{report['train_rgb_patches']['max']:.4g} / {report['train_rgb_patches']['mean']:.4g}"
        )

    for key, pth in (("train_labels", param.TRAIN_LABELS_PATH), ("test_labels", param.TEST_LABELS_PATH)):
        if not pth.is_file():
            text.append(f"[缺失] {pth}")
            continue
        lab = np.load(str(pth))
        report[key] = _label_report(lab, param.NUM_CLASSES, key)
        text.append(
            f"[OK] {key}: n={report[key]['n']} 类数(出现)={len(report[key]['per_class_count'])} "
            f"row{report[key]['row_range']}"
        )

    if param.LABEL_SHIFT_PATH.is_file():
        sh = int(np.load(str(param.LABEL_SHIFT_PATH)).ravel()[0])
        report["label_shift"] = sh
        text.append(f"[OK] label_shift = {sh}")

    if param.TRAIN_RGB_HR_PATH.is_file():
        mm = open_mmap(param.TRAIN_RGB_HR_PATH)
        report["rgb_hr"] = _safe_stat("rgb_hr", mm, args.max_pixels)
        text.append(f"[OK] rgb_hr: shape={mm.shape}")
    if param.RGB_HR_META_PATH.is_file():
        report["rgb_hr_meta"] = json.loads(param.RGB_HR_META_PATH.read_text(encoding="utf-8"))
        m = report["rgb_hr_meta"]
        text.append(f"[OK] rgb_hr meta: rh={m.get('rh')} rw={m.get('rw')}")

    text.extend(
        [
            "",
            "说明: train_patches 为 (H,W, HSI+LiDAR); train_rgb 为同栅格 (H,W,3).",
        ]
    )

    (out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))
    print(f"\n已写入: {out / 'summary.txt'} 与 {out / 'summary.json'}")

    if args.no_figures or not tp.is_file() or rgb is None:
        if not args.no_figures and (not tp.is_file() or rgb is None):
            print("跳过作图: 缺 train_patches 或 train_rgb_patches", file=sys.stderr)
        return 0

    feats = open_mmap(tp)
    fhwc = feats[:, :, : hsi_ch + lidar_ch]
    rhw = rgb[:, :, :3]

    _fig_overview(
        fhwc,
        rhw,
        hsi_ch,
        lidar_ch,
        band_triplet,
        out / "fig01_overview_hsi_lidar_rgb.png",
    )
    print(f"作图: {out / 'fig01_overview_hsi_lidar_rgb.png'}")

    if param.TRAIN_LABELS_PATH.is_file():
        tidx = np.load(str(param.TRAIN_LABELS_PATH))
        _fig_patches(
            fhwc,
            tidx,
            hsi_ch,
            int(param.PATCH_WINDOW_SIZE),
            9,
            0,
            out / "fig02_sample_patches_hsi.png",
        )
        print(f"作图: {out / 'fig02_sample_patches_hsi.png'}")

    if param.TRAIN_RGB_HR_PATH.is_file():
        try:
            hr = open_mmap(param.TRAIN_RGB_HR_PATH)
            ds = max(1, int(max(hr.shape[0], hr.shape[1]) // 1000) or 1)
            _fig_rgb_hr(
                np.asarray(hr[:, :, :3], dtype=np.float32),
                out / "fig03_rgb_hr_full.png",
                ds,
            )
            print(f"作图: {out / 'fig03_rgb_hr_full.png'} (HR 大时 stride 显示)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] rgb_hr 作图跳过: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
