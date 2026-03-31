"""
从已有 checkpoint 在测试集上评估，写出混淆矩阵摘要、错分列表，并按错分对保存 patch 供人工查看。

用法（在仓库根目录）：
  python eval_checkpoint.py --checkpoint path/to/best_model.pt --out-dir path/to/out
  python eval_checkpoint.py --checkpoint path/to/final --out-dir path/to/out

可选与训练对齐：
  set MMDIFF_MODALITY_COMBO=hsi+rgb+lidar
  python eval_checkpoint.py --checkpoint ... --out-dir ...

不修改 pipeline/param 内既有逻辑，仅 import 复用。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint + save misclassified patches")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="best_model.pt 或含 classifier.pt/model.pt 的目录（如 final、checkpoint-300）",
    )
    p.add_argument("--out-dir", type=str, required=True, help="输出目录")
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 表示使用 param.BATCH_SIZE",
    )
    p.add_argument(
        "--max-per-pair",
        type=int,
        default=50,
        help="每个 true->pred 错分对最多保存多少个 patch（防磁盘爆）",
    )
    p.add_argument(
        "--modalities",
        type=str,
        default="",
        help='与训练一致，如 "hsi+rgb+lidar"；会在 import param 前设置 MMDIFF_MODALITY_COMBO',
    )
    p.add_argument(
        "--student-checkpoint",
        type=str,
        default="",
        help="覆盖学生扩散目录（默认 param.STUDENT_CHECKPOINT）",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader num_workers；默认使用 param.NUM_WORKERS",
    )
    return p.parse_args()


def _resolve_classifier_weight(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"checkpoint 路径不存在: {p}")
    for name in ("classifier.pt", "model.pt"):
        cand = p / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"目录 {p} 下未找到 classifier.pt 或 model.pt；"
        f"若评估 best_model.pt，请直接传该文件路径到 --checkpoint"
    )


def _to_uint8_rgb(rgb_chw: np.ndarray) -> np.ndarray:
    x = np.transpose(rgb_chw, (1, 2, 0)).astype(np.float32)
    if float(x.max()) <= 1.5:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _to_uint8_gray(arr_2d: np.ndarray) -> np.ndarray:
    x = arr_2d.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - lo) / (hi - lo) * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def _save_png(img: np.ndarray, out_path: Path) -> None:
    if Image is None:
        return
    Image.fromarray(img).save(out_path)


def _build_test_loader_local(
    feats_vol: np.ndarray,
    rgb_vol,
    test_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
):
    """与 pipeline.data.build_test_loader 一致，仅允许本脚本指定 num_workers。"""
    from torch.utils.data import DataLoader

    from param import PATCH_WINDOW_SIZE
    from pipeline.data import PatchDataset

    ds = PatchDataset(
        feats_vol,
        rgb_vol,
        test_indices,
        window_size=PATCH_WINDOW_SIZE,
        training=False,
        rot_factor=1,
        supcon_dual_view=False,
    )
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main() -> int:
    args = _parse_args()

    if args.modalities:
        os.environ["MMDIFF_MODALITY_COMBO"] = args.modalities.strip()

    # 延迟 import：确保环境变量已写入后再加载 param
    from param import (
        BATCH_SIZE,
        DIFFUSION_NOISE_MODE,
        DIFFUSION_NORMALIZE_INPUT,
        FEAT_SCALES,
        NUM_CLASSES,
        NUM_WORKERS,
        RANDOM_SEED,
        STUDENT_CHECKPOINT,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        USE_CENTER_LOSS,
        USE_RGB_PATCHES,
        opt,
    )
    from model import create_multimodal_classifier
    from pipeline.classification_metrics import accuracies
    from pipeline.data import batch_to_dict, load_test_indices_shifted, load_train_bundle
    from pipeline.logging_utils import save_confusion_detail_log
    from pipeline.student_diffusion import StudentDiffusionWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ckpt_path = _resolve_classifier_weight(args.checkpoint)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_vol, rgb_vol, _train_idx, label_shift = load_train_bundle()
    test_indices = load_test_indices_shifted(label_shift)

    bs = int(args.batch_size) if args.batch_size > 0 else int(BATCH_SIZE)
    nw = int(args.num_workers) if args.num_workers is not None else int(NUM_WORKERS)
    test_loader = _build_test_loader_local(feats_vol, rgb_vol, test_indices, bs, nw)

    student_dir = (args.student_checkpoint or "").strip() or str(STUDENT_CHECKPOINT)
    diffusion = StudentDiffusionWrapper(
        student_dir,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        noise_mode=DIFFUSION_NOISE_MODE,
        noise_seed_base=RANDOM_SEED,
        normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
        feat_layers=FEAT_SCALES,
    )

    model = create_multimodal_classifier(opt, diffusion).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    loss_fn = model.loss_func

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    running_loss = 0.0
    total = 0
    saved_per_pair: dict[tuple[int, int], int] = defaultdict(int)
    mis_rows: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval", dynamic_ncols=True):
            data_dict, labels = batch_to_dict(batch, device, USE_RGB_PATCHES, use_supcon=False)

            if USE_CENTER_LOSS:
                _, logits_c = model(data_dict, return_center_logits=True)
                logits = logits_c
            else:
                logits = model(data_dict)

            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)

            bs_ = labels.size(0)
            running_loss += float(loss.item()) * bs_
            total += bs_

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            sample_idx_np = data_dict["sample_indices"].detach().cpu().numpy()

            all_preds.append(preds_np)
            all_targets.append(labels_np)

            if USE_RGB_PATCHES:
                hsi_b, lidar_b, rgb_b, _lab, _idx = batch
                rgb_np = rgb_b.numpy()
            else:
                hsi_b, lidar_b, _lab, _idx = batch
                rgb_np = None

            hsi_np = hsi_b.numpy()
            lidar_np = lidar_b.numpy()

            for i in range(bs_):
                t_cls = int(labels_np[i])
                p_cls = int(preds_np[i])
                if p_cls == t_cls:
                    continue

                ds_idx = int(sample_idx_np[i])
                row = int(test_indices[ds_idx, 1])
                col = int(test_indices[ds_idx, 2])

                mis_rows.append(
                    {
                        "dataset_index": ds_idx,
                        "row": row,
                        "col": col,
                        "true": t_cls,
                        "pred": p_cls,
                    }
                )

                pair = (t_cls, p_cls)
                if saved_per_pair[pair] >= int(args.max_per_pair):
                    continue
                saved_per_pair[pair] += 1

                sub = (
                    out_dir
                    / "errors"
                    / f"{t_cls:02d}_to_{p_cls:02d}"
                    / f"idx_{ds_idx:06d}_r{row}_c{col}"
                )
                sub.mkdir(parents=True, exist_ok=True)

                np.save(sub / "hsi.npy", hsi_np[i])
                np.save(sub / "lidar.npy", lidar_np[i])
                if rgb_np is not None:
                    np.save(sub / "rgb.npy", rgb_np[i])
                    _save_png(_to_uint8_rgb(rgb_np[i]), sub / "rgb.png")
                _save_png(_to_uint8_gray(lidar_np[i][0]), sub / "lidar.png")

                meta = {
                    "checkpoint": str(ckpt_path),
                    "dataset_index": ds_idx,
                    "row": row,
                    "col": col,
                    "true": t_cls,
                    "pred": p_cls,
                }
                (sub / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    conf = confusion_matrix(targets_all, preds_all, labels=np.arange(NUM_CLASSES))

    eval_loss = running_loss / max(total, 1)
    eval_acc = float(np.mean(preds_all == targets_all))
    oa, _usr, _prod, kappa, _s_sqr, aa = accuracies(conf)

    save_confusion_detail_log(out_dir / "conf_detail.log", conf, NUM_CLASSES)

    csv_path = out_dir / "misclassified.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset_index", "row", "col", "true", "pred"],
        )
        w.writeheader()
        w.writerows(mis_rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "student_checkpoint": student_dir,
        "num_test_samples": int(len(targets_all)),
        "num_errors": int(np.sum(preds_all != targets_all)),
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "use_center_loss_path": bool(USE_CENTER_LOSS),
        "use_rgb_patches": bool(USE_RGB_PATCHES),
        "max_per_pair": int(args.max_per_pair),
        "saved_per_pair_counts": {
            f"{t}->{p}": int(v) for (t, p), v in sorted(saved_per_pair.items())
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
