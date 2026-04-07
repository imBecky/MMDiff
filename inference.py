#!/usr/bin/env python3
"""
验证集最简推理：加载 best_model.pt，按与训练相同的划分跑一轮前向，导出错分清单与可视化。

用法（在仓库根目录，且当前 param / 环境与训练一致）：
  conda activate hbq
  python inference.py
  python inference.py --checkpoint ../../autodl-tmp/classifier/0406-1718_exp_hsi_se/best_model.pt

环境与数据路径以 param.py / MMDIFF_* 为准；验证集 = split_train_val_indices(VAL_RATIO, RANDOM_SEED)。

说明：`--help` 不导入 param，避免本机缺少 `autodl-fs` 数据时无法查看帮助。实跑前请设好与训练一致的 `MMDIFF_*`（如模态组合）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _rgb_hwc_for_plot(rgb_chw: Any) -> Any:
    """(3,H,W) float -> (H,W,3) for imshow, clipped to [0,1]."""
    import numpy as np

    x = np.transpose(rgb_chw, (1, 2, 0))
    return np.clip(x, 0.0, 1.0)


def _save_misclassified_figure(
    *,
    out_path: Path,
    hsi_chw: Any,
    lidar_chw: Any,
    rgb_chw: Optional[Any],
    title: str,
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c, h, w = hsi_chw.shape
    cy, cx = h // 2, w // 2
    spectrum = hsi_chw[:, cy, cx]

    if rgb_chw is not None:
        _, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].imshow(_rgb_hwc_for_plot(rgb_chw))
        axes[0].set_title('RGB patch')
        axes[0].axis('off')
        ax_l = axes[1]
        ax_s = axes[2]
    else:
        _, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
        ax_l = axes[0]
        ax_s = axes[1]

    lid = lidar_chw[0] if lidar_chw.shape[0] >= 1 else lidar_chw.mean(0)
    im = ax_l.imshow(lid, cmap='viridis')
    ax_l.set_title('LiDAR (ch0)')
    ax_l.axis('off')
    plt.colorbar(im, ax=ax_l, fraction=0.046, pad=0.04)

    ax_s.plot(np.arange(c), spectrum, 'b-', linewidth=1.0)
    ax_s.set_xlabel('band index')
    ax_s.set_ylabel('value')
    ax_s.set_title('HSI spectrum (center px)')
    ax_s.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def _build_contact_sheet(paths: List[Path], out_path: Path, max_tiles: int) -> None:
    if not paths:
        return
    import numpy as np
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    take = paths[:max_tiles]
    n = len(take)
    cols = min(4, n)
    rows = (n + cols - 1) // cols if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < n:
            p = take[i]
            try:
                img = mpimg.imread(str(p))
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, p.name, ha='center', va='center', fontsize=8)
            ax.set_title(p.stem[:42], fontsize=8)
        ax.axis('off')

    plt.suptitle(f'Misclassified (first {n})', fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def run_inference(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    import model as Model
    from param import (
        BATCH_SIZE,
        DIFFUSION_NOISE_MODE,
        DIFFUSION_NORMALIZE_INPUT,
        FEAT_SCALES,
        NUM_CLASSES,
        RANDOM_SEED,
        RGB_DIFFUSION_TEACHER_CHECKPOINT,
        RGB_SOURCE,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        TRAIN_QUICK_VERIFY,
        TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
        USE_CENTER_LOSS,
        USE_RGB_PATCHES,
        VAL_RATIO,
        opt,
    )
    from pipeline.data import (
        batch_to_dict,
        build_test_loader,
        load_rgb_hr_meta,
        load_rgb_hr_volume,
        load_train_bundle,
        split_train_val_indices,
        subset_train_indices_balanced,
    )
    from pipeline.student_diffusion import StudentDiffusionWrapper

    def create_classifier(diffusion):
        return Model.create_multimodal_classifier(opt, diffusion)

    def build_diffusion():
        rgb_src = (RGB_SOURCE or 'student').strip().lower()
        if rgb_src not in ('diffusion', 'student', 'cached_teacher'):
            raise ValueError(f'RGB_SOURCE / MMDIFF_RGB_SOURCE 无效: {rgb_src!r}')
        if rgb_src == 'diffusion':
            return StudentDiffusionWrapper(
                RGB_DIFFUSION_TEACHER_CHECKPOINT,
                STUDENT_NUM_TRAIN_TIMESTEPS,
                noise_mode=DIFFUSION_NOISE_MODE,
                noise_seed_base=RANDOM_SEED,
                normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
                feat_layers=FEAT_SCALES,
            )
        return None

    def forward_logits(model, data_dict: Dict[str, Any]) -> torch.Tensor:
        if USE_CENTER_LOSS:
            _logits_g, logits_c = model(data_dict, return_center_logits=True)
            return logits_c
        return model(data_dict)

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        print(f'[error] checkpoint 不存在: {ckpt}', file=sys.stderr)
        return 2

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ckpt.parent / 'inference_val_errors'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    feats_vol, rgb_vol, train_indices, _label_shift = load_train_bundle()
    rgb_hr_vol = None
    hr_rh = 1
    hr_rw = 1
    if USE_RGB_PATCHES:
        rgb_hr_vol = load_rgb_hr_volume()
        _hr_meta = load_rgb_hr_meta()
        hr_rh = int(_hr_meta['rh'])
        hr_rw = int(_hr_meta['rw'])

    if TRAIN_QUICK_VERIFY:
        train_indices = subset_train_indices_balanced(
            train_indices,
            TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
            RANDOM_SEED,
            NUM_CLASSES,
        )

    _tr_idx, va_idx, _tr_pos, va_pos = split_train_val_indices(
        train_indices, VAL_RATIO, RANDOM_SEED
    )
    if va_idx is None or len(va_idx) == 0:
        print(
            f'[error] 无验证集：请设 0 < VAL_RATIO < 1（当前 VAL_RATIO={VAL_RATIO}）',
            file=sys.stderr,
        )
        return 3

    batch_size = (
        int(args.batch_size) if args.batch_size is not None else int(BATCH_SIZE)
    )

    val_loader = build_test_loader(
        feats_vol,
        rgb_vol,
        va_idx,
        batch_size,
        global_row_indices=va_pos,
        rgb_strict_view=bool(USE_RGB_PATCHES),
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
    )

    diffusion = build_diffusion()
    model = create_classifier(diffusion).to(device)

    try:
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    loss_fn = model.loss_func

    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    rows_records: List[Dict[str, Any]] = []
    figure_paths: List[Path] = []

    use_rgb = USE_RGB_PATCHES
    n_err_fig = 0
    running_loss = 0.0
    total = 0

    val_offset = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val inference', leave=False):
            data_dict, labels = batch_to_dict(batch, device, use_rgb, use_supcon=False)
            logits = forward_logits(model, data_dict)
            loss = loss_fn(logits, labels)
            bs = int(labels.size(0))
            running_loss += float(loss.item()) * bs
            total += bs

            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            preds_all.append(pred.detach().cpu().numpy())
            targets_all.append(labels.detach().cpu().numpy())

            mask = pred != labels
            if mask.any():
                wrong_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                gr_cpu = data_dict['global_row'].detach().cpu().numpy()
                lab_cpu = labels.detach().cpu().numpy()
                pred_cpu = pred.detach().cpu().numpy()
                hsi_cpu = data_dict['hsi'].detach().cpu().numpy()
                lid_cpu = data_dict['lidar'].detach().cpu().numpy()
                rgb_cpu = data_dict['rgb'].detach().cpu().numpy() if use_rgb else None
                prob_cpu = probs.detach().cpu().numpy()

                for j in wrong_idx.reshape(-1).tolist():
                    val_i = val_offset + int(j)
                    if val_i >= len(va_idx):
                        continue
                    global_row = int(gr_cpu[j])
                    row_map = int(va_idx[val_i, 1])
                    col_map = int(va_idx[val_i, 2])
                    yt = int(lab_cpu[j])
                    yp = int(pred_cpu[j])
                    conf_pred = float(prob_cpu[j, yp])
                    conf_true = float(prob_cpu[j, yt])

                    rec: Dict[str, Any] = {
                        'val_index': val_i,
                        'train_labels_row': global_row,
                        'map_row': row_map,
                        'map_col': col_map,
                        'y_true': yt,
                        'y_pred': yp,
                        'prob_pred': conf_pred,
                        'prob_true': conf_true,
                        'figure': '',
                    }

                    if args.max_figures is None or n_err_fig < args.max_figures:
                        fname = f'e{global_row}_v{val_i}_true{yt}_pred{yp}.png'
                        fp = fig_dir / fname
                        title = (
                            f'row@{global_row} pos=({row_map},{col_map})  '
                            f'true={yt} pred={yp}  P(pred)={conf_pred:.3f}'
                        )
                        _save_misclassified_figure(
                            out_path=fp,
                            hsi_chw=hsi_cpu[j],
                            lidar_chw=lid_cpu[j],
                            rgb_chw=rgb_cpu[j] if rgb_cpu is not None else None,
                            title=title,
                        )
                        rec['figure'] = str(fp.relative_to(out_dir))
                        figure_paths.append(fp)
                        n_err_fig += 1

                    rows_records.append(rec)

            val_offset += bs

    preds = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64)
    targets = np.concatenate(targets_all) if targets_all else np.array([], dtype=np.int64)
    acc = float(np.mean(preds == targets)) if len(preds) else 0.0
    avg_loss = running_loss / max(total, 1)

    with open(out_dir / 'misclassified.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'val_index',
                'train_labels_row',
                'map_row',
                'map_col',
                'y_true',
                'y_pred',
                'prob_pred',
                'prob_true',
                'figure',
            ],
        )
        writer.writeheader()
        for r in rows_records:
            writer.writerow(r)

    summary = {
        'checkpoint': str(ckpt),
        'device': str(device),
        'val_ratio': VAL_RATIO,
        'random_seed': RANDOM_SEED,
        'batch_size': batch_size,
        'use_rgb_patches': USE_RGB_PATCHES,
        'use_center_loss_eval': USE_CENTER_LOSS,
        'rgb_source': (RGB_SOURCE or '').strip().lower(),
        'n_val': int(len(va_idx)),
        'val_loss': avg_loss,
        'val_acc': acc,
        'n_misclassified': int((preds != targets).sum()) if len(preds) else 0,
        'n_figures_written': int(n_err_fig),
        'out_dir': str(out_dir),
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if figure_paths and args.contact_sheet:
        _build_contact_sheet(
            figure_paths,
            out_dir / 'contact_sheet.png',
            args.contact_max,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not rows_records:
        print('[info] 验证集无错分样本。')
    return 0


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_ckpt = (
        repo_root / '../../autodl-tmp/classifier/0406-1718_exp_hsi_se/best_model.pt'
    ).resolve()
    p = argparse.ArgumentParser(description='验证集推理 + 错分导出')
    p.add_argument(
        '--checkpoint',
        type=str,
        default=str(default_ckpt),
        help='best_model.pt 或任意 state_dict 路径',
    )
    p.add_argument(
        '--out-dir',
        type=str,
        default='',
        help='输出目录（默认：<checkpoint 父目录>/inference_val_errors）',
    )
    p.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（默认与 param.BATCH_SIZE 一致）',
    )
    p.add_argument(
        '--max-figures',
        type=int,
        default=500,
        help='最多保存多少张错分图（默认 500；-1 表示不限制）',
    )
    p.add_argument(
        '--no-contact-sheet',
        action='store_true',
        help='不生成 contact_sheet.png',
    )
    p.add_argument(
        '--contact-max',
        type=int,
        default=24,
        help='contact_sheet 中最多拼多少张',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_figures is not None and args.max_figures < 0:
        args.max_figures = None
    args.contact_sheet = not args.no_contact_sheet

    raise SystemExit(run_inference(args))


if __name__ == '__main__':
    main()
