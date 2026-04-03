#!/usr/bin/env python3
"""
轻量 RGB student 蒸馏：拟合离线预计算的 teacher token（MSE + 余弦）。

依赖：先运行 `python utils/precompute_rgb_teacher_tokens.py --split train`

用法:
  python utils/train_rgb_distill.py --epochs 100 --early-stopping-patience 15 --out path/to/rgb_student.pt

TensorBoard 默认写入 param.TB_LOG_ROOT 下 rgb_student_distill_<时间戳>/；可用 --tb-dir 覆盖。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from param import (  # noqa: E402
    BATCH_SIZE,
    CLS_DIFFUSION_TIMESTEPS,
    CLS_TOKEN_DIM,
    DATA_DIR,
    FEAT_SCALES,
    PATCH_WINDOW_SIZE,
    RANDOM_SEED,
    TB_LOG_ROOT,
    TRAIN_ROT_AUGMENT_FACTOR,
    NUM_WORKERS,
)
from model.rgb_student import LightweightRgbEncoder  # noqa: E402
from pipeline.rgb_teacher_cache import mmap_tokens  # noqa: E402
from pipeline.data import (  # noqa: E402
    _apply_rot_k,
    _crop_patch_hwc,
    _random_rot_k,
    load_train_bundle,
    split_train_val_indices,
)


class _RgbDistillDataset(Dataset):
    """每条样本对应 train 划分中的一行；训练时随机 rot_k，目标为 cache[global_row, rot_k]。"""

    def __init__(
        self,
        rgb_vol: np.ndarray,
        indices: np.ndarray,
        global_rows: np.ndarray,
        cache: np.ndarray,
        training: bool,
        rot_factor: int,
    ):
        self.rgb = rgb_vol
        self.indices = np.asarray(indices, dtype=np.int64)
        self.global_rows = np.asarray(global_rows, dtype=np.int64)
        self.cache = cache
        self.training = training
        self.rot_factor = int(rot_factor)
        self.w = int(PATCH_WINDOW_SIZE)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        lab = int(self.indices[i, 0])
        row = int(self.indices[i, 1])
        col = int(self.indices[i, 2])
        gr = int(self.global_rows[i])
        if self.training and self.rot_factor > 1:
            rk = _random_rot_k(self.rot_factor)
        else:
            rk = 0
        rp = _crop_patch_hwc(self.rgb, row, col, self.w)
        if rk:
            rp = _apply_rot_k(rp, rk)
        # mmap 上的数组只读；from_numpy 需要可写 buffer，否则 PyTorch 报警
        x = np.array(np.transpose(rp, (2, 0, 1)), dtype=np.float32, copy=True)
        tgt = np.asarray(self.cache[gr, rk], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(x),
            torch.from_numpy(tgt),
            torch.tensor(gr, dtype=torch.long),
            torch.tensor(rk, dtype=torch.long),
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='RGB student 蒸馏 teacher token')
    p.add_argument('--epochs', type=int, default=100, help='最大 epoch 数（配合早停）')
    p.add_argument(
        '--early-stopping-patience',
        type=int,
        default=15,
        help='验证集 loss 连续多少 epoch 未创新低则停止；0 表示关闭早停',
    )
    p.add_argument('--batch-size', type=int, default=0, help='0 表示使用 param.BATCH_SIZE')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--cache', type=str, default='', help='rgb_teacher_tokens_train.npy 路径')
    p.add_argument('--out', type=str, default='rgb_student_distill.pt')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=-1, help='-1 使用 RANDOM_SEED')
    p.add_argument('--tb-dir', type=str, default='', help='TensorBoard 目录；默认 TB_LOG_ROOT/rgb_student_distill_<时间戳>')
    p.add_argument('--no-tb', action='store_true', help='不写 TensorBoard')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    seed = int(RANDOM_SEED if args.seed < 0 else args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cache_path = Path(args.cache) if args.cache else (Path(DATA_DIR) / 'rgb_teacher_tokens_train.npy')
    if not cache_path.is_file():
        raise FileNotFoundError(
            f'未找到 teacher 缓存 {cache_path}，请先运行 utils/precompute_rgb_teacher_tokens.py --split train'
        )
    cache = mmap_tokens(cache_path)

    _feats, rgb_vol, train_indices, _ls = load_train_bundle()

    tr_idx, va_idx, tr_pos, va_pos = split_train_val_indices(train_indices, float(args.val_ratio), seed)
    rot = int(TRAIN_ROT_AUGMENT_FACTOR)
    if rot not in (1, 2, 4):
        rot = 1

    train_ds = _RgbDistillDataset(rgb_vol, tr_idx, tr_pos, cache, training=True, rot_factor=rot)
    val_ds = _RgbDistillDataset(rgb_vol, va_idx, va_pos, cache, training=False, rot_factor=1)

    bs = int(args.batch_size) if args.batch_size > 0 else int(BATCH_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=min(8, int(NUM_WORKERS)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=min(8, int(NUM_WORKERS)),
        pin_memory=torch.cuda.is_available(),
    )

    diffusion_ts = list(CLS_DIFFUSION_TIMESTEPS)
    feat_names = list(FEAT_SCALES)
    num_tokens = len(diffusion_ts) * len(feat_names)
    d_model = int(CLS_TOKEN_DIM)

    model = LightweightRgbEncoder(
        in_ch=3,
        patch_h=int(PATCH_WINDOW_SIZE),
        patch_w=int(PATCH_WINDOW_SIZE),
        d_model=d_model,
        num_tokens=num_tokens,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def step_batch(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, tgt)
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(tgt, dim=-1)
        cos = 1.0 - (pred_n * tgt_n).sum(dim=-1).mean()
        return mse + 0.1 * cos

    best_val = float('inf')
    out_path = Path(args.out)
    patience = int(args.early_stopping_patience)
    epochs_no_improve = 0
    writer: SummaryWriter | None = None
    if not args.no_tb:
        ts = (os.environ.get('MMDIFF_RUN_TIMESTAMP') or '').strip() or datetime.now().strftime('%Y%m%d-%H%M%S')
        tb_root = Path(args.tb_dir) if args.tb_dir else Path(TB_LOG_ROOT) / f'rgb_student_distill_{ts}'
        tb_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_root))
        print(f'TensorBoard logdir: {tb_root.resolve()}  （tensorboard --logdir 指向父目录或该目录）')

    stopped_early = False
    for epoch in range(int(args.epochs)):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, tgt, _gr, _rk in tqdm(train_loader, desc=f'Epoch {epoch+1} train', leave=False):
            x = x.to(device)
            tgt = tgt.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = step_batch(pred, tgt)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * x.size(0)
            n_tr += x.size(0)
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for x, tgt, _gr, _rk in val_loader:
                x = x.to(device)
                tgt = tgt.to(device)
                pred = model(x)
                loss = step_batch(pred, tgt)
                va_loss += float(loss.item()) * x.size(0)
                n_va += x.size(0)
        va_loss /= max(n_va, 1)
        print(f'Epoch {epoch+1}: train_loss={tr_loss:.6f} val_loss={va_loss:.6f}')
        if writer is not None:
            writer.add_scalar('loss/train', tr_loss, epoch)
            writer.add_scalar('loss/val', va_loss, epoch)

        improved = va_loss < best_val - 1e-12
        if improved:
            best_val = va_loss
            epochs_no_improve = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sd = {f'rgb_student.{k}': v for k, v in model.state_dict().items()}
            torch.save(sd, out_path)
            print(f'  saved best -> {out_path}')
            if writer is not None:
                writer.add_scalar('distill/best_val', best_val, epoch)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f'早停：val_loss 连续 {patience} 个 epoch 未下降（自上次最佳起）。'
                    f' 在第 {epoch + 1} epoch 结束。'
                )
                stopped_early = True
                if writer is not None:
                    writer.add_text('distill/stop', f'early_stop epoch={epoch + 1} patience={patience}', epoch)
                break

        if writer is not None:
            writer.add_scalar('distill/epochs_without_improve', epochs_no_improve, epoch)

    if writer is not None:
        writer.flush()
        writer.close()

    print(
        f'完成。最佳 val_loss={best_val:.6f}，权重: {out_path}'
        + (f'  （早停于 epoch {epoch + 1}）' if stopped_early else '')
    )


if __name__ == '__main__':
    main()
