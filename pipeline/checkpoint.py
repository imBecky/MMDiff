import os
import re
from pathlib import Path

# 与 train_distill 一致：checkpoint 子目录名
# 断点目录名 checkpoint-<n>：n 为 epoch（1-based）；旧版可能为 global_step，仍可按数字最大者选取
_CKPT_STEP_RE = re.compile(r"^checkpoint-(\d+)$")


def get_latest_checkpoint_path(root_ckps_dir) -> str:
    """在 root_ckps_dir 下递归寻找最新 checkpoint 目录（checkpoint-<epoch>，epoch 为 1-based 编号）。"""
    root = Path(root_ckps_dir)
    if not root.exists():
        return ""
    candidates = []
    for p in root.rglob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = _CKPT_STEP_RE.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        candidates.append((step, p.stat().st_mtime, str(p)))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def ensure_model_path_parent(model_path: Path):
    """torch.save 写入 MODEL_PATH 前须存在父目录（否则报 Parent directory does not exist）。"""
    model_path.parent.mkdir(parents=True, exist_ok=True)


def save_classifier_training_state(
    optimizer,
    lr_scheduler,
    next_epoch: int,
    global_step: int,
    best_acc: float,
    best_epoch: int,
    save_dir,
    run_log_dir: str,
    run_ckps_dir: str,
):
    import torch

    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "run_log_dir": run_log_dir,
        "run_ckps_dir": run_ckps_dir,
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, "training_state.pt"))


def save_classifier_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    next_epoch: int,
    global_step: int,
    best_acc: float,
    best_epoch: int,
    save_dir: str,
    run_log_dir: str,
    run_ckps_dir: str,
):
    """保存分类器权重 + 训练状态（与 train_distill 的 pipeline+training_state 对应）。"""
    import torch

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "classifier.pt"))
    save_classifier_training_state(
        optimizer,
        lr_scheduler,
        next_epoch,
        global_step,
        best_acc,
        best_epoch,
        save_dir,
        run_log_dir,
        run_ckps_dir,
    )
