"""
训练开始前：将论文常用控制变量打印到控制台，并写入与 run_training 一致的 model.log（或显式路径）。

供 main.py、utils/main_compare.py 共用。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def _is_compare_run() -> bool:
    return (os.environ.get('MMDIFF_COMPARE_RUN') or '').strip().lower() in ('1', 'true', 'yes')


def ensure_run_timestamp_for_tb() -> None:
    """与 pipeline.prepare_tb_run_dir 一致：未设时间戳时用当前时刻，避免入口与 run_training 各调一次时目录不一致。"""
    if not (os.environ.get('MMDIFF_RUN_TIMESTAMP') or '').strip():
        os.environ['MMDIFF_RUN_TIMESTAMP'] = datetime.now().strftime('%m%d-%H%M')


def _normalize_resume_path(resume_checkpoint: str) -> str:
    s = (resume_checkpoint or '').strip()
    if not s:
        return ''
    p = Path(s).expanduser()
    abs_s = os.path.abspath(str(p)) if not p.is_absolute() else str(p.resolve())
    if not os.path.isdir(abs_s):
        raise FileNotFoundError(f'断点目录不存在或不是文件夹: {abs_s}')
    return abs_s


def resolve_training_model_log(no_artifacts: bool) -> Path | None:
    """与 pipeline.runner.run_training 中 log_path 规则一致（新 run / resume）。"""
    if no_artifacts:
        return None
    from param import RESUME_CHECKPOINT, TB_LOG_ROOT
    from pipeline.checkpoint import _CKPT_STEP_RE
    from pipeline.logging_utils import prepare_tb_run_dir

    run_log_dir_str = ''
    resume_ckpt = _normalize_resume_path(RESUME_CHECKPOINT)
    if resume_ckpt:
        ts_path = os.path.join(resume_ckpt, 'training_state.pt')
        if os.path.isfile(ts_path):
            import torch

            resume_ts = torch.load(ts_path, map_location='cpu')
            run_log_dir_str = resume_ts.get('run_log_dir') or ''
        else:
            ckpt_path = Path(resume_ckpt)
            m = _CKPT_STEP_RE.match(ckpt_path.name)
            if m:
                run_log_dir_str = str(TB_LOG_ROOT / ckpt_path.parent.name)
            elif ckpt_path.name == 'final' and ckpt_path.parent.is_dir():
                run_log_dir_str = str(TB_LOG_ROOT / ckpt_path.parent.name)

    if run_log_dir_str and os.path.isdir(run_log_dir_str):
        return Path(run_log_dir_str) / 'model.log'
    run_dir = prepare_tb_run_dir()
    return run_dir / 'model.log'


def build_control_variable_lines() -> list[str]:
    import param

    sn = (param.SCHEDULER_NAME or '').strip().lower()
    if sn in ('cosine', 'cosine_annealing'):
        sched_line = (
            f"scheduler={param.SCHEDULER_NAME} | "
            f"eta_min_ratio={param.SCHEDULER_COSINE_ETA_MIN_RATIO} "
            f"warmup_ratio={param.SCHEDULER_COSINE_WARMUP_RATIO} "
            f"warmup_steps={param.SCHEDULER_COSINE_WARMUP_STEPS}"
        )
    else:
        sched_line = (
            f"scheduler={param.SCHEDULER_NAME} | "
            f"step_ratios={list(param.SCHED_STEP_RATIOS)} gammas={list(param.SCHED_GAMMAS)}"
        )

    combo = (os.environ.get('MMDIFF_MODALITY_COMBO') or '').strip() or param.MODALITY_COMBO
    exp_tag = (os.environ.get('MMDIFF_EXPERIMENT_TAG') or '').strip()
    compare_model = (os.environ.get('MMDIFF_COMPARE_MODEL') or '').strip()
    run_ts = (os.environ.get('MMDIFF_RUN_TIMESTAMP') or '').strip()
    compare = _is_compare_run()
    title = (
        '========== 对比实验控制变量（论文常用登记项） =========='
        if compare
        else '========== 主训练控制变量（论文常用登记项） =========='
    )
    if compare:
        id_line = (
            f'实验标识 | compare_model={compare_model} experiment_tag={exp_tag or "(未设)"} '
            f'run_timestamp={run_ts or "(未设)"} modality_combo={combo}'
        )
    else:
        id_line = (
            f'实验标识 | experiment_tag={exp_tag or "(未设)"} '
            f'run_timestamp={run_ts or "(未设)"} modality_combo={combo}'
        )

    return [
        title,
        id_line,
        (
            f'训练超参 | batch_size={param.BATCH_SIZE} epochs={param.NUM_EPOCHS} '
            f'lr={param.LEARNING_RATE:g} weight_decay={param.WEIGHT_DECAY:g} '
            f'betas={param.OPTIMIZER_BETAS} clip_grad_norm={param.CLIP_GRAD_NORM}'
        ),
        sched_line,
        (
            f'数据与划分 | patch_size={param.PATCH_WINDOW_SIZE} hsi_channels={param.HSI_CHANNELS} '
            f'n_cls={param.NUM_CLASSES} val_ratio={param.VAL_RATIO} random_seed={param.RANDOM_SEED} '
            f'rot_aug_factor={param.TRAIN_ROT_AUGMENT_FACTOR} num_workers={param.NUM_WORKERS}'
        ),
        (
            f'验证/早停 | eval_val_start_epoch={param.EVAL_VAL_START_EPOCH} '
            f'eval_min_train_acc={param.EVAL_MIN_TRAIN_ACC} eval_interval_epochs={param.EVAL_INTERVAL_EPOCHS} '
            f'early_stopping_patience={param.EARLY_STOPPING_PATIENCE}'
        ),
        f'data_dir={param.DATA_DIR}',
        f'resume_checkpoint={(param.RESUME_CHECKPOINT or "").strip() or "(无)"}',
        param.MULTIMODAL_ABLATION_LOG_LINE,
        '========== （控制变量摘要结束） ==========',
    ]


def emit_training_control_variable_summary(
    *,
    no_artifacts: bool = False,
    log_file: Path | None = None,
) -> None:
    """打印到控制台；非 no-artifacts 时写入 log_file 或按 run_training 解析的 model.log。"""
    ensure_run_timestamp_for_tb()
    text = '\n'.join(build_control_variable_lines())
    print(text, flush=True)
    if no_artifacts:
        return
    target = log_file if log_file is not None else resolve_training_model_log(False)
    if target is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    from pipeline.logging_utils import get_logger

    get_logger(target).info('%s', text)
