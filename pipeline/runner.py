"""
分类训练主流程：数据与循环与「模型如何构建」解耦，通过 create_classifier(opt, diffusion) 注入。
其中 diffusion 已废弃，恒为 None（RGB 仅 LightweightRgbEncoder）。
替换你自己的 model 包时，只需实现与当前 MultimodalClassifier 相同的外部契约：
  - forward(data_dict)，可选 return_center_logits=True（当 param.USE_CENTER_LOSS）
  - loss_func, optimizer, 可选 exp_lr_scheduler
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from param import (
    BATCH_SIZE,
    BEST_MODEL_FILENAME,
    CHECK_PROJECTION_GRAD,
    CHECK_PROJECTION_GRAD_INTERVAL,
    CKPS_DIR,
    EARLY_STOPPING_PATIENCE,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    LOG_PATH,
    MULTIMODAL_ABLATION_LOG_LINE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
    RESUME_CHECKPOINT,
    RGB_STUDENT_CHECKPOINT,
    SAVE_EVERY_EPOCH,
    TB_LOG_ROOT,
    TRAIN_QUICK_VERIFY,
    TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
    USE_CENTER_LOSS,
    USE_RGB_PATCHES,
    USE_SUPCON,
    VAL_RATIO,
    opt,
)

from .checkpoint import (
    _CKPT_STEP_RE,
    save_classifier_checkpoint,
)
from .data import (
    batch_to_dict,
    build_dataloaders,
    build_test_loader,
    load_rgb_hr_meta,
    load_rgb_hr_volume,
    load_test_indices_shifted,
    load_train_bundle,
    split_train_val_indices,
    subset_train_indices_balanced,
)
from .logging_utils import (
    get_console_logger,
    get_logger,
    get_summary_writer,
    log_and_print,
    log_config,
    log_model_and_training_detail,
    maybe_attach_forward_dataflow_trace,
    prepare_tb_run_dir,
    save_confusion_detail_log,
)
from .loop import (
    compute_classification_loss,
    evaluate,
    log_projection_gradients,
    train_one_epoch,
)
from .classification_metrics import accuracies

CreateClassifierFn = Callable[[Any, Any], torch.nn.Module]


def _state_dict_shape_compatible(model: torch.nn.Module, sd: dict) -> dict:
    """仅加载与当前模型同形状的权重；支持 checkpoint 带 module. 前缀；避免 rgb_student 与 token_dim 不一致时崩。"""
    target = model.state_dict()
    good: dict = {}
    for k, v in sd.items():
        if k in target and v.shape == target[k].shape:
            good[k] = v
            continue
        if k.startswith("module."):
            k2 = k[7:]
            if k2 in target and v.shape == target[k2].shape:
                good[k2] = v
    return good


def _load_rgb_student_checkpoint_filtered(
    model: torch.nn.Module, sd: dict, logger, ck_path: Path
) -> None:
    """只加载形状一致的键；绝不因 fc 等维度变化抛 RuntimeError。"""
    compatible = _state_dict_shape_compatible(model, sd)
    tgt = model.state_dict()
    skipped = []
    for k, v in sd.items():
        tk = k[7:] if k.startswith("module.") else k
        if tk not in tgt:
            continue
        if v.shape != tgt[tk].shape:
            skipped.append(tk)
    if skipped:
        logger.warning(
            "RGB student checkpoint 与当前模型形状不一致，已跳过 %d 个键（token_dim 等与蒸馏不一致时 fc 会重初始化）: %s%s",
            len(skipped),
            ", ".join(sorted(skipped)[:12]),
            " ..." if len(skipped) > 12 else "",
        )
    model.load_state_dict(compatible, strict=False)
    logger.info("已加载 RGB student 权重（形状匹配键）: %s", ck_path)


@dataclass
class TrainingRunOptions:
    """训练运行选项（由 main 解析传入）。"""
    no_artifacts: bool = False
    save_conf_detail: bool = True


def _is_compare_run() -> bool:
    return (os.environ.get('MMDIFF_COMPARE_RUN') or '').strip().lower() in ('1', 'true', 'yes')


def _compare_run_saves_checkpoints() -> bool:
    """对比实验是否写磁盘断点。默认否；设 MMDIFF_COMPARE_SAVE_CKPT=1 时与主训练一致落盘。"""
    return (os.environ.get('MMDIFF_COMPARE_SAVE_CKPT') or '').strip().lower() in ('1', 'true', 'yes')


def _normalize_resume_path(resume_checkpoint: str) -> str:
    s = (resume_checkpoint or '').strip()
    if not s:
        return ''
    p = Path(s).expanduser()
    abs_s = os.path.abspath(str(p)) if not p.is_absolute() else str(p.resolve())
    if not os.path.isdir(abs_s):
        raise FileNotFoundError(f'断点目录不存在或不是文件夹: {abs_s}')
    return abs_s


def _scale_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any],
    factor: float,
) -> None:
    """缩放当前各 param_group 的 lr；若有 LambdaLR 等基于 base_lrs 的调度器，同步缩放 base_lrs。"""
    for g in optimizer.param_groups:
        g['lr'] = float(g['lr']) * factor
    if lr_scheduler is not None and hasattr(lr_scheduler, 'base_lrs'):
        lr_scheduler.base_lrs = [float(b) * factor for b in lr_scheduler.base_lrs]


def run_training(
    create_classifier: CreateClassifierFn,
    run_options: Optional[TrainingRunOptions] = None,
) -> None:
    """完整训练 + 验证 + 测试；模型由 create_classifier(opt, diffusion) 提供。"""
    opts = run_options or TrainingRunOptions()
    no_artifacts = bool(opts.no_artifacts)
    save_conf_detail = bool(opts.save_conf_detail)

    resume_ckpt = _normalize_resume_path(RESUME_CHECKPOINT)
    resume_ts = None
    run_log_dir_str = ''
    run_ckps_dir_str = ''
    start_epoch = 0
    resume_global_step = 0
    resume_best_acc = -1.0
    resume_best_epoch = 0

    if resume_ckpt:
        ts_path = os.path.join(resume_ckpt, 'training_state.pt')
        if os.path.isfile(ts_path):
            resume_ts = torch.load(ts_path, map_location='cpu')
            run_log_dir_str = resume_ts.get('run_log_dir') or ''
            run_ckps_dir_str = resume_ts.get('run_ckps_dir') or ''
            start_epoch = int(resume_ts.get('next_epoch', resume_ts.get('epoch', 0)))
            resume_global_step = int(resume_ts.get('global_step', 0))
            resume_best_acc = float(resume_ts.get('best_acc', -1.0))
            resume_best_epoch = int(resume_ts.get('best_epoch', 0))
        else:
            ckpt_path = Path(resume_ckpt)
            m = _CKPT_STEP_RE.match(ckpt_path.name)
            if m:
                run_ckps_dir_str = str(ckpt_path.parent)
                inferred_tag = ckpt_path.parent.name
                run_log_dir_str = str(TB_LOG_ROOT / inferred_tag)
                resume_global_step = int(m.group(1))
                print(
                    f'[恢复] 未找到 training_state.pt，已从目录名推断 global_step={resume_global_step}，'
                    f'TB 目录: {run_log_dir_str}'
                )
            else:
                ckpt_path = Path(resume_ckpt)
                if ckpt_path.name == 'final' and ckpt_path.parent.is_dir():
                    run_ckps_dir_str = str(ckpt_path.parent)
                    inferred_tag = ckpt_path.parent.name
                    run_log_dir_str = str(TB_LOG_ROOT / inferred_tag)
                    print(
                        f'[恢复] 未找到 training_state.pt，已从 final 推断 run_ckps_dir={run_ckps_dir_str}，'
                        f'TB: {run_log_dir_str}'
                    )
                else:
                    print('[恢复] 仅将加载 classifier.pt / model.pt，epoch 与优化器从 0 开始')

    if no_artifacts:
        run_log_dir_str = ''
        run_ckps_dir_str = ''

    if no_artifacts:
        run_dir = None
        log_path = None
        logger = get_console_logger()
        writer = None
    elif run_log_dir_str and os.path.isdir(run_log_dir_str):
        run_dir = Path(run_log_dir_str)
        log_path = run_dir / 'model.log'
        logger = get_logger(log_path)
        writer = get_summary_writer(logger, run_dir)
    else:
        run_dir = prepare_tb_run_dir()
        run_log_dir_str = str(run_dir)
        if resume_ts is not None:
            run_ckps_dir_str = ''
        log_path = run_dir / 'model.log'
        logger = get_logger(log_path)
        writer = get_summary_writer(logger, run_dir)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    compare_run = _is_compare_run()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feats_vol, rgb_vol, train_indices, label_shift = load_train_bundle()
    rgb_hr_vol = None
    hr_rh = 1
    hr_rw = 1
    if USE_RGB_PATCHES:
        rgb_hr_vol = load_rgb_hr_volume()
        _hr_meta = load_rgb_hr_meta()
        hr_rh = int(_hr_meta['rh'])
        hr_rw = int(_hr_meta['rw'])
    if TRAIN_QUICK_VERIFY:
        n_before = len(train_indices)
        train_indices = subset_train_indices_balanced(
            train_indices,
            TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
            RANDOM_SEED,
            NUM_CLASSES,
        )
        logger.info(
            '[Quick verify] 训练索引分层子集: %d -> %d（每类至多 %d）',
            n_before,
            len(train_indices),
            TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
        )
    tr_idx, va_idx, tr_pos, va_pos = split_train_val_indices(train_indices, VAL_RATIO, RANDOM_SEED)
    defer_test_load = 0 < VAL_RATIO < 1.0
    test_idx = None
    if not defer_test_load:
        test_idx = load_test_indices_shifted(label_shift)
    if defer_test_load:
        logger.info(
            '延迟加载 test_labels（仅 final eval）；训练期选优使用验证集 (VAL_RATIO=%s)',
            VAL_RATIO,
        )
    train_loader, val_loader, test_loader = build_dataloaders(
        feats_vol,
        rgb_vol,
        tr_idx,
        va_idx,
        test_idx,
        BATCH_SIZE,
        defer_test=defer_test_load,
        train_global_rows=tr_pos,
        val_global_rows=va_pos,
        rgb_strict_view=bool(USE_RGB_PATCHES),
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
    )
    opt['len_train_dataloader'] = len(train_loader)
    # 续训：用首次训练时的总 step 数计算 LambdaLR 边界，避免仅因增大 NUM_EPOCHS 而重算 b1/b2 导致 LR 跳变
    if resume_ts is not None:
        slr = resume_ts.get('scheduler_lr_total_steps')
        if slr is not None:
            opt['scheduler_lr_total_steps'] = int(slr)
        elif not opt.get('scheduler_lr_total_steps'):
            logger.warning(
                '断点 training_state 无 scheduler_lr_total_steps；若续训总 epoch 与首次不同，'
                '请设环境变量 MMDIFF_SCHEDULER_LR_TOTAL_STEPS=首次训练时的 (n_epoch×train_batches)，否则 LR 边界可能与 last_epoch 错位。'
            )
        if opt.get('scheduler_lr_total_steps'):
            logger.info(
                '续训 LR 边界 scheduler_lr_total_steps=%s（LambdaLR 衰减比例仍按此总步数计算）',
                opt['scheduler_lr_total_steps'],
            )
    early_stop_patience = max(0, int(EARLY_STOPPING_PATIENCE))
    if early_stop_patience > 0:
        logger.info(
            'Early stopping: patience=%d（按验证次数计，OA 须严格优于历史最优才重置）',
            early_stop_patience,
        )

    if compare_run:
        logger.info(
            'VAL_RATIO=%s | train_batches=%d val_batches=%s | compare_model=%s',
            VAL_RATIO,
            len(train_loader),
            len(val_loader) if val_loader is not None else None,
            os.environ.get('MMDIFF_COMPARE_MODEL', ''),
        )
    else:
        logger.info(
            'VAL_RATIO=%s | train_batches=%d val_batches=%s | rgb=patch_encoder',
            VAL_RATIO,
            len(train_loader),
            len(val_loader) if val_loader is not None else None,
        )
        logger.info('%s', MULTIMODAL_ABLATION_LOG_LINE)
    if CHECK_PROJECTION_GRAD:
        logger.info(
            'CHECK_PROJECTION_GRAD=True，将在 backward 后按间隔记录投影梯度（间隔=%d）',
            CHECK_PROJECTION_GRAD_INTERVAL,
        )
    logger.info(
        '数据 | train_patches=%s train_rgb_patches=%s | train_labels 行数=%s val=%s',
        feats_vol.shape,
        rgb_vol.shape if rgb_vol is not None else None,
        tr_idx.shape,
        va_idx.shape if va_idx is not None else None,
    )
    log_config(
        logger,
        writer,
        device,
        None,
        None,
        tr_idx[:, 0],
        None,
        None,
        test_idx[:, 0] if test_idx is not None else None,
        train_rgb=None,
        test_rgb=None,
        log_file_path=log_path,
    )

    if not no_artifacts:
        if compare_run and not _compare_run_saves_checkpoints():
            run_ckps_dir_str = ''
            logger.info(
                '对比实验：不保存 classifier 断点（best/周期/final/协议侧权重）；'
                '若需落盘请设 MMDIFF_COMPARE_SAVE_CKPT=1'
            )
        elif not run_ckps_dir_str:
            CKPS_DIR.mkdir(parents=True, exist_ok=True)
            run_ckps_dir_str = str(CKPS_DIR / run_dir.name)
            os.makedirs(run_ckps_dir_str, exist_ok=True)
            logger.info('Checkpoint 目录: %s', run_ckps_dir_str)
        else:
            os.makedirs(run_ckps_dir_str, exist_ok=True)
            logger.info('Checkpoint 目录: %s', run_ckps_dir_str)
    else:
        logger.info('无文件产物模式（--no-artifacts）：不创建 TB/断点目录，不写 TensorBoard、周期断点、final')

    best_path = (
        os.path.join(run_ckps_dir_str, BEST_MODEL_FILENAME) if run_ckps_dir_str else None
    )

    if compare_run:
        from .two_branch_protocol import run_two_branch_protocol_if_needed

        if run_two_branch_protocol_if_needed(
            create_classifier=create_classifier,
            compare_run=compare_run,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feats_vol=feats_vol,
            rgb_vol=rgb_vol,
            label_shift=label_shift,
            defer_test_load=defer_test_load,
            logger=logger,
            writer=writer,
            run_ckps_dir_str=run_ckps_dir_str or '',
            best_path=best_path,
            no_artifacts=no_artifacts,
            save_conf_detail=save_conf_detail,
            run_log_dir_str=run_log_dir_str,
        ):
            return

        from .dfinet_protocol import run_dfinet_protocol_if_needed

        if run_dfinet_protocol_if_needed(
            create_classifier=create_classifier,
            compare_run=compare_run,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            feats_vol=feats_vol,
            rgb_vol=rgb_vol,
            label_shift=label_shift,
            defer_test_load=defer_test_load,
            logger=logger,
            writer=writer,
            run_ckps_dir_str=run_ckps_dir_str or '',
            best_path=best_path,
            no_artifacts=no_artifacts,
            save_conf_detail=save_conf_detail,
            run_log_dir_str=run_log_dir_str,
        ):
            return

    diffusion = None
    if not compare_run:
        logger.info('RGB 分支：LightweightRgbEncoder（patch），不使用扩散模型')
    model = create_classifier(opt, diffusion).to(device)

    loss_fn = model.loss_func
    optimizer = model.optimizer
    lr_scheduler = getattr(model, 'exp_lr_scheduler', None)
    clip_grad = float(opt.get('train', {}).get('clip_grad_norm', 0) or 0)

    if resume_ckpt:
        clf_pt = os.path.join(resume_ckpt, 'classifier.pt')
        if not os.path.isfile(clf_pt):
            alt = os.path.join(resume_ckpt, 'model.pt')
            clf_pt = alt if os.path.isfile(alt) else clf_pt
        if os.path.isfile(clf_pt):
            model.load_state_dict(torch.load(clf_pt, map_location=device), strict=True)
            logger.info('已从断点加载分类器权重: %s', clf_pt)
        else:
            logger.warning('断点目录中未找到 classifier.pt / model.pt，使用随机初始化')

        if resume_ts is not None:
            if resume_ts.get('optimizer'):
                optimizer.load_state_dict(resume_ts['optimizer'])
            sched_sd = resume_ts.get('scheduler')
            if lr_scheduler is not None and sched_sd is not None:
                lr_scheduler.load_state_dict(sched_sd)
            logger.info(
                '已恢复优化器与学习率调度 | next_epoch=%d global_step=%d best_acc=%.4f best_epoch=%d',
                start_epoch,
                resume_global_step,
                float(resume_ts.get('best_acc', -1.0)),
                int(resume_ts.get('best_epoch', 0)),
            )

        # 从断点续训：在已加载/初始化的 lr 上再打 0.5（不依赖改 param）
        _scale_optimizer_learning_rate(optimizer, lr_scheduler, 0.5)
        logger.info(
            '续训学习率缩放 ×0.5 | 当前 param_group lr=%s',
            [float(g['lr']) for g in optimizer.param_groups],
        )
    elif not compare_run:
        ck = (RGB_STUDENT_CHECKPOINT or '').strip()
        if ck:
            ck_path = Path(ck)
            if ck_path.is_file():
                sd = torch.load(str(ck_path), map_location=device)
                _load_rgb_student_checkpoint_filtered(model, sd, logger, ck_path)
            else:
                logger.warning('MMDIFF_RGB_STUDENT_CHECKPOINT 不存在: %s', ck_path)
        else:
            logger.info('MMDIFF_RGB_STUDENT_CHECKPOINT 未设置：RGB student 保持随机初始化（消融 random）')

    if (
        not compare_run
        and not resume_ckpt
        and (os.environ.get('MMDIFF_FREEZE_RGB_STUDENT') or '').strip().lower()
        in ('1', 'true', 'yes', 'on')
    ):
        rs = getattr(model, 'rgb_student', None)
        if rs is not None:
            rs.requires_grad_(False)
            model.refresh_optimizer_after_param_freeze()
            optimizer = model.optimizer
            lr_scheduler = getattr(model, 'exp_lr_scheduler', None)
            logger.info('RGB student 已冻结（MMDIFF_FREEZE_RGB_STUDENT=1），优化器已重建（不含 rgb_student）')
        else:
            logger.warning('MMDIFF_FREEZE_RGB_STUDENT=1 但模型无 rgb_student，忽略')

    log_model_and_training_detail(logger, writer, model, opt, clip_grad, diffusion)
    maybe_attach_forward_dataflow_trace(logger, model)

    best_acc = resume_best_acc
    best_epoch = resume_best_epoch
    global_step = resume_global_step
    selection_loader = val_loader
    selection_split = 'val'
    if val_loader is None:
        logger.warning('VAL_RATIO=0：无验证集，将用测试集选 best（仅便于跑通，正式实验请设 VAL_RATIO>0）')
        selection_loader = test_loader
        selection_split = 'test'

    last_eval_epoch = None
    last_ovr_acc = 0.0
    last_eval_sel_loss = float('nan')
    last_eval_sel_acc = float('nan')
    last_kappa = 0.0
    last_s_sqr = 0.0
    last_aa = 0.0

    best_state_dict = None

    # 周期性 checkpoint-<n> 仅从总 epoch 数的后 20% 起写（epoch 为 0-based）；best / final 仍照常
    periodic_ckpt_min_epoch = max(0, int(NUM_EPOCHS * 0.5))

    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='Epochs')
    last_epoch_0based = start_epoch - 1
    early_stop_evals_without_gain = 0

    try:
        for epoch in epoch_bar:
            last_epoch_0based = epoch
            epoch_start = time.perf_counter()
            eval_sel_loss = float('nan')
            eval_sel_acc = float('nan')
            ovr_acc = 0.0
            kappa = 0.0
            s_sqr = 0.0
            aa = 0.0

            train_loss, train_acc, global_step = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                logger,
                writer=writer,
                epoch=epoch,
                global_step=global_step,
                clip_grad_norm=clip_grad,
                lr_scheduler=lr_scheduler,
            )
            run_eval = epoch >= EVAL_VAL_START_EPOCH and selection_loader is not None
            run_selection = run_eval and train_acc >= EVAL_MIN_TRAIN_ACC

            should_run_eval = run_selection and (
                EVAL_INTERVAL_EPOCHS <= 0
                or last_eval_epoch is None
                or (epoch - last_eval_epoch) >= EVAL_INTERVAL_EPOCHS
            )

            if should_run_eval:
                prev_best_acc = best_acc
                use_center = bool(USE_CENTER_LOSS)
                _, _, conf, eval_sel_loss, eval_sel_acc = evaluate(
                    model,
                    selection_loader,
                    loss_fn,
                    device,
                    NUM_CLASSES,
                    logger,
                    writer=writer,
                    epoch=epoch,
                    split=selection_split,
                    use_center_logits=use_center,
                )
                ovr_acc, usr_acc, prod_acc, kappa, s_sqr, aa = accuracies(conf)
                last_eval_epoch = epoch
                last_ovr_acc = ovr_acc
                last_eval_sel_loss = eval_sel_loss
                last_eval_sel_acc = eval_sel_acc
                last_kappa = kappa
                last_s_sqr = s_sqr
                last_aa = aa
            elif run_selection:
                ovr_acc = last_ovr_acc
                eval_sel_loss = last_eval_sel_loss
                eval_sel_acc = last_eval_sel_acc
                kappa = last_kappa
                s_sqr = last_s_sqr
                aa = last_aa
            epoch_seconds = time.perf_counter() - epoch_start

            if writer is not None:
                writer.add_scalar('train/epoch_loss', train_loss, epoch + 1)
                writer.add_scalar('train/epoch_acc', train_acc, epoch + 1)
                if should_run_eval:
                    writer.add_scalar(f'{selection_split}/overall_accuracy', ovr_acc, epoch + 1)
                    writer.add_scalar(f'{selection_split}/average_accuracy', aa, epoch + 1)
                    writer.add_scalar(f'{selection_split}/kappa', kappa, epoch + 1)
                    writer.add_scalar(f'{selection_split}/kappa_variance', s_sqr, epoch + 1)
                writer.add_scalar('timing/epoch_seconds', epoch_seconds, epoch + 1)
            if run_selection and should_run_eval:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f %s_loss=%.4f %s_acc=%.4f ovr_acc=%.4f aa=%.4f kappa=%.4f epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    eval_sel_loss,
                    selection_split,
                    eval_sel_acc,
                    ovr_acc,
                    aa,
                    kappa,
                    epoch_seconds,
                )
            elif run_selection:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f (skip %s: 每 %d epoch eval) last_ovr_acc=%.4f last_aa=%.4f epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    EVAL_INTERVAL_EPOCHS,
                    last_ovr_acc,
                    last_aa,
                    epoch_seconds,
                )
            elif run_eval:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f (skip %s: train_acc < %.0f%%) epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    100 * EVAL_MIN_TRAIN_ACC,
                    epoch_seconds,
                )
            else:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f (skip %s until epoch>=%d) epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    EVAL_VAL_START_EPOCH,
                    epoch_seconds,
                )

            if should_run_eval and ovr_acc >= best_acc:
                if best_path:
                    torch.save(model.state_dict(), best_path)
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_acc = ovr_acc
                best_epoch = epoch
                logger.info(
                    'Saved new best model at epoch %03d with %s overall accuracy %.4f',
                    epoch + 1,
                    selection_split,
                    ovr_acc,
                )
                if best_path:
                    logger.info('  best 权重: %s', best_path)
                if writer is not None:
                    writer.add_scalar(f'best/{selection_split}_overall_accuracy', best_acc, epoch + 1)
                    writer.add_scalar(f'best/{selection_split}_average_accuracy', aa, epoch + 1)

            if should_run_eval and early_stop_patience > 0:
                if ovr_acc > prev_best_acc + 1e-12:
                    early_stop_evals_without_gain = 0
                else:
                    early_stop_evals_without_gain += 1
                    logger.info(
                        'Early stopping: %d/%d 次验证 OA 未严格提升（当前 oa=%.6f，本轮验证前 best=%.6f）',
                        early_stop_evals_without_gain,
                        early_stop_patience,
                        ovr_acc,
                        prev_best_acc,
                    )
                    if early_stop_evals_without_gain >= early_stop_patience:
                        logger.info(
                            'Early stopping: 已达 patience=%d，结束训练（最后完成 epoch=%d）',
                            early_stop_patience,
                            epoch + 1,
                        )
                        if writer is not None:
                            writer.add_scalar('train/early_stopped', 1.0, epoch + 1)
                        break

            # 仅在已满足验证且 train_acc 过门槛时按间隔保存（与 run_selection 一致）；且仅在总 epoch 后 20%
            if (
                run_selection
                and SAVE_EVERY_EPOCH > 0
                and run_ckps_dir_str
                and epoch >= periodic_ckpt_min_epoch
                and (epoch + 1) % SAVE_EVERY_EPOCH == 0
            ):
                ep_ckpt = os.path.join(run_ckps_dir_str, f'checkpoint-{epoch + 1}')
                save_classifier_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch + 1,
                    global_step,
                    best_acc,
                    best_epoch,
                    ep_ckpt,
                    run_log_dir_str,
                    run_ckps_dir_str,
                )
                logger.info(
                    '断点已保存至 %s (next_epoch=%d, 每 %d epoch 一次；仅 epoch≥%d 即后 20%%；且需 train_acc≥%.0f%%)',
                    ep_ckpt,
                    epoch + 1,
                    SAVE_EVERY_EPOCH,
                    periodic_ckpt_min_epoch + 1,
                    100 * EVAL_MIN_TRAIN_ACC,
                )

            sel_loss_str = f'{eval_sel_loss:.4f}' if run_selection else '-'
            epoch_bar.set_postfix(
                train_loss=f'{train_loss:.4f}',
                train_acc=f'{train_acc:.4f}',
                val_loss=sel_loss_str,
                best_acc=f'{np.round(100 * max(best_acc, 0.0), 2)}%',
                best_epoch=best_epoch + 1,
            )

        final_next_epoch = last_epoch_0based + 1
        if final_next_epoch < start_epoch:
            final_next_epoch = start_epoch

        if run_ckps_dir_str and best_path and not os.path.isfile(best_path):
            torch.save(model.state_dict(), best_path)
            best_epoch = last_epoch_0based
            logger.warning(
                '训练过程中未出现新的 best 记录，已将最后一轮权重写入 %s',
                best_path,
            )

        if run_ckps_dir_str:
            final_dir = os.path.join(run_ckps_dir_str, 'final')
            save_classifier_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                final_next_epoch,
                global_step,
                best_acc,
                best_epoch,
                final_dir,
                run_log_dir_str,
                run_ckps_dir_str,
            )
            logger.info(
                '训练结束，最终断点已保存至 %s（next_epoch=%d，含 classifier.pt 与 training_state.pt）',
                final_dir,
                final_next_epoch,
            )

        best_model = create_classifier(opt, diffusion).to(device)
        loaded = False
        if best_path and os.path.isfile(best_path):
            try:
                best_model.load_state_dict(
                    torch.load(best_path, map_location=device), strict=True
                )
                loaded = True
                logger.info('Final test 使用 best 权重: %s', best_path)
            except RuntimeError as e:
                logger.warning(
                    'best 权重与当前模型结构不一致，尝试内存或最后一轮权重: %s',
                    e,
                )
        if not loaded:
            if best_state_dict is not None:
                best_model.load_state_dict(
                    {k: v.to(device) for k, v in best_state_dict.items()}
                )
                logger.info('Final test 使用内存中记录的 best 权重')
            else:
                best_model.load_state_dict(model.state_dict())
                logger.info('Final test 使用最后一轮 model 权重')

        use_center_final = bool(USE_CENTER_LOSS)
        if test_loader is None:
            logger.info('加载测试集（final evaluation）...')
            ti = load_test_indices_shifted(label_shift)
            test_loader = build_test_loader(
                feats_vol,
                rgb_vol,
                ti,
                BATCH_SIZE,
                rgb_strict_view=bool(USE_RGB_PATCHES),
                rgb_hr_vol=rgb_hr_vol,
                hr_rh=hr_rh,
                hr_rw=hr_rw,
            )
        preds_final, targets_final, conf_final, eval_loss_final, eval_acc_final = evaluate(
            best_model,
            test_loader,
            loss_fn,
            device,
            NUM_CLASSES,
            logger,
            writer=writer,
            epoch=best_epoch,
            split='test',
            use_center_logits=use_center_final,
        )
        ovr_acc_final, usr_acc, prod_acc, kappa, s_sqr, aa_final = accuracies(conf_final)

        n_test = int(len(targets_final))
        uniq_true = np.unique(targets_final)
        uniq_pred = np.unique(preds_final)
        row_sum = conf_final.sum(axis=1)
        n_classes_present = int(np.sum(row_sum > 0))
        logger.info(
            'Final test 诊断 | 样本数=%d | 真值类别数=%d unique=%s | 预测类别数=%d unique=%s',
            n_test,
            n_classes_present,
            uniq_true.tolist(),
            len(uniq_pred),
            uniq_pred.tolist(),
        )
        if n_classes_present < 2:
            logger.warning(
                '测试集真值仅覆盖 %d 个类别，整体准确率 OA=%.4f 不能说明泛化；'
                '若仅单类真值，模型恒预测该类即可 OA=100%%。请核对 test_labels.npy、'
                'label_shift 与 data_prepare。',
                n_classes_present,
                ovr_acc_final,
            )
        elif n_test < 50:
            logger.warning(
                '测试样本数较少（n=%d），指标波动大，建议以完整 test 集为准。',
                n_test,
            )

        if save_conf_detail and run_log_dir_str:
            conf_log_path = Path(run_log_dir_str) / 'conf_detail.log'
            save_confusion_detail_log(conf_log_path, conf_final, NUM_CLASSES)
            logger.info('混淆矩阵与误分类对已写入 %s', conf_log_path)
        elif save_conf_detail and not run_log_dir_str:
            logger.info('未写入 conf_detail（无 run 目录，例如 --no-artifacts）')

        if writer is not None:
            writer.add_scalar('final/test_loss', eval_loss_final, best_epoch + 1)
            writer.add_scalar('final/test_acc', eval_acc_final, best_epoch + 1)
            writer.add_scalar('final/overall_accuracy', ovr_acc_final, best_epoch + 1)
            writer.add_scalar('final/average_accuracy', aa_final, best_epoch + 1)
            writer.add_scalar('final/kappa', kappa, best_epoch + 1)
            writer.add_text(
                'final/summary',
                '\n'.join([
                    f'best_epoch: {best_epoch + 1}',
                    f'test_accuracy: {np.round(100 * ovr_acc_final, 2)}%',
                    f'average_accuracy: {np.round(100 * aa_final, 2)}',
                    f'user_accuracy: {np.round(100 * usr_acc, 2)}',
                    f'producer_accuracy: {np.round(100 * prod_acc, 2)}',
                    f'kappa: {np.round(kappa, 4)}',
                    f'kappa_variance: {np.round(s_sqr, 6)}',
                ]),
            )

        log_and_print(logger, 'Best epoch is', best_epoch + 1)
        log_and_print(logger, 'Test accuracy is', np.round(100 * ovr_acc_final, 2), '%')
        log_and_print(logger, 'Average accuracy (AA) is', np.round(100 * aa_final, 2), '%')
        log_and_print(logger, 'User accuracy is', np.round(100 * usr_acc, 2))
        log_and_print(logger, 'Producer accuracy is', np.round(100 * prod_acc, 2))
        log_and_print(logger, 'Kappa coefficient is', np.round(kappa, 4))
        log_and_print(logger, 'Kappa variance is', np.round(s_sqr, 6))

        if run_log_dir_str:
            summary_path = Path(run_log_dir_str) / 'metrics_summary.json'
            payload = {
                'experiment_tag': os.environ.get('MMDIFF_EXPERIMENT_TAG', ''),
                'modality_combo': opt.get('model_cls', {}).get('modality_combo', ''),
                'enabled_modalities': opt.get('model_cls', {}).get('enabled_modalities', []),
                'compare_run': (os.environ.get('MMDIFF_COMPARE_RUN') or '').strip().lower() in ('1', 'true', 'yes'),
                'compare_model': os.environ.get('MMDIFF_COMPARE_MODEL', ''),
                'best_epoch': int(best_epoch + 1),
                'oa': float(ovr_acc_final),
                'aa': float(aa_final),
                'kappa': float(kappa),
                'oa_percent': float(np.round(100 * ovr_acc_final, 4)),
                'aa_percent': float(np.round(100 * aa_final, 4)),
                'kappa_rounded': float(np.round(kappa, 6)),
            }
            summary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            logger.info('metrics_summary 已写入 %s', summary_path)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def verify_projection_gradients(create_classifier: CreateClassifierFn) -> None:
    """单 batch 前向+反向，检查各模态投影是否有非零梯度；用于快速验证建图（退出码 0 通过，1 失败）。"""
    log_path = Path(LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_path)
    logger.info('[verify_projection_grad] %s', MULTIMODAL_ABLATION_LOG_LINE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fv, rv, tr_ind, ls = load_train_bundle()
    rgb_hr_vol = None
    hr_rh = 1
    hr_rw = 1
    if USE_RGB_PATCHES:
        rgb_hr_vol = load_rgb_hr_volume()
        _hr_meta = load_rgb_hr_meta()
        hr_rh = int(_hr_meta['rh'])
        hr_rw = int(_hr_meta['rw'])
    tr_i, va_i, tr_p, va_p = split_train_val_indices(tr_ind, VAL_RATIO, RANDOM_SEED)
    defer = 0 < VAL_RATIO < 1.0
    te_i = None if defer else load_test_indices_shifted(ls)
    train_loader, _, _ = build_dataloaders(
        fv,
        rv,
        tr_i,
        va_i,
        te_i,
        BATCH_SIZE,
        defer_test=defer,
        train_global_rows=tr_p,
        val_global_rows=va_p,
        rgb_strict_view=bool(USE_RGB_PATCHES),
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
    )
    opt['len_train_dataloader'] = len(train_loader)
    model = create_classifier(opt, None).to(device)
    maybe_attach_forward_dataflow_trace(logger, model)
    loss_fn = model.loss_func
    optimizer = model.optimizer
    model.train()
    batch = next(iter(train_loader))
    data_dict, labels = batch_to_dict(batch, device, USE_RGB_PATCHES, USE_SUPCON)
    optimizer.zero_grad()
    loss, _, _ = compute_classification_loss(model, data_dict, labels, loss_fn)
    loss.backward()
    ok = log_projection_gradients(model, logger, None, 0)
    if ok:
        logger.info('[verify_projection_grad] 通过：各模态投影均有非零梯度。')
        sys.exit(0)
    logger.error('[verify_projection_grad] 未通过：存在零梯度或 grad 缺失。')
    sys.exit(1)
