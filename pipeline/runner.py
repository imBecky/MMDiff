"""
分类训练主流程：数据与循环与「模型如何构建」解耦，通过 create_classifier(opt, diffusion) 注入。
替换你自己的 model 包时，只需实现与当前 MultimodalClassifier 相同的外部契约：
  - forward(data_dict)，可选 return_center_logits=True（当 param.USE_CENTER_LOSS）
  - loss_func, optimizer, 可选 exp_lr_scheduler
"""
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
    DIFFUSION_NOISE_MODE,
    DIFFUSION_NORMALIZE_INPUT,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    FEAT_SCALES,
    LOG_PATH,
    MULTIMODAL_ABLATION_LOG_LINE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
    RESUME_CHECKPOINT,
    SAVE_EVERY_EPOCH,
    STUDENT_CHECKPOINT,
    STUDENT_NUM_TRAIN_TIMESTEPS,
    TB_LOG_ROOT,
    TRAIN_QUICK_VERIFY,
    TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS,
    USE_CENTER_LOSS,
    USE_RGB_PATCHES,
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
    prepare_tb_run_dir,
)
from .loop import (
    compute_classification_loss,
    evaluate,
    log_projection_gradients,
    train_one_epoch,
)
from .metrics import accuracies
from .student_diffusion import StudentDiffusionWrapper

CreateClassifierFn = Callable[[Any, Any], torch.nn.Module]


@dataclass
class TrainingRunOptions:
    """训练运行选项（由 main 解析传入）。"""
    no_artifacts: bool = False


def _normalize_resume_path(resume_checkpoint: str) -> str:
    s = (resume_checkpoint or '').strip()
    if not s:
        return ''
    p = Path(s).expanduser()
    abs_s = os.path.abspath(str(p)) if not p.is_absolute() else str(p.resolve())
    if not os.path.isdir(abs_s):
        raise FileNotFoundError(f'断点目录不存在或不是文件夹: {abs_s}')
    return abs_s


def run_training(
    create_classifier: CreateClassifierFn,
    run_options: Optional[TrainingRunOptions] = None,
) -> None:
    """完整训练 + 验证 + 测试；模型由 create_classifier(opt, diffusion) 提供。"""
    opts = run_options or TrainingRunOptions()
    no_artifacts = bool(opts.no_artifacts)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feats_vol, rgb_vol, train_indices, label_shift = load_train_bundle()
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
    tr_idx, va_idx = split_train_val_indices(train_indices, VAL_RATIO, RANDOM_SEED)
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
    )
    opt['len_train_dataloader'] = len(train_loader)
    logger.info(
        'VAL_RATIO=%s | train_batches=%d val_batches=%s | diffusion_noise=%s normalize_input=%s',
        VAL_RATIO,
        len(train_loader),
        len(val_loader) if val_loader is not None else None,
        DIFFUSION_NOISE_MODE,
        DIFFUSION_NORMALIZE_INPUT,
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
        if not run_ckps_dir_str:
            CKPS_DIR.mkdir(parents=True, exist_ok=True)
            run_ckps_dir_str = str(CKPS_DIR / run_dir.name)
        os.makedirs(run_ckps_dir_str, exist_ok=True)
        logger.info('Checkpoint 目录: %s', run_ckps_dir_str)
    else:
        logger.info('无文件产物模式（--no-artifacts）：不创建 TB/断点目录，不写 TensorBoard、周期断点、final')

    best_path = (
        os.path.join(run_ckps_dir_str, BEST_MODEL_FILENAME) if run_ckps_dir_str else None
    )

    diffusion = StudentDiffusionWrapper(
        STUDENT_CHECKPOINT,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        noise_mode=DIFFUSION_NOISE_MODE,
        noise_seed_base=RANDOM_SEED,
        normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
        feat_layers=FEAT_SCALES,
    )
    logger.info('Initial Student Diffusion Model Finished (scheduler from checkpoint: %s)', type(diffusion.scheduler).__name__)
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

    best_state_dict = None

    epoch_bar = tqdm(range(start_epoch, NUM_EPOCHS), desc='Epochs')

    try:
        for epoch in epoch_bar:
            epoch_start = time.perf_counter()
            eval_sel_loss = float('nan')
            eval_sel_acc = float('nan')
            ovr_acc = 0.0
            kappa = 0.0
            s_sqr = 0.0

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
                ovr_acc, usr_acc, prod_acc, kappa, s_sqr = accuracies(conf)
                last_eval_epoch = epoch
                last_ovr_acc = ovr_acc
                last_eval_sel_loss = eval_sel_loss
                last_eval_sel_acc = eval_sel_acc
                last_kappa = kappa
                last_s_sqr = s_sqr
            elif run_selection:
                ovr_acc = last_ovr_acc
                eval_sel_loss = last_eval_sel_loss
                eval_sel_acc = last_eval_sel_acc
                kappa = last_kappa
                s_sqr = last_s_sqr
            epoch_seconds = time.perf_counter() - epoch_start

            if writer is not None:
                writer.add_scalar('train/epoch_loss', train_loss, epoch + 1)
                writer.add_scalar('train/epoch_acc', train_acc, epoch + 1)
                if should_run_eval:
                    writer.add_scalar(f'{selection_split}/overall_accuracy', ovr_acc, epoch + 1)
                    writer.add_scalar(f'{selection_split}/kappa', kappa, epoch + 1)
                    writer.add_scalar(f'{selection_split}/kappa_variance', s_sqr, epoch + 1)
                writer.add_scalar('timing/epoch_seconds', epoch_seconds, epoch + 1)
            if run_selection and should_run_eval:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f %s_loss=%.4f %s_acc=%.4f ovr_acc=%.4f kappa=%.4f epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    eval_sel_loss,
                    selection_split,
                    eval_sel_acc,
                    ovr_acc,
                    kappa,
                    epoch_seconds,
                )
            elif run_selection:
                logger.info(
                    'Epoch %03d summary | train_loss=%.4f train_acc=%.4f (skip %s: 每 %d epoch eval) last_ovr_acc=%.4f epoch_seconds=%.2f',
                    epoch + 1,
                    train_loss,
                    train_acc,
                    selection_split,
                    EVAL_INTERVAL_EPOCHS,
                    last_ovr_acc,
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

            # 仅在已满足验证且 train_acc 过门槛时按间隔保存（与 run_selection 一致）
            if (
                run_selection
                and SAVE_EVERY_EPOCH > 0
                and run_ckps_dir_str
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
                    '断点已保存至 %s (next_epoch=%d, 每 %d epoch 一次，且需 train_acc≥%.0f%%)',
                    ep_ckpt,
                    epoch + 1,
                    SAVE_EVERY_EPOCH,
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

        if run_ckps_dir_str and best_path and not os.path.isfile(best_path):
            torch.save(model.state_dict(), best_path)
            best_epoch = NUM_EPOCHS - 1
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
                NUM_EPOCHS,
                global_step,
                best_acc,
                best_epoch,
                final_dir,
                run_log_dir_str,
                run_ckps_dir_str,
            )
            logger.info(
                '训练结束，最终断点已保存至 %s（含 classifier.pt 与 training_state.pt）',
                final_dir,
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
                feats_vol, rgb_vol, ti, BATCH_SIZE
            )
        _, _, conf_final, eval_loss_final, eval_acc_final = evaluate(
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
        ovr_acc_final, usr_acc, prod_acc, kappa, s_sqr = accuracies(conf_final)

        if writer is not None:
            writer.add_scalar('final/test_loss', eval_loss_final, best_epoch + 1)
            writer.add_scalar('final/test_acc', eval_acc_final, best_epoch + 1)
            writer.add_scalar('final/overall_accuracy', ovr_acc_final, best_epoch + 1)
            writer.add_scalar('final/kappa', kappa, best_epoch + 1)
            writer.add_text(
                'final/summary',
                '\n'.join([
                    f'best_epoch: {best_epoch + 1}',
                    f'test_accuracy: {np.round(100 * ovr_acc_final, 2)}%',
                    f'user_accuracy: {np.round(100 * usr_acc, 2)}',
                    f'producer_accuracy: {np.round(100 * prod_acc, 2)}',
                    f'kappa: {np.round(kappa, 4)}',
                    f'kappa_variance: {np.round(s_sqr, 6)}',
                ]),
            )

        log_and_print(logger, 'Best epoch is', best_epoch + 1)
        log_and_print(logger, 'Test accuracy is', np.round(100 * ovr_acc_final, 2), '%')
        log_and_print(logger, 'User accuracy is', np.round(100 * usr_acc, 2))
        log_and_print(logger, 'Producer accuracy is', np.round(100 * prod_acc, 2))
        log_and_print(logger, 'Kappa coefficient is', np.round(kappa, 4))
        log_and_print(logger, 'Kappa variance is', np.round(s_sqr, 6))
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
    tr_i, va_i = split_train_val_indices(tr_ind, VAL_RATIO, RANDOM_SEED)
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
    )
    opt['len_train_dataloader'] = len(train_loader)
    diffusion = StudentDiffusionWrapper(
        STUDENT_CHECKPOINT,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        noise_mode=DIFFUSION_NOISE_MODE,
        noise_seed_base=RANDOM_SEED,
        normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
        feat_layers=FEAT_SCALES,
    )
    model = create_classifier(opt, diffusion).to(device)
    loss_fn = model.loss_func
    optimizer = model.optimizer
    model.train()
    batch = next(iter(train_loader))
    data_dict, labels = batch_to_dict(batch, device, USE_RGB_PATCHES)
    optimizer.zero_grad()
    loss, _ = compute_classification_loss(model, data_dict, labels, loss_fn)
    loss.backward()
    ok = log_projection_gradients(model, logger, None, 0)
    if ok:
        logger.info('[verify_projection_grad] 通过：各模态投影均有非零梯度。')
        sys.exit(0)
    logger.error('[verify_projection_grad] 未通过：存在零梯度或 grad 缺失。')
    sys.exit(1)
