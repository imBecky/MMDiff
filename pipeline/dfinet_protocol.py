"""
DFINet（Gao et al. 2022）按 DFINet.ipynb 的训练方式：联合损失 + SGD。

与 notebook 一致：optimizer = SGD(lr, momentum=0.9, weight_decay=5e-4)；
loss = CE + α·L_dist + β·L_cons（α=β=0.01，可由环境变量覆盖）。

lr 默认读 param.train.optimizer.lr（公平对比），未设时用 1e-3（与 notebook 一致）。

设 MMDIFF_DFINET_END2END=1 时跳过本协议（退回 runner 普通 CE+AdamW，仅调试，非论文方法）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from param import (
    BATCH_SIZE,
    CLIP_GRAD_NORM,
    EARLY_STOPPING_PATIENCE,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    NUM_CLASSES,
    NUM_EPOCHS,
    USE_RGB_PATCHES,
    opt,
)
from model.compare_model.dfinet_loss import dfinet_calc_loss
from model.compare_model.registry import DFINET_PROTOCOL_COMPARE_NAMES

from .checkpoint import save_classifier_checkpoint
from .classification_metrics import accuracies
from .data import batch_to_dict, build_test_loader, load_test_indices_shifted
from .loop import evaluate
from .logging_utils import log_and_print, save_confusion_detail_log

CreateClassifierFn = Callable[[Any, Any], torch.nn.Module]


def _use_dfinet_protocol() -> bool:
    if (os.environ.get('MMDIFF_DFINET_END2END') or '').strip().lower() in ('1', 'true', 'yes'):
        return False
    name = (os.environ.get('MMDIFF_COMPARE_MODEL') or '').strip().lower().replace('-', '_')
    return name in DFINET_PROTOCOL_COMPARE_NAMES


def run_dfinet_protocol_if_needed(
    *,
    create_classifier: CreateClassifierFn,
    compare_run: bool,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    feats_vol: np.ndarray,
    rgb_vol: Optional[np.ndarray],
    label_shift: np.ndarray,
    defer_test_load: bool,
    logger,
    writer,
    run_ckps_dir_str: str,
    best_path: Optional[str],
    no_artifacts: bool,
    save_conf_detail: bool,
    run_log_dir_str: str,
) -> bool:
    if not compare_run or not _use_dfinet_protocol():
        return False

    ds = opt.get('dataset', {})
    n_cls = int(ds.get('n_cls') or NUM_CLASSES)
    train_cfg = opt.get('train', {})
    optim_cfg = train_cfg.get('optimizer', {})
    lr = float(optim_cfg.get('lr') or 1e-3)
    clip_grad = float(train_cfg.get('clip_grad_norm', CLIP_GRAD_NORM) or 0)

    alpha = float((os.environ.get('MMDIFF_DFINET_ALPHA') or '0.01').strip() or 0.01)
    beta = float((os.environ.get('MMDIFF_DFINET_BETA') or '0.01').strip() or 0.01)

    logger.info(
        'DFINet：论文协议训练 | SGD lr=%g momentum=0.9 wd=5e-4 | 联合损失 α=%g β=%g（MMDIFF_DFINET_ALPHA/BETA 可覆盖）',
        lr,
        alpha,
        beta,
    )

    diffusion = None
    model = create_classifier(opt, diffusion).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    ce_eval = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    global_step = 0
    early_stop_patience = max(0, int(EARLY_STOPPING_PATIENCE))
    early_stop_evals_without_gain = 0
    last_eval_epoch = None
    selection_loader = val_loader
    selection_split = 'val'
    if val_loader is None:
        logger.warning('VAL_RATIO=0：DFINet 用测试集选优')
        selection_loader = test_loader
        selection_split = 'test'

    epoch_bar = tqdm(range(NUM_EPOCHS), desc='DFINet [paper loss]')
    last_epoch = -1
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        last_epoch = epoch
        prev_best_acc = best_acc
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        inner = tqdm(train_loader, desc='Train', leave=False, dynamic_ncols=True)
        for batch in inner:
            data_dict, labels = batch_to_dict(batch, device, USE_RGB_PATCHES)
            optimizer.zero_grad()
            f1, f2, h1, l1, logits = model(data_dict, return_aux=True)
            loss = dfinet_calc_loss(f1, f2, h1, l1, logits, labels, n_cls, alpha, beta)
            loss.backward()
            if clip_grad and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            global_step += 1

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            inner.set_postfix(loss=f'{running_loss / total:.4f}', acc=f'{correct / total:.4f}')

        tr_loss = running_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        epoch_seconds = time.perf_counter() - epoch_start

        if writer is not None:
            writer.add_scalar('train/epoch_loss', tr_loss, epoch + 1)
            writer.add_scalar('train/epoch_acc', tr_acc, epoch + 1)
            writer.add_scalar('timing/epoch_seconds', epoch_seconds, epoch + 1)

        run_eval = epoch >= EVAL_VAL_START_EPOCH and selection_loader is not None
        run_selection = run_eval and tr_acc >= EVAL_MIN_TRAIN_ACC
        should_run_eval = run_selection and (
            EVAL_INTERVAL_EPOCHS <= 0
            or last_eval_epoch is None
            or (epoch - last_eval_epoch) >= EVAL_INTERVAL_EPOCHS
        )
        ovr_acc = 0.0
        aa = 0.0
        kappa = 0.0

        if should_run_eval:
            _, _, conf, _, _ = evaluate(
                model,
                selection_loader,
                ce_eval,
                device,
                NUM_CLASSES,
                logger,
                writer=writer,
                epoch=epoch,
                split=selection_split,
                use_center_logits=False,
            )
            ovr_acc, _, _, kappa, s_sqr, aa = accuracies(conf)
            last_eval_epoch = epoch
            if writer is not None:
                writer.add_scalar(f'{selection_split}/overall_accuracy', ovr_acc, epoch + 1)
                writer.add_scalar(f'{selection_split}/average_accuracy', aa, epoch + 1)
                writer.add_scalar(f'{selection_split}/kappa', kappa, epoch + 1)
                writer.add_scalar(f'{selection_split}/kappa_variance', s_sqr, epoch + 1)
            logger.info(
                'DFINet epoch %03d | train_loss=%.4f train_acc=%.4f %s oa=%.4f aa=%.4f kappa=%.4f',
                epoch + 1,
                tr_loss,
                tr_acc,
                selection_split,
                ovr_acc,
                aa,
                kappa,
            )

        if should_run_eval and ovr_acc >= best_acc:
            best_acc = ovr_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if best_path:
                torch.save(best_state, best_path)
                logger.info('DFINet best -> %s', best_path)

        if should_run_eval and early_stop_patience > 0:
            if ovr_acc > prev_best_acc + 1e-12:
                early_stop_evals_without_gain = 0
            else:
                early_stop_evals_without_gain += 1
                if early_stop_evals_without_gain >= early_stop_patience:
                    logger.info('DFINet Early stopping at epoch %d', epoch + 1)
                    break

        epoch_bar.set_postfix(train_acc=f'{tr_acc:.4f}', best_oa=f'{100 * max(best_acc, 0):.2f}%')

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if run_ckps_dir_str:
        final_dir = os.path.join(run_ckps_dir_str, 'final')
        save_classifier_checkpoint(
            model,
            optimizer,
            None,
            last_epoch + 1,
            global_step,
            best_acc,
            best_epoch,
            final_dir,
            run_log_dir_str,
            run_ckps_dir_str,
        )

    if test_loader is None:
        ti = load_test_indices_shifted(label_shift)
        test_loader = build_test_loader(feats_vol, rgb_vol, ti, BATCH_SIZE)

    preds_final, targets_final, conf_final, eval_loss_final, eval_acc_final = evaluate(
        model,
        test_loader,
        ce_eval,
        device,
        NUM_CLASSES,
        logger,
        writer=writer,
        epoch=best_epoch,
        split='test',
        use_center_logits=False,
    )
    ovr_acc_final, usr_acc, prod_acc, kappa, s_sqr, aa_final = accuracies(conf_final)

    log_and_print(logger, 'Best epoch is', best_epoch + 1)
    log_and_print(logger, 'Test accuracy is', np.round(100 * ovr_acc_final, 2), '%')
    log_and_print(logger, 'Average accuracy (AA) is', np.round(100 * aa_final, 2), '%')
    log_and_print(logger, 'Kappa coefficient is', np.round(kappa, 4))

    if save_conf_detail and run_log_dir_str:
        conf_log_path = Path(run_log_dir_str) / 'conf_detail.log'
        save_confusion_detail_log(conf_log_path, conf_final, NUM_CLASSES)

    if run_log_dir_str:
        summary_path = Path(run_log_dir_str) / 'metrics_summary.json'
        summary_path.write_text(
            json.dumps(
                {
                    'experiment_tag': os.environ.get('MMDIFF_EXPERIMENT_TAG', ''),
                    'compare_run': True,
                    'compare_model': os.environ.get('MMDIFF_COMPARE_MODEL', ''),
                    'dfinet_protocol': 'paper_joint_loss_sgd',
                    'best_epoch': int(best_epoch + 1),
                    'oa': float(ovr_acc_final),
                    'aa': float(aa_final),
                    'kappa': float(kappa),
                },
                indent=2,
                ensure_ascii=False,
            )
            + '\n',
            encoding='utf-8',
        )

    if writer is not None:
        writer.flush()
        writer.close()

    return True
