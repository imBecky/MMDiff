"""
Two-branch CNN（BUCT）三阶段训练：与官方 main.py / finetune_Net 协议一致。

仅在对比实验且模型名为 two_branch* 且未设 MMDIFF_TWO_BRANCH_END2END=1 时由 runner 调用。
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
    EARLY_STOPPING_PATIENCE,
    HSI_CHANNELS,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    NUM_CLASSES,
    NUM_EPOCHS,
    TB_LOG_ROOT,
    opt,
)
from model.compare_model.two_branch_cnn_core import (
    TwoBranchFinetuneModel,
    TwoBranchHSIStage,
    TwoBranchLiDARStage,
)

from .checkpoint import save_classifier_checkpoint
from .classification_metrics import accuracies
from .data import build_test_loader, load_test_indices_shifted
from .loop import evaluate, train_one_epoch
from .logging_utils import log_and_print, save_confusion_detail_log

CreateClassifierFn = Callable[[Any, Any], torch.nn.Module]


class _FinetuneForward(nn.Module):
    """供 train_one_epoch / evaluate 使用 data_dict 接口。"""

    def __init__(self, net: TwoBranchFinetuneModel) -> None:
        super().__init__()
        self.net = net

    def forward(self, data_dict: dict, return_center_logits: bool = False, return_supcon_proj: bool = False):
        logits = self.net(data_dict['hsi'], data_dict['lidar'])
        if return_center_logits:
            return logits, logits
        return logits


def _epochs_phase(env_key: str, default: int) -> int:
    raw = (os.environ.get(env_key) or '').strip()
    if raw:
        return max(1, int(raw))
    return max(1, int(default))


def _use_two_branch_protocol() -> bool:
    if (os.environ.get('MMDIFF_TWO_BRANCH_END2END') or '').strip().lower() in ('1', 'true', 'yes'):
        return False
    name = (os.environ.get('MMDIFF_COMPARE_MODEL') or '').strip().lower().replace('-', '_')
    return name in (
        'two_branch_cnn',
        'two_branch',
        'twobranch_cnn',
        'xu2017_ms',
    )


def run_two_branch_protocol_if_needed(
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
    """
    若启用三阶段协议则执行完整训练+测试并返回 True（调用方应 return）；
    否则返回 False。
    """
    if not compare_run or not _use_two_branch_protocol():
        return False

    ds = opt.get('dataset', {})
    patch_size = int(ds.get('patch_size') or 11)
    hsi_c = int(ds.get('hsi_channels') or HSI_CHANNELS)
    lidar_c = int(ds.get('lidar_channel') or 1)
    n_cls = int(ds.get('n_cls') or NUM_CLASSES)

    epochs_hsi = _epochs_phase('MMDIFF_TWO_BRANCH_EPOCHS_HSI', NUM_EPOCHS)
    epochs_lidar = _epochs_phase('MMDIFF_TWO_BRANCH_EPOCHS_LIDAR', NUM_EPOCHS)
    epochs_ft = _epochs_phase('MMDIFF_TWO_BRANCH_EPOCHS_FINETUNE', NUM_EPOCHS)

    logger.info(
        'Two-branch CNN：三阶段训练（BUCT 协议）| HSI epochs=%d | LiDAR epochs=%d | Finetune epochs=%d | '
        'Adam 1e-4（支路）/ SGD 5e-3 mom=1e-6（融合，支路冻结）',
        epochs_hsi,
        epochs_lidar,
        epochs_ft,
    )

    loss_fn = nn.CrossEntropyLoss()
    clip_grad = float(opt.get('train', {}).get('clip_grad_norm', 0) or 0)
    diffusion = None

    hsi_path = os.path.join(run_ckps_dir_str or '.', 'two_branch_hsi_stage.pt')
    lidar_path = os.path.join(run_ckps_dir_str or '.', 'two_branch_lidar_stage.pt')

    # ---------- Phase 1: HSI ----------
    hsi_model = TwoBranchHSIStage(patch_size, hsi_c, n_cls).to(device)
    opt_hsi = torch.optim.Adam(hsi_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    global_step = 0
    for epoch in range(epochs_hsi):
        tr_loss, tr_acc, global_step = train_one_epoch(
            hsi_model,
            train_loader,
            loss_fn,
            opt_hsi,
            device,
            logger,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            clip_grad_norm=clip_grad,
            lr_scheduler=None,
        )
        logger.info(
            'Two-branch [1/3 HSI] epoch %03d/%03d | train_loss=%.4f train_acc=%.4f',
            epoch + 1,
            epochs_hsi,
            tr_loss,
            tr_acc,
        )
    if run_ckps_dir_str:
        torch.save(hsi_model.state_dict(), hsi_path)
        logger.info('已保存 HSI 阶段权重: %s', hsi_path)

    # ---------- Phase 2: LiDAR ----------
    lidar_model = TwoBranchLiDARStage(patch_size, lidar_c, n_cls).to(device)
    opt_lidar = torch.optim.Adam(lidar_model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    for epoch in range(epochs_lidar):
        tr_loss, tr_acc, global_step = train_one_epoch(
            lidar_model,
            train_loader,
            loss_fn,
            opt_lidar,
            device,
            logger,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            clip_grad_norm=clip_grad,
            lr_scheduler=None,
        )
        logger.info(
            'Two-branch [2/3 LiDAR] epoch %03d/%03d | train_loss=%.4f train_acc=%.4f',
            epoch + 1,
            epochs_lidar,
            tr_loss,
            tr_acc,
        )
    if run_ckps_dir_str:
        torch.save(lidar_model.state_dict(), lidar_path)
        logger.info('已保存 LiDAR 阶段权重: %s', lidar_path)

    # ---------- Phase 3: Finetune ----------
    finetune = TwoBranchFinetuneModel(patch_size, hsi_c, lidar_c, n_cls).to(device)
    finetune.load_from_hs_lidar_stages(
        hsi_model.state_dict(),
        lidar_model.state_dict(),
        strict=True,
    )
    finetune.freeze_encoders()
    opt_ft = torch.optim.SGD(
        finetune.trainable_fusion_parameters(),
        lr=5e-3,
        momentum=1e-6,
    )

    model = _FinetuneForward(finetune).to(device)
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    early_stop_patience = max(0, int(EARLY_STOPPING_PATIENCE))
    early_stop_evals_without_gain = 0
    last_eval_epoch = None
    selection_loader = val_loader
    selection_split = 'val'
    if val_loader is None:
        logger.warning('VAL_RATIO=0：Finetune 阶段用测试集选优（仅便于跑通）')
        selection_loader = test_loader
        selection_split = 'test'

    epoch_bar = tqdm(range(epochs_ft), desc='Two-branch [3/3 Finetune]')
    last_epoch = -1
    for epoch in epoch_bar:
        last_epoch = epoch
        prev_best_acc = best_acc
        tr_loss, tr_acc, global_step = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            opt_ft,
            device,
            logger,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            clip_grad_norm=clip_grad,
            lr_scheduler=None,
        )

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
                loss_fn,
                device,
                NUM_CLASSES,
                logger,
                writer=writer,
                epoch=epoch,
                split=selection_split,
                use_center_logits=False,
            )
            ovr_acc, _, _, kappa, _, aa = accuracies(conf)
            last_eval_epoch = epoch
            logger.info(
                'Two-branch [3/3 Finetune] epoch %03d | train_acc=%.4f %s oa=%.4f aa=%.4f kappa=%.4f',
                epoch + 1,
                tr_acc,
                selection_split,
                ovr_acc,
                aa,
                kappa,
            )

        if should_run_eval and ovr_acc >= best_acc:
            best_acc = ovr_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in finetune.state_dict().items()}
            if best_path:
                # 与 TwoBranchCNNClassifier.net 权重一致，便于 load
                torch.save(best_state, best_path)
                logger.info('Finetune best 已写入 %s (epoch=%d oa=%.4f)', best_path, epoch + 1, ovr_acc)

        if should_run_eval and early_stop_patience > 0:
            if ovr_acc > prev_best_acc + 1e-12:
                early_stop_evals_without_gain = 0
            else:
                early_stop_evals_without_gain += 1
                if early_stop_evals_without_gain >= early_stop_patience:
                    logger.info('Finetune Early stopping at epoch %d', epoch + 1)
                    break

        epoch_bar.set_postfix(train_acc=f'{tr_acc:.4f}', best_oa=f'{100 * max(best_acc, 0):.2f}%')

    if best_state is not None:
        finetune.load_state_dict(best_state)
    if best_path and best_state is not None and not os.path.isfile(best_path):
        torch.save(best_state, best_path)

    # final checkpoint 目录（与主流程 classifier 结构一致）
    if run_ckps_dir_str:
        final_dir = os.path.join(run_ckps_dir_str, 'final')
        final_cls = create_classifier(opt, diffusion).to(device)
        final_cls.net.load_state_dict(finetune.state_dict())
        save_classifier_checkpoint(
            final_cls,
            final_cls.optimizer,
            getattr(final_cls, 'exp_lr_scheduler', None),
            last_epoch + 1,
            global_step,
            best_acc,
            best_epoch,
            final_dir,
            run_log_dir_str,
            run_ckps_dir_str,
        )

    # ---------- Final test（与 runner 一致）----------
    best_model = create_classifier(opt, diffusion).to(device)
    if best_path and os.path.isfile(best_path):
        sd = torch.load(best_path, map_location=device)
        if all(k.startswith('net.') for k in sd.keys()):
            best_model.load_state_dict(sd, strict=True)
        else:
            best_model.net.load_state_dict(sd, strict=True)
        logger.info('Final test 使用 best 权重: %s', best_path)
    elif best_state is not None:
        best_model.net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info('Final test 使用内存中的 Finetune best 权重')
    else:
        best_model.net.load_state_dict(finetune.state_dict())
        logger.info('Final test 使用最后一轮 Finetune 权重')

    if test_loader is None:
        logger.info('加载测试集（final evaluation）...')
        ti = load_test_indices_shifted(label_shift)
        test_loader = build_test_loader(feats_vol, rgb_vol, ti, BATCH_SIZE)

    preds_final, targets_final, conf_final, eval_loss_final, eval_acc_final = evaluate(
        best_model,
        test_loader,
        best_model.loss_func,
        device,
        NUM_CLASSES,
        logger,
        writer=writer,
        epoch=best_epoch,
        split='test',
        use_center_logits=False,
    )
    ovr_acc_final, usr_acc, prod_acc, kappa, s_sqr, aa_final = accuracies(conf_final)

    logger.info(
        'Final test | OA=%.4f AA=%.4f Kappa=%.4f',
        ovr_acc_final,
        aa_final,
        kappa,
    )
    log_and_print(logger, 'Best epoch is', best_epoch + 1)
    log_and_print(logger, 'Test accuracy is', np.round(100 * ovr_acc_final, 2), '%')
    log_and_print(logger, 'Average accuracy (AA) is', np.round(100 * aa_final, 2), '%')
    log_and_print(logger, 'Kappa coefficient is', np.round(kappa, 4))

    if save_conf_detail and run_log_dir_str:
        conf_log_path = Path(run_log_dir_str) / 'conf_detail.log'
        save_confusion_detail_log(conf_log_path, conf_final, NUM_CLASSES)
        logger.info('混淆矩阵与误分类对已写入 %s', conf_log_path)

    if writer is not None:
        writer.add_scalar('final/overall_accuracy', ovr_acc_final, best_epoch + 1)
        writer.add_scalar('final/average_accuracy', aa_final, best_epoch + 1)
        writer.add_scalar('final/kappa', kappa, best_epoch + 1)

    if run_log_dir_str:
        summary_path = Path(run_log_dir_str) / 'metrics_summary.json'
        summary_path.write_text(
            json.dumps(
                {
                    'experiment_tag': os.environ.get('MMDIFF_EXPERIMENT_TAG', ''),
                    'compare_run': True,
                    'compare_model': os.environ.get('MMDIFF_COMPARE_MODEL', ''),
                    'two_branch_protocol': 'buct_three_stage',
                    'best_epoch': int(best_epoch + 1),
                    'oa': float(ovr_acc_final),
                    'aa': float(aa_final),
                    'kappa': float(kappa),
                    'oa_percent': float(np.round(100 * ovr_acc_final, 4)),
                },
                indent=2,
                ensure_ascii=False,
            )
            + '\n',
            encoding='utf-8',
        )
        logger.info('metrics_summary 已写入 %s', summary_path)

    if writer is not None:
        writer.flush()
        writer.close()

    return True


def should_run_two_branch_protocol(compare_run: bool) -> bool:
    return bool(compare_run and _use_two_branch_protocol())
