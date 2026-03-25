import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from param import (
    CHECK_PROJECTION_GRAD,
    CHECK_PROJECTION_GRAD_INTERVAL,
    EVAL_LOG_INTERVAL,
    LOSS_WEIGHT_CENTER,
    LOSS_WEIGHT_GLOBAL,
    SUPCON_TEMPERATURE,
    SUPCON_WEIGHT,
    TRAIN_LOG_INTERVAL,
    USE_CENTER_LOSS,
    USE_RGB_PATCHES,
    USE_SUPCON,
)

from .data import batch_to_dict


def _log_train_step_line(logger, msg: str) -> None:
    """有文件 handler 时只 logger.info；仅控制台时用 tqdm.write，避免与进度条抢行、混成 lr=…it/s。"""
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if has_file:
        logger.info(msg)
    if has_stream and not has_file:
        tqdm.write(msg)


def _module_grad_l2_norm(module: nn.Module):
    """返回子模块可训练参数的梯度 L2 范数、有梯度的参数个数、requires_grad 但 grad 为 None 的个数。"""
    total_sq = 0.0
    n_with = 0
    n_missing = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_missing += 1
            continue
        g = p.grad.detach().float()
        total_sq += float((g * g).sum().item())
        n_with += 1
    total = total_sq ** 0.5
    return total, n_with, n_missing


def log_projection_gradients(model, logger, writer, step_for_tb: int):
    """
    在 loss.backward() 之后、clip_grad_norm_ 之前调用。
    汇总 model.projections 各模态子模块的梯度；若某模态范数长期为 0 或存在 grad is None，需检查前向是否被 no_grad 截断。
    返回 True 表示每个含可训练参数的模态子模块均有非零梯度范数（用于 --verify-projection-grad）。
    """
    if not hasattr(model, 'projections'):
        logger.warning('[proj_grad] model 无 projections 属性，跳过梯度检查。')
        return False
    parts = []
    any_trainable = False
    all_nonzero = True
    grad_eps = 1e-12
    for name, sub in model.projections.items():
        norm, n_with, n_missing = _module_grad_l2_norm(sub)
        has_trainable = any(p.requires_grad for p in sub.parameters())
        if has_trainable:
            any_trainable = True
            if n_with == 0 or norm < grad_eps:
                all_nonzero = False
        parts.append(f'{name}: L2={norm:.6g} (grad_ok={n_with}, grad_none={n_missing})')
        if writer is not None:
            writer.add_scalar(f'train/grad_proj/{name}_l2', norm, step_for_tb)
            writer.add_scalar(f'train/grad_proj/{name}_missing', n_missing, step_for_tb)
    msg = '[proj_grad] ' + ' | '.join(parts)
    logger.info(msg)
    if not any_trainable:
        logger.warning('[proj_grad] projections 下无可训练参数，检查配置。')
        return False
    if all_nonzero:
        logger.info('[proj_grad] 结论: 各模态可训练投影均有非零梯度。')
    else:
        logger.warning('[proj_grad] 结论: 存在零梯度或 grad 缺失，请检查建图与 no_grad。')
    return all_nonzero


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """SupCon：同 batch 内同类为正（含另一视图），分母为除自身外全体。"""
    device = features.device
    features = F.normalize(features, dim=1)
    batch_size = features.shape[0]
    similarity_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    labels_eq = torch.eq(labels, labels.T).float().to(device)
    eye = torch.eye(batch_size, device=device, dtype=labels_eq.dtype)
    mask_pos = labels_eq * (1.0 - eye)

    eye_mask = 1.0 - eye
    exp_logits = torch.exp(similarity_matrix) * eye_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    denom = mask_pos.sum(1).clamp(min=1e-12)
    mean_log_prob_pos = (mask_pos * log_prob).sum(1) / denom
    return -mean_log_prob_pos.mean()


def compute_classification_loss(model, data_dict, labels, loss_fn):
    """
    训练用：可选 全局 + 中心 双项交叉熵（见 param.USE_CENTER_LOSS）。
    可选 SupCon：param.USE_SUPCON，在 c_rep 投影上监督对比，与 CE 相加。
    loss_fn 与 MultimodalClassifier.loss_func 一致（CrossEntropyLoss）。
    返回的 logits 用于 train_acc 统计：若 USE_CENTER_LOSS 则用 logits_c，与 eval 保持一致。
    """
    extra = {}
    if USE_SUPCON:
        if USE_CENTER_LOSS:
            logits_g, logits_c, z = model(
                data_dict, return_center_logits=True, return_supcon_proj=True,
            )
            loss_ce = (
                LOSS_WEIGHT_GLOBAL * loss_fn(logits_g, labels)
                + LOSS_WEIGHT_CENTER * loss_fn(logits_c, labels)
            )
            logits = logits_c
        else:
            logits_c, z = model(data_dict, return_supcon_proj=True)
            loss_ce = loss_fn(logits_c, labels)
            logits = logits_c
        loss_s = supervised_contrastive_loss(z, labels, SUPCON_TEMPERATURE)
        loss = loss_ce + SUPCON_WEIGHT * loss_s
        extra['ce'] = float(loss_ce.detach().item())
        extra['supcon'] = float(loss_s.detach().item())
        return loss, logits, extra

    if USE_CENTER_LOSS:
        logits_g, logits_c = model(data_dict, return_center_logits=True)
        loss = (
            LOSS_WEIGHT_GLOBAL * loss_fn(logits_g, labels)
            + LOSS_WEIGHT_CENTER * loss_fn(logits_c, labels)
        )
        return loss, logits_c, extra
    logits = model(data_dict)
    return loss_fn(logits, labels), logits, extra


def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    logger,
    writer=None,
    epoch=0,
    global_step=0,
    clip_grad_norm=0.0,
    lr_scheduler=None,
):
    """训练一个 epoch。"""
    use_rgb = USE_RGB_PATCHES
    use_supcon = USE_SUPCON
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc='Train', leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress_bar, start=1):
        data_dict, labels = batch_to_dict(batch, device, use_rgb, use_supcon)

        optimizer.zero_grad()
        loss, logits, extra = compute_classification_loss(model, data_dict, labels, loss_fn)
        loss.backward()
        if CHECK_PROJECTION_GRAD and (
            batch_idx % CHECK_PROJECTION_GRAD_INTERVAL == 0
            or batch_idx == 1
            or batch_idx == len(loader)
        ):
            log_projection_gradients(model, logger, writer, global_step + 1)
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        global_step += 1

        batch_loss = loss.item()
        batch_acc = (preds == labels).float().mean().item()
        current_lr = optimizer.param_groups[0]['lr']

        progress_bar.set_postfix(
            loss=f'{running_loss / total:.4f}',
            acc=f'{correct / total:.4f}',
        )

        if writer is not None:
            writer.add_scalar('train/step_loss', batch_loss, global_step)
            writer.add_scalar('train/step_acc', batch_acc, global_step)
            writer.add_scalar('train/lr', current_lr, global_step)
            if extra.get('supcon') is not None:
                writer.add_scalar('train/step_supcon', extra['supcon'], global_step)
            if extra.get('ce') is not None:
                writer.add_scalar('train/step_ce', extra['ce'], global_step)

        if batch_idx % TRAIN_LOG_INTERVAL == 0 or batch_idx == len(loader):
            _log_train_step_line(
                logger,
                'Epoch %03d Train %04d/%04d | loss=%.4f acc=%.4f lr=%.4g'
                % (
                    epoch + 1,
                    batch_idx,
                    len(loader),
                    running_loss / total,
                    correct / total,
                    current_lr,
                ),
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, global_step


def evaluate(
    model,
    loader,
    loss_fn,
    device,
    num_classes,
    logger,
    writer=None,
    epoch=0,
    split='test',
    use_center_logits=False,
):
    use_rgb = USE_RGB_PATCHES
    model.eval()
    preds = []
    targets = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc='Eval', leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(progress_bar, start=1):
            data_dict, labels = batch_to_dict(batch, device, use_rgb)
            if use_center_logits and USE_CENTER_LOSS:
                logits_g, logits_c = model(data_dict, return_center_logits=True)
                logits = logits_c
            else:
                logits = model(data_dict)
            loss = loss_fn(logits, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            batch_preds = torch.argmax(logits, dim=1)
            batch_acc = (batch_preds == labels).float().mean().item()

            preds.append(batch_preds.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

            progress_bar.set_postfix(
                loss=f'{running_loss / total:.4f}',
                acc=f'{batch_acc:.4f}',
            )

            if batch_idx % EVAL_LOG_INTERVAL == 0 or batch_idx == len(loader):
                _log_train_step_line(
                    logger,
                    'Epoch %03d %s %04d/%04d | loss=%.4f acc=%.4f'
                    % (
                        epoch + 1,
                        split.capitalize(),
                        batch_idx,
                        len(loader),
                        running_loss / total,
                        batch_acc,
                    ),
                )

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    conf = confusion_matrix(targets, preds, labels=np.arange(num_classes))
    eval_loss = running_loss / total
    eval_acc = float(np.mean(preds == targets))

    if writer is not None:
        writer.add_scalar(f'{split}/epoch_loss', eval_loss, epoch + 1)
        writer.add_scalar(f'{split}/epoch_acc', eval_acc, epoch + 1)

    return preds, targets, conf, eval_loss, eval_acc
