import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import math

from param import (
    BEST_MODEL_FILENAME,
    BATCH_SIZE,
    CHECK_PROJECTION_GRAD,
    CLIP_GRAD_NORM,
    DATA_DIR,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    HSI_AGG_MODE_CFG,
    HSI_CONV_HIDDEN_CFG,
    HSI_RESIDUAL_BLOCKS_CFG,
    HSI_SE_RATIO_CFG,
    LABEL_SHIFT_PATH,
    LEARNING_RATE,
    LOG_PATH,
    LOSS_WEIGHT_CENTER,
    LOSS_WEIGHT_GLOBAL,
    MULTIMODAL_ABLATION_LOG_LINE,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    OPTIMIZER_BETAS,
    PATCH_WINDOW_SIZE,
    RANDOM_SEED,
    RGB_STUDENT_CHECKPOINT,
    RUN_NAME_PREFIX,
    SAVE_EVERY_EPOCH,
    STUDENT_SIZE,
    TB_LOG_ROOT,
    TEST_LABELS_PATH,
    TEST_PATCHES_PATH,
    TEST_RGB_PATCHES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_PATCHES_PATH,
    TRAIN_ROT_AUGMENT_FACTOR,
    USE_CENTER_LOSS,
    USE_RGB_PATCHES,
    VAL_RATIO,
    WEIGHT_DECAY,
    opt,
)
from torch.utils.tensorboard import SummaryWriter


def _is_compare_run() -> bool:
    return (os.environ.get('MMDIFF_COMPARE_RUN') or '').strip().lower() in ('1', 'true', 'yes')


def _compare_tb_model_slug() -> str:
    """对比实验未设 MMDIFF_EXPERIMENT_TAG 时：仅用 `--model` 注册名，便于 TB 侧栏阅读。

    （旧版曾在目录名里拼主模型的 B/H/SE/bs 与 modality，对对比基线无意义且冗长。）
    如需在目录名里区分更多因素，请设置 ``MMDIFF_EXPERIMENT_TAG``。
    """
    model = (os.environ.get('MMDIFF_COMPARE_MODEL') or 'compare').strip().lower()
    return re.sub(r'[^\w\-.]', '_', model) or 'compare'


def _lr_slug_for_run_dir() -> str:
    """目录名用短 lr 字符串；优先 MMDIFF_LR_TAG，否则由 LEARNING_RATE 生成。"""
    raw = (os.environ.get('MMDIFF_LR_TAG') or '').strip()
    if raw:
        return re.sub(r'[^\w\-.]', '_', raw)
    lr = float(LEARNING_RATE)
    if lr == 0.0:
        return '0'
    exp = int(math.floor(math.log10(abs(lr))))
    mant = lr / (10**exp)
    # 常见 1e-4、6e-4、4.8e-3 等用紧凑科学计数
    if abs(mant - 1.0) < 1e-12:
        return f'1e{exp}'
    if abs(mant - 6.0) < 1e-12 and exp == -4:
        return '6e-4'
    if abs(mant - 4.8) < 1e-12 and exp == -3:
        return '4p8e-3'
    return re.sub(r'[^\w\-.]', '_', f'{lr:g}')


def _compact_tb_tag(tag: str, max_len: int = 40) -> str:
    """去掉冗长前缀，截断，便于 TensorBoard 侧栏显示。"""
    safe = re.sub(r'[^\w\-.]', '_', tag.strip())
    for prefix in ('multimodal_hsi_rgb_lidar_', 'multimodal_'):
        if safe.startswith(prefix):
            safe = safe[len(prefix) :]
            break
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe or 'run'


def prepare_tb_run_dir():
    """与 TensorBoard 使用同一 run 目录（TB_LOG_ROOT / run_name）。

    默认短名（便于 TB 看图）：``{ts}_e{NN}_lr{slug}`` 或 ``{ts}_{紧凑tag}``。
    若同时设置了 ``MMDIFF_EXPERIMENT_NUM`` 与 ``MMDIFF_EXPERIMENT_TAG``，短名会在 e+lr 后
    **追加紧凑 tag**（避免串行多组消融共用同一时间戳时写入同一目录）。
    长目录名：``MMDIFF_TB_LONG_TAG=1`` 时在 e+lr 后再拼**完整** EXPERIMENT_TAG（替代紧凑后缀）。
    """
    TB_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get('MMDIFF_EXPERIMENT_TAG', '').strip()
    ts = (os.environ.get('MMDIFF_RUN_TIMESTAMP') or '').strip() or datetime.now().strftime('%m%d-%H%M')
    prefix = (RUN_NAME_PREFIX or '').strip()
    exp_raw = (os.environ.get('MMDIFF_EXPERIMENT_NUM') or '').strip()
    want_long = (os.environ.get('MMDIFF_TB_LONG_TAG') or '').strip().lower() in ('1', 'true', 'yes', 'y')

    if exp_raw:
        try:
            exp_n = int(exp_raw)
            exp_part = f'e{exp_n:02d}'
        except ValueError:
            exp_part = 'eXX'
        lr_slug = _lr_slug_for_run_dir()
        run_name = f'{ts}_{exp_part}_lr{lr_slug}'
        if want_long and tag:
            safe = re.sub(r'[^\w\-.]', '_', tag)
            run_name = f'{run_name}_{safe}'
        elif tag:
            run_name = f'{run_name}_{_compact_tb_tag(tag)}'
    elif tag:
        if want_long:
            safe = re.sub(r'[^\w\-.]', '_', tag)
        else:
            safe = _compact_tb_tag(tag)
        if prefix:
            run_name = f'{prefix}_{ts}_{safe}'
        else:
            run_name = f'{ts}_{safe}'
    else:
        if _is_compare_run():
            body = _compare_tb_model_slug()
            if prefix:
                run_name = f'{prefix}_{body}_{ts}'
            else:
                run_name = f'cmp_{body}_{ts}'
        elif prefix:
            run_name = f'{prefix}_{ts}'
        else:
            run_name = ts
    run_dir = TB_LOG_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_confusion_detail_log(
    out_path: Path,
    conf: np.ndarray,
    num_classes: int,
    *,
    global_top_k: int = 40,
) -> None:
    """
    将测试集混淆矩阵与主要误分类对写入单独文件（与 model.log 同目录时常用文件名 conf_detail.log）。
    conf: sklearn confusion_matrix，行=真实类，列=预测类。
    """
    cm = np.asarray(conf, dtype=np.int64)
    if cm.shape != (num_classes, num_classes):
        raise ValueError(f'conf shape {cm.shape} 与 num_classes={num_classes} 不一致')

    lines: list[str] = []
    lines.append('confusion matrix (rows=true class index, cols=pred class index)')
    lines.append(np.array2string(cm))
    lines.append('')
    lines.append('--- Per true class: top off-diagonal predictions (true -> pred : count) ---')
    for i in range(num_classes):
        row = cm[i].copy()
        row[i] = 0
        order = np.argsort(row)[::-1]
        parts: list[str] = []
        for j in order[: min(3, num_classes)]:
            if int(row[j]) > 0:
                parts.append(f'{i} -> {j}: {int(row[j])}')
        lines.append(f'true class {i}: ' + ('; '.join(parts) if parts else '(none)'))

    pairs: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(key=lambda x: -x[0])
    lines.append('')
    lines.append(f'--- Global off-diagonal pairs sorted by count (top {global_top_k}) ---')
    for cnt, i, j in pairs[:global_top_k]:
        lines.append(f'{i} -> {j}: {cnt}')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def get_logger(log_path=None):
    """训练入口统一日志：写入 log_path（默认 param.LOG_PATH）；并让 model 里用的 'base' logger 同文件输出。"""
    path = Path(log_path) if log_path is not None else LOG_PATH
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path, encoding='utf-8')
    file_handler.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(file_handler)

    # MultimodalClassifier 等复用 logging.getLogger('base')，与上面同一文件，避免重复建 experiments/.../logs
    base_log = logging.getLogger('base')
    if not base_log.handlers:
        base_log.setLevel(logging.INFO)
        base_fh = logging.FileHandler(path, encoding='utf-8')
        base_fh.setFormatter(fmt)
        base_log.addHandler(base_fh)
        base_log.propagate = False

    return logger


def get_console_logger():
    """仅控制台输出，不创建日志文件（配合 --no-artifacts）。"""
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log = logging.getLogger('MMDiff.training')
    if not log.handlers:
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        log.addHandler(ch)
        log.propagate = False
    base_log = logging.getLogger('base')
    if not base_log.handlers:
        base_log.setLevel(logging.INFO)
        ch2 = logging.StreamHandler()
        ch2.setFormatter(fmt)
        base_log.addHandler(ch2)
        base_log.propagate = False
    return log


def log_and_print(logger, *parts):
    message = ' '.join(str(part) for part in parts)
    print(*parts)
    logger.info(message)


def get_summary_writer(logger, run_dir):
    if SummaryWriter is None:
        logger.warning('TensorBoard is unavailable because tensorboard is not installed.')
        return None

    logger.info('TensorBoard logs will be written to %s', run_dir)
    return SummaryWriter(log_dir=str(run_dir))


def log_config(
    logger,
    writer,
    device,
    train_hsi,
    train_lidar,
    train_labels,
    test_hsi,
    test_lidar,
    test_labels,
    train_rgb=None,
    test_rgb=None,
    log_file_path=None,
):
    def _sh(x):
        return x.shape if x is not None else None

    compare_run = _is_compare_run()
    logger.info('Using device: %s', device)
    logger.info(
        'Data paths | DATA_DIR=%s TRAIN_PATCHES=%s TRAIN_RGB=%s USE_RGB_PATCHES=%s LABEL_SHIFT=%s',
        DATA_DIR,
        TRAIN_PATCHES_PATH,
        TRAIN_RGB_PATCHES_PATH,
        USE_RGB_PATCHES,
        LABEL_SHIFT_PATH,
    )
    if train_hsi is None and train_lidar is None:
        logger.info(
            'Data shapes (patches + label indices) | train_labels=%s test_labels=%s',
            _sh(train_labels),
            _sh(test_labels),
        )
    elif train_rgb is not None:
        logger.info(
            'Data shapes | train_hsi=%s train_lidar=%s train_rgb=%s train_labels=%s '
            'test_hsi=%s test_lidar=%s test_rgb=%s test_labels=%s',
            _sh(train_hsi),
            _sh(train_lidar),
            _sh(train_rgb),
            _sh(train_labels),
            _sh(test_hsi),
            _sh(test_lidar),
            _sh(test_rgb),
            _sh(test_labels),
        )
    else:
        logger.info(
            'Data shapes | train_hsi=%s train_lidar=%s train_labels=%s test_hsi=%s test_lidar=%s test_labels=%s',
            _sh(train_hsi),
            _sh(train_lidar),
            _sh(train_labels),
            _sh(test_hsi),
            _sh(test_lidar),
            _sh(test_labels),
        )
    logger.info(
        'Training config | batch_size=%d epochs=%d lr=%.4g betas=%s weight_decay=%.6f',
        BATCH_SIZE,
        NUM_EPOCHS,
        LEARNING_RATE,
        OPTIMIZER_BETAS,
        WEIGHT_DECAY,
    )

    if writer is None:
        return

    writer.add_text(
        'config/paths',
        '\n'.join([
            f'DATA_DIR: {DATA_DIR}',
            f'TRAIN_PATCHES_PATH: {TRAIN_PATCHES_PATH}',
            f'TEST_PATCHES_PATH: {TEST_PATCHES_PATH}',
            f'TRAIN_RGB_PATCHES_PATH: {TRAIN_RGB_PATCHES_PATH}',
            f'TEST_RGB_PATCHES_PATH: {TEST_RGB_PATCHES_PATH}',
            f'USE_RGB_PATCHES: {USE_RGB_PATCHES}',
            f'dataset.modalities: {opt.get("dataset", {}).get("modalities")}',
            f'TRAIN_LABELS_PATH: {TRAIN_LABELS_PATH}',
            f'TEST_LABELS_PATH: {TEST_LABELS_PATH}',
            f'LABEL_SHIFT_PATH: {LABEL_SHIFT_PATH}',
            f'PATCH_WINDOW_SIZE: {PATCH_WINDOW_SIZE}',
            f'best_model (per run ckpt dir): {BEST_MODEL_FILENAME}',
            f'LOG_PATH: {log_file_path if log_file_path is not None else LOG_PATH}',
        ]),
    )
    if compare_run:
        writer.add_text(
            'config/hparams',
            '\n'.join([
                f'device: {device}',
                f'batch_size: {BATCH_SIZE}',
                f'epochs: {NUM_EPOCHS}',
                f'learning_rate: {LEARNING_RATE}',
                f'optimizer_betas: {OPTIMIZER_BETAS}',
                f'weight_decay: {WEIGHT_DECAY}',
                f'num_classes: {NUM_CLASSES}',
                f'compare_model: {os.environ.get("MMDIFF_COMPARE_MODEL", "")}',
                f'EVAL_MIN_TRAIN_ACC: {EVAL_MIN_TRAIN_ACC}',
                f'EVAL_INTERVAL_EPOCHS: {EVAL_INTERVAL_EPOCHS}',
                f'EVAL_VAL_START_EPOCH: {EVAL_VAL_START_EPOCH}',
                f'CHECK_PROJECTION_GRAD: {CHECK_PROJECTION_GRAD}',
            ]),
        )
    else:
        mc3 = opt.get('module_cast3') or {}
        writer.add_text(
            'config/hparams',
            '\n'.join([
                f'device: {device}',
                f'batch_size: {BATCH_SIZE}',
                f'epochs: {NUM_EPOCHS}',
                f'learning_rate: {LEARNING_RATE}',
                f'optimizer_betas: {OPTIMIZER_BETAS}',
                f'weight_decay: {WEIGHT_DECAY}',
                f'num_classes: {NUM_CLASSES}',
                f'USE_CENTER_LOSS: {USE_CENTER_LOSS}',
                f'LOSS_WEIGHT_GLOBAL: {LOSS_WEIGHT_GLOBAL}',
                f'LOSS_WEIGHT_CENTER: {LOSS_WEIGHT_CENTER}',
                f'EVAL_MIN_TRAIN_ACC: {EVAL_MIN_TRAIN_ACC}',
                f'EVAL_INTERVAL_EPOCHS: {EVAL_INTERVAL_EPOCHS}',
                f'EVAL_VAL_START_EPOCH: {EVAL_VAL_START_EPOCH}',
                f'CHECK_PROJECTION_GRAD: {CHECK_PROJECTION_GRAD}',
                f"module_cast3 lidar_hidden: {mc3.get('lidar_hidden', '-')}",
                f"module_cast3 hsi_residual_blocks: {mc3.get('hsi_residual_blocks', '-')}",
                f"module_cast3 hsi_conv_hidden: {mc3.get('hsi_conv_hidden', '-')}",
                f"module_cast3 hsi_se_ratio: {mc3.get('hsi_se_ratio', '-')}",
                f"module_cast3 hsi_agg_mode: {mc3.get('hsi_agg_mode', HSI_AGG_MODE_CFG)}",
                MULTIMODAL_ABLATION_LOG_LINE,
            ]),
        )


def log_module_structure(logger, model: nn.Module) -> None:
    """将 nn.Module 的层级 repr 逐行写入日志（静态结构，与 print(model) 同源）。"""
    try:
        struct = str(model)
    except Exception as exc:
        logger.warning('模型 str() 失败: %s', exc)
        return
    if not struct:
        return
    logger.info('----- model structure (nn.Module repr,不随输入变化) -----')
    for ln in struct.splitlines():
        logger.info('%s', ln.rstrip('\r'))


def maybe_attach_forward_dataflow_trace(logger, model: nn.Module) -> None:
    """
    按真实前向执行顺序记录子模块的输入/输出形状（随 batch、分支、arch_variant 而变）。
    开启：MMDIFF_FORWARD_TRACE=1（或 MMDIFF_LOG_DATAFLOW=1）
    可选：MMDIFF_FORWARD_TRACE_DEPTH（默认 3，模块名按 '.' 分段数，过深的子模块不打印）
         MMDIFF_FORWARD_TRACE_MAX_FORWARDS（默认 1，只追踪前 N 次根 model.forward，防日志爆炸）
    说明：hook 在子模块 forward **返回后**触发，故顺序为「先叶子/encoder 块，后含它们的父模块」，
    与 PyTorch 调用栈一致；若要更粗只看顶层，把 DEPTH 设为 1。
    """
    raw = (
        os.environ.get('MMDIFF_FORWARD_TRACE') or os.environ.get('MMDIFF_LOG_DATAFLOW') or ''
    ).strip().lower()
    if raw not in ('1', 'true', 'yes', 'y'):
        return

    try:
        depth = max(1, int(os.environ.get('MMDIFF_FORWARD_TRACE_DEPTH') or '3'))
    except ValueError:
        depth = 3
    try:
        max_fwd = max(1, int(os.environ.get('MMDIFF_FORWARD_TRACE_MAX_FORWARDS') or '1'))
    except ValueError:
        max_fwd = 1

    def shape_desc(x) -> str:
        if isinstance(x, torch.Tensor):
            return str(tuple(x.shape))
        if isinstance(x, (list, tuple)):
            parts = []
            for t in x:
                if isinstance(t, torch.Tensor):
                    parts.append(str(tuple(t.shape)))
                else:
                    parts.append(type(t).__name__)
            return '[' + ', '.join(parts) + ']'
        if x is None:
            return 'None'
        return type(x).__name__

    class _TraceCtx:
        __slots__ = ('max_root', 'root_i', 'active', 'seq')

        def __init__(self, mf: int):
            self.max_root = mf
            self.root_i = 0
            self.active = False
            self.seq = 0

        def root_pre(self, *_a, **_k):
            self.active = self.root_i < self.max_root
            self.root_i += 1
            self.seq = 0

    ctx = _TraceCtx(max_fwd)
    handles: list = []

    def make_hook(full_name: str):
        def _hook(_mod, inp, out):
            if not ctx.active:
                return
            ctx.seq += 1
            try:
                ins = inp
                if isinstance(inp, tuple) and len(inp) == 1:
                    ins = inp[0]
                logger.info(
                    '[dataflow] %03d %s | in=%s out=%s',
                    ctx.seq,
                    full_name,
                    shape_desc(ins),
                    shape_desc(out),
                )
            except Exception as exc:
                logger.info('[dataflow] %03d %s | (log err: %s)', ctx.seq, full_name, exc)

        return _hook

    handles.append(model.register_forward_pre_hook(ctx.root_pre))
    for name, mod in model.named_modules():
        if not name:
            continue
        if len(name.split('.')) > depth:
            continue
        handles.append(mod.register_forward_hook(make_hook(name)))

    logger.info('========== Forward dataflow trace（动态，MMDIFF_FORWARD_TRACE=1）==========')
    logger.info(
        '本次 run将对前 %d 次根前向打点；模块名深度<=%d；'
        '行序=各模块 forward 返回顺序（子模块通常先于父模块）',
        max_fwd,
        depth,
    )


def log_model_and_training_detail(
    logger,
    writer,
    model: nn.Module,
    opt,
    clip_grad_norm: float,
    diffusion=None,
):
    """
    在模型构建（及可选 resume）之后写入：参数量与子模块规模、opt 训练/调度/分类头、扩散与数据相关要点。
    同时写入 TensorBoard config/model、config/train_extended（若 writer 非空）。
    """
    train_cfg = opt.get('train') or {}
    compare_run = _is_compare_run()
    optim_cfg = train_cfg.get('optimizer') or {}
    sched_cfg = dict(train_cfg.get('scheduler') or {})
    mc = opt.get('model_cls') or {}
    ds = opt.get('dataset') or {}
    mc3 = opt.get('module_cast3') or {}
    unet_cfg = (opt.get('model') or {}).get('unet') or {}

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info('========== Model ==========')
    logger.info('class=%s', type(model).__name__)
    logger.info(
        'parameters total=%d (%.2fM) trainable=%d (%.2fM)',
        total,
        total / 1e6,
        trainable,
        trainable / 1e6,
    )
    for name, child in model.named_children():
        if not isinstance(child, nn.Module):
            continue
        n = sum(p.numel() for p in child.parameters())
        if n:
            logger.info('  submodule %-20s %9d params', name, n)

    log_module_structure(logger, model)

    if hasattr(model, 'd_model') and not compare_run:
        av = getattr(model, 'arch_variant', None)
        if av is None:
            av = '-'
        mem = getattr(model, 'mem_len', '-') if hasattr(model, 'mem_len') else '-'
        logger.info(
            'classifier layout | d_model=%s mem_len=%s seq_len=%s num_classes=%s arch_variant=%s',
            getattr(model, 'd_model', '-'),
            mem,
            getattr(model, 'seq_len', '-'),
            getattr(model, 'num_classes', '-'),
            av,
        )

    logger.info('========== Training / opt ==========')
    logger.info(
        'run | RANDOM_SEED=%s VAL_RATIO=%s num_workers=%s TRAIN_ROT_AUGMENT_FACTOR=%s',
        RANDOM_SEED,
        VAL_RATIO,
        ds.get('num_workers', NUM_WORKERS),
        TRAIN_ROT_AUGMENT_FACTOR,
    )
    logger.info(
        'file hparams | BATCH_SIZE=%s NUM_EPOCHS=%s LEARNING_RATE=%s betas=%s weight_decay=%s CLIP_GRAD_NORM=%s',
        BATCH_SIZE,
        NUM_EPOCHS,
        LEARNING_RATE,
        OPTIMIZER_BETAS,
        WEIGHT_DECAY,
        CLIP_GRAD_NORM,
    )
    logger.info(
        'optimizer (opt) | type=%s lr=%s weight_decay=%s betas=%s',
        optim_cfg.get('type'),
        optim_cfg.get('lr'),
        optim_cfg.get('weight_decay'),
        optim_cfg.get('betas'),
    )
    logger.info(
        'effective clip_grad (runner)=%s | center_loss=%s w_global=%s w_center=%s',
        clip_grad_norm,
        USE_CENTER_LOSS,
        LOSS_WEIGHT_GLOBAL,
        LOSS_WEIGHT_CENTER,
    )
    logger.info('lr_scheduler (opt train.scheduler)=%s', sched_cfg)
    logger.info(
        'dataset (opt) | n_cls=%s hsi_channels=%s lidar_channel=%s modalities=%s resolution=%s',
        ds.get('n_cls'),
        ds.get('hsi_channels'),
        ds.get('lidar_channel'),
        ds.get('modalities'),
        ds.get('resolution'),
    )
    if compare_run:
        logger.info('compare run | model=%s', os.environ.get('MMDIFF_COMPARE_MODEL', ''))
    else:
        logger.info(
            'model_cls | token_dim=%s transformer: heads=%s layers=%s ff=%s dropout=%s head_hidden=%s',
            mc.get('token_dim'),
            mc.get('transformer_heads'),
            mc.get('transformer_layers'),
            mc.get('transformer_ff_dim'),
            mc.get('transformer_dropout'),
            mc.get('head_hidden'),
        )
        logger.info(
            'model_cls | rgb_source=%s init_type=%s scale=%s',
            mc.get('rgb_source', 'student'),
            mc.get('init_type'),
            mc.get('scale'),
        )
        logger.info(
            'model_cls | rgb_to_lidar_guidance_mode=%s (none=关 film=RGB student FiLM→LiDAR token)',
            mc.get('rgb_to_lidar_guidance_mode', 'none'),
        )
        logger.info(
            'rgb student | MMDIFF_RGB_STUDENT_CHECKPOINT=%s MMDIFF_FREEZE_RGB_STUDENT=%s',
            RGB_STUDENT_CHECKPOINT or '(empty=random init)',
            (os.environ.get('MMDIFF_FREEZE_RGB_STUDENT') or '').strip() or '0',
        )
        logger.info(
            'module_cast3 | lidar_hidden=%s hsi_residual_blocks=%s hsi_conv_hidden=%s hsi_se_ratio=%s hsi_agg_mode=%s',
            mc3.get('lidar_hidden'),
            mc3.get('hsi_residual_blocks', HSI_RESIDUAL_BLOCKS_CFG),
            mc3.get('hsi_conv_hidden', HSI_CONV_HIDDEN_CFG),
            mc3.get('hsi_se_ratio', HSI_SE_RATIO_CFG),
            mc3.get('hsi_agg_mode', HSI_AGG_MODE_CFG),
        )
        logger.info('legacy opt | STUDENT_SIZE=%s（占位 SR3 image_size，分类不加载扩散）', STUDENT_SIZE)
    logger.info(
        'checkpoint habit | SAVE_EVERY_EPOCH=%s periodic from epoch>=%d (1-based, last 20%% of NUM_EPOCHS) BEST=%s',
        SAVE_EVERY_EPOCH,
        max(1, int(NUM_EPOCHS * 0.8) + 1),
        BEST_MODEL_FILENAME,
    )
    logger.info(
        'eval gates | EVAL_VAL_START_EPOCH=%s EVAL_MIN_TRAIN_ACC=%s EVAL_INTERVAL_EPOCHS=%s',
        EVAL_VAL_START_EPOCH,
        EVAL_MIN_TRAIN_ACC,
        EVAL_INTERVAL_EPOCHS,
    )
    if not compare_run:
        logger.info('%s', MULTIMODAL_ABLATION_LOG_LINE)
    if unet_cfg and not compare_run:
        logger.info('model.unet (condensed) | inner_channel=%s multiplier=%s res_blocks=%s dropout=%s',
                    unet_cfg.get('inner_channel'), unet_cfg.get('channel_multiplier'),
                    unet_cfg.get('res_blocks'), unet_cfg.get('dropout'))


    logger.info('========== (model / training detail end) ==========')

    if writer is None:
        return

    mod_lines = [
        f'class: {type(model).__name__}',
        f'total_params: {total} ({total / 1e6:.2f}M)',
        f'trainable_params: {trainable} ({trainable / 1e6:.2f}M)',
    ]
    for name, child in model.named_children():
        if isinstance(child, nn.Module):
            n = sum(p.numel() for p in child.parameters())
            if n:
                mod_lines.append(f'{name}: {n}')
    if hasattr(model, 'd_model') and not compare_run:
        mod_lines.extend([
            f'd_model: {getattr(model, "d_model", None)}',
            f'seq_len: {getattr(model, "seq_len", None)}',
        ])
    writer.add_text('config/model', '\n'.join(mod_lines))

    if compare_run:
        ext_lines = [
            f'RANDOM_SEED: {RANDOM_SEED}',
            f'VAL_RATIO: {VAL_RATIO}',
            f'train_rot_augment_factor: {TRAIN_ROT_AUGMENT_FACTOR}',
            f'num_workers (dataset): {ds.get("num_workers", NUM_WORKERS)}',
            f'CLIP_GRAD_NORM (param): {CLIP_GRAD_NORM}',
            f'clip_grad_norm (runner effective): {clip_grad_norm}',
            f'optimizer: {optim_cfg}',
            f'lr_scheduler: {sched_cfg}',
            f'dataset: {ds}',
            f'compare_model: {os.environ.get("MMDIFF_COMPARE_MODEL", "")}',
            f'SAVE_EVERY_EPOCH: {SAVE_EVERY_EPOCH}',
            f'periodic_ckpt_1based_min_epoch: {max(1, int(NUM_EPOCHS * 0.8) + 1)}',
        ]
    else:
        ext_lines = [
            f'RANDOM_SEED: {RANDOM_SEED}',
            f'VAL_RATIO: {VAL_RATIO}',
            f'train_rot_augment_factor: {TRAIN_ROT_AUGMENT_FACTOR}',
            f'num_workers (dataset): {ds.get("num_workers", NUM_WORKERS)}',
            f'CLIP_GRAD_NORM (param): {CLIP_GRAD_NORM}',
            f'clip_grad_norm (runner effective): {clip_grad_norm}',
            f'optimizer: {optim_cfg}',
            f'lr_scheduler: {sched_cfg}',
            f'model_cls: {mc}',
            f'module_cast3: {mc3}',
            f'dataset: {ds}',
            f'STUDENT_SIZE: {STUDENT_SIZE}',
            f'SAVE_EVERY_EPOCH: {SAVE_EVERY_EPOCH}',
            f'periodic_ckpt_1based_min_epoch: {max(1, int(NUM_EPOCHS * 0.8) + 1)}',
            MULTIMODAL_ABLATION_LOG_LINE,
        ]
    writer.add_text('config/train_extended', '\n'.join(ext_lines))
