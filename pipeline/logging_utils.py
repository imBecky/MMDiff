import logging
import os
import re
from datetime import datetime
from pathlib import Path

from param import (
    BEST_MODEL_FILENAME,
    BATCH_SIZE,
    CHECK_PROJECTION_GRAD,
    DATA_DIR,
    EVAL_INTERVAL_EPOCHS,
    EVAL_MIN_TRAIN_ACC,
    EVAL_VAL_START_EPOCH,
    LEARNING_RATE,
    LOG_PATH,
    LOSS_WEIGHT_CENTER,
    LOSS_WEIGHT_GLOBAL,
    RUN_NAME_PREFIX,
    MULTIMODAL_ABLATION_LOG_LINE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OPTIMIZER_BETAS,
    TB_LOG_ROOT,
    TEST_LABELS_PATH,
    TEST_PATCHES_PATH,
    TEST_RGB_PATCHES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_PATCHES_PATH,
    TRAIN_RGB_PATCHES_PATH,
    USE_CENTER_LOSS,
    USE_RGB_PATCHES,
    WEIGHT_DECAY,
    opt,
)
from torch.utils.tensorboard import SummaryWriter


def prepare_tb_run_dir():
    """与 TensorBoard 使用同一 run 目录（TB_LOG_ROOT / run_name）。"""
    TB_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get('GFDIFF_EXPERIMENT_TAG', '').strip()
    if tag:
        safe = re.sub(r'[^\w\-.]', '_', tag)
        run_name = f'{RUN_NAME_PREFIX}_{safe}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    else:
        run_name = f'{RUN_NAME_PREFIX}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir = TB_LOG_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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
    logger.info('Using device: %s', device)
    if train_rgb is not None:
        logger.info(
            'Data shapes | train_hsi=%s train_lidar=%s train_rgb=%s train_labels=%s '
            'test_hsi=%s test_lidar=%s test_rgb=%s test_labels=%s',
            train_hsi.shape,
            train_lidar.shape,
            train_rgb.shape,
            train_labels.shape,
            test_hsi.shape,
            test_lidar.shape,
            test_rgb.shape,
            test_labels.shape,
        )
    else:
        logger.info(
            'Data shapes | train_hsi=%s train_lidar=%s train_labels=%s test_hsi=%s test_lidar=%s test_labels=%s',
            train_hsi.shape,
            train_lidar.shape,
            train_labels.shape,
            test_hsi.shape,
            test_lidar.shape,
            test_labels.shape,
        )
    logger.info(
        'Training config | batch_size=%d epochs=%d lr=%.8f betas=%s weight_decay=%.6f',
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
            f'best_model (per run ckpt dir): {BEST_MODEL_FILENAME}',
            f'LOG_PATH: {log_file_path if log_file_path is not None else LOG_PATH}',
        ]),
    )
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
            f"module_cast3 hsi_group_size/mid: {mc3.get('hsi_group_size', '-')}/{mc3.get('mid', '-')}",
            MULTIMODAL_ABLATION_LOG_LINE,
        ]),
    )
