from __future__ import annotations

from collections import OrderedDict
import os
import re
import warnings
from pathlib import Path
from typing import Optional
import torch
import utils.logger as Logger

# ---------------------------------------------------------------------------
# 数据与训练（常用修改处）
# ---------------------------------------------------------------------------
# 本地实验：直接改本文件常量，或 export MMDIFF_* 覆盖（见 _apply_mmdiff_env_overrides 文档字符串）。
# run.sh 仅负责时间戳/实验 tag、关机等；训练默认以本文件为准。
# ---------------------------------------------------------------------------
# 直接 python main.py（无 MMDIFF_*）：以下宽版各模态 stem + 分类头为默认；loss / center 距离 bias 与 historical run.sh 非分支项对齐。
# 例：lidar 48+2、HSI 5×96+SE32、HSI_AGG=multi_token、cls 192/192、transformer 1×384、LOSS_WEIGHT_GLOBAL=0.25、
# CENTER_DISTANCE_BIAS_ALPHA=3.5、tau=2.0（bias 仅为 alpha*exp(-dist/tau)，见 model/multimodal.py）。
SCHED_STEP_RATIOS = [0.6, 0.7]
SCHED_GAMMAS = [0.4, 0.1]
SCHEDULER_NAME = 'cosine'
# 仅 MMDIFF_SCHEDULER_NAME=cosine 时生效；warmup_ratio 为总 optimizer step 的占比（每 epoch 步数固定时等价于总轮数的相同比例）
SCHEDULER_COSINE_ETA_MIN_RATIO = 0.01
SCHEDULER_COSINE_WARMUP_RATIO = 0.1
SCHEDULER_COSINE_WARMUP_STEPS = 0
NUM_EPOCHS = 250
NUM_WORKERS = 24


def _apply_scheduler_env():
    global SCHED_STEP_RATIOS, SCHED_GAMMAS, SCHEDULER_NAME
    global SCHEDULER_COSINE_ETA_MIN_RATIO, SCHEDULER_COSINE_WARMUP_RATIO, SCHEDULER_COSINE_WARMUP_STEPS

    def _parse_csv_floats(name: str):
        raw = (os.environ.get(name) or '').strip()
        if not raw:
            return None
        return [float(x.strip()) for x in raw.split(',') if x.strip()]

    sr = _parse_csv_floats('MMDIFF_SCHED_STEP_RATIOS')
    if sr is not None and len(sr) >= 2:
        SCHED_STEP_RATIOS = sr
    sg = _parse_csv_floats('MMDIFF_SCHED_GAMMAS')
    if sg is not None and len(sg) >= 2:
        SCHED_GAMMAS = sg
    s = (os.environ.get('MMDIFF_SCHEDULER_NAME') or '').strip()
    if s:
        SCHEDULER_NAME = s
    for env_k, gk, cast in (
        ('MMDIFF_SCHED_COSINE_ETA_MIN_RATIO', 'SCHEDULER_COSINE_ETA_MIN_RATIO', float),
        ('MMDIFF_SCHED_COSINE_WARMUP_RATIO', 'SCHEDULER_COSINE_WARMUP_RATIO', float),
        ('MMDIFF_SCHED_COSINE_WARMUP_STEPS', 'SCHEDULER_COSINE_WARMUP_STEPS', int),
    ):
        v = (os.environ.get(env_k) or '').strip()
        if v:
            globals()[gk] = cast(v)


_apply_scheduler_env()
CLIP_GRAD_NORM = 1.0
EVAL_VAL_START_EPOCH = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4
BATCH_SIZE = 64
CLS_TRANSFORMER_DROPOUT = 0.1
# 续训时 LambdaLR 衰减边界用：与首次训练一致的总 step 数（通常由 checkpoint 自动写入；旧断点可 export MMDIFF_SCHEDULER_LR_TOTAL_STEPS）
SCHEDULER_LR_TOTAL_STEPS = 0.2

DATA_DIR = Path('../../autodl-fs/houston2018/prepared')
# 由 data_prepare.py 生成：整幅 HSI+LiDAR / RGB + 像素索引表
TRAIN_PATCHES_PATH = DATA_DIR / 'train_patches.npy'
TEST_PATCHES_PATH = TRAIN_PATCHES_PATH
TRAIN_RGB_PATCHES_PATH = DATA_DIR / 'train_rgb_patches.npy'
TEST_RGB_PATCHES_PATH = TRAIN_RGB_PATCHES_PATH
TRAIN_LABELS_PATH = DATA_DIR / 'train_labels.npy'
TEST_LABELS_PATH = DATA_DIR / 'test_labels.npy'
LABEL_SHIFT_PATH = DATA_DIR / 'label_shift.npy'
PATCH_WINDOW_SIZE = 11  # 须与 data_prepare 一致
# 训练时在线旋转增强：1=关；2=0/180°；4=0/90/180/270°
TRAIN_ROT_AUGMENT_FACTOR = 4

# ---------------------------------------------------------------------------
# 多模态分支消融：启用的模态集合（HSI/RGB/LiDAR）
# ---------------------------------------------------------------------------
# 用户通过环境变量指定启用组合，示例：
#   MMDIFF_MODALITY_COMBO=hsi
#   MMDIFF_MODALITY_COMBO=rgb
#   MMDIFF_MODALITY_COMBO=hsi+lidar
#   MMDIFF_MODALITY_COMBO=hsi+rgb+lidar
SUPPORTED_MODALITIES = ('hsi', 'rgb', 'lidar')
DEFAULT_MODALITY_COMBO = 'hsi+rgb+lidar'

def _parse_modality_combo(raw: str) -> tuple[str, ...]:
    s = (raw or '').strip().lower()
    if not s:
        s = DEFAULT_MODALITY_COMBO
    # 兼容 + , 空格与中横线作为分隔符
    s = s.replace('-', '+').replace(',', '+')
    toks = [t for t in re.split(r'\s*\+\s*', s) if t]
    if len(toks) == 1 and toks[0] in ('all', 'allmodal', 'all_modal'):
        return SUPPORTED_MODALITIES
    enabled_set = set()
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if t not in SUPPORTED_MODALITIES:
            raise ValueError(
                f'未知模态 {t!r}；应为 {SUPPORTED_MODALITIES} 的组合，例如 "hsi+rgb"'
            )
        enabled_set.add(t)
    if not enabled_set:
        raise ValueError('MMDIFF_MODALITY_COMBO 解析为空，请指定非空组合')
    # 固定顺序保证 token 拼接/日志稳定
    ordered = tuple(m for m in SUPPORTED_MODALITIES if m in enabled_set)
    return ordered


ENABLED_MODALITIES = _parse_modality_combo(os.environ.get('MMDIFF_MODALITY_COMBO', '') or '')
MODALITY_COMBO = '+'.join(ENABLED_MODALITIES)
_RGB_ENABLED = 'rgb' in ENABLED_MODALITIES
# 不在 import 时抛错，否则 data_prepare.py 等无法先 import param 再生成 .npy
if _RGB_ENABLED and not TRAIN_RGB_PATCHES_PATH.is_file():
    warnings.warn(
        f'启用 rgb 模态但尚未找到 {TRAIN_RGB_PATCHES_PATH}；'
        f'训练前请先运行 data_prepare.py 生成 train_rgb_patches.npy。',
        UserWarning,
        stacklevel=1,
    )
USE_RGB_PATCHES = bool(_RGB_ENABLED)
RGB_CHANNELS = 3
RUN_NAME_PREFIX = ''
# 验证集最佳权重保存在本次 run 的断点目录下：{run_ckps_dir}/{BEST_MODEL_FILENAME}
BEST_MODEL_FILENAME = 'best_model.pt'
# 仅兼容旧说明；训练流程不再写入此路径
MODEL_PATH = Path('./models/model.pt')
# 训练断点根目录（每次运行会创建子目录 run_tag，内含 checkpoint-<epoch>、final、best_model.pt）
CKPS_DIR = Path('../../autodl-tmp/classifier')
SAVE_EVERY_EPOCH = 100
# 从断点恢复：指向某次 run 下的 checkpoint-<n>（1-based epoch）或 final；空字符串表示新训练
# run.sh 可 export MMDIFF_RESUME_CHECKPOINT=绝对路径或相对仓库根的路径，覆盖本常量（便于 dep2/dep4 各用各的目录）
RESUME_CHECKPOINT = ''
TB_LOG_ROOT = Path('../../tf-logs')
# main 训练时实际写入 TB 同一 run 子目录下的 model.log；此处为未显式传路径时的回退
LOG_PATH = TB_LOG_ROOT / 'model.log'

# ---------------------------------------------------------------------------
# 小数据快速验证：仅用训练集的部分样本跑通流程、观察能否过拟合（不改 test 集）
# True 时：在 load_train_bundle 之后、train/val 划分之前，从训练索引按类分层随机抽取
# 每类至多 TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS 条；再按 VAL_RATIO 划验证集
# 若每类样本过少导致 stratify 报错，可暂设 VAL_RATIO = 0
# ---------------------------------------------------------------------------
TRAIN_QUICK_VERIFY = False
TRAIN_QUICK_VERIFY_SAMPLES_PER_CLASS = 150

HSI_CHANNELS = 48
LIDAR_CHANNELS = 1
# Houston2018 前景地物类为 20；GT 中背景常为 0，patch 仅含 y>0 像素，标签平移后为 0....19。
NUM_CLASSES = 20

RANDOM_SEED = 42
OPTIMIZER_BETAS = (0.9, 0.999)
TRAIN_LOG_INTERVAL = 20
EVAL_LOG_INTERVAL = 10

# 训练时检查各模态投影（HSI/LiDAR/RGB 等）在 backward 后是否有非零梯度；用于排查 no_grad 截断或未入图。
# 为 True 时，在 backward 之后、clip 之前按间隔汇总范数并写日志（可选 TensorBoard）。
CHECK_PROJECTION_GRAD = False
CHECK_PROJECTION_GRAD_INTERVAL = 10  # 每 N 个 batch；第 1 个与每 epoch 最后一个 batch 也会记录

VAL_RATIO = 0.1
# 本 epoch 训练准确率低于该值时不跑验证/选集 eval（0~1）；目标「先过拟合训练集」
EVAL_MIN_TRAIN_ACC = 0.0
# 过门槛后每 N 个 epoch 跑一次 eval（test 集大时减少 eval 频率；0 表示每 epoch 都跑）
EVAL_INTERVAL_EPOCHS = 1

# 早停：仅在「实际跑完验证集评估」时计数；连续 patience 次 OA 未严格超过历史最优则结束。0=关闭。
# 环境变量：MMDIFF_EARLY_STOPPING_PATIENCE
EARLY_STOPPING_PATIENCE = 30

USE_CENTER_LOSS = True
LOSS_WEIGHT_GLOBAL = 0.25


def _sync_loss_weights_from_global() -> None:
    """LOSS_WEIGHT_CENTER 恒为 1-LOSS_WEIGHT_GLOBAL；仅以 GLOBAL 为唯一可调量。"""
    global LOSS_WEIGHT_CENTER
    wg = float(LOSS_WEIGHT_GLOBAL)
    assert 0.0 < wg < 1.0, (
        f'LOSS_WEIGHT_GLOBAL 须在区间 (0, 1)，当前 {wg!r}（不设 clamp，请先修正配置）'
    )
    LOSS_WEIGHT_CENTER = 1.0 - wg


_sync_loss_weights_from_global()

# 清单实验：每项为 (global, ...) 只看 global；第二项仅为兼容旧写法、会被忽略（CENTER 由 1-global 得出）。须为外层元组包住多行：( (0.2, 0.8), ) 勿写成 ((0.2,0.8))。生效：MULTIMODAL_ABLATION_AXIS / INDEX 或 MMDIFF_ABLATION_*。
CENTER_GLOBAL_ABLATION = ((0.2, 0.8),)
MULTIMODAL_ABLATION_AXIS = None  # None | 'center_global'
MULTIMODAL_ABLATION_INDEX = 0

# RGB(student) -> LiDAR 引导（MultimodalClassifier）：none | film（FiLM 调制 lidar_g/lidar_c）
# 仅 rgb_source=student 且同时启用 rgb+lidar 时生效，否则自动关。
# 覆盖：MMDIFF_RGB_TO_LIDAR_GUIDANCE（别名 A 同 film）
def _normalize_rgb_to_lidar_guidance(raw: Optional[str]) -> str:
    s = (raw or 'none').strip().upper()
    if s in ('', 'NONE', 'OFF', '0', 'FALSE'):
        return 'none'
    if s in ('FILM', 'A'):
        return 'film'
    raise ValueError(
        'RGB_TO_LIDAR_GUIDANCE_MODE / MMDIFF_RGB_TO_LIDAR_GUIDANCE 须为 '
        f'none|film|A，当前 {raw!r}'
    )


RGB_TO_LIDAR_GUIDANCE_MODE = _normalize_rgb_to_lidar_guidance(
    os.environ.get('MMDIFF_RGB_TO_LIDAR_GUIDANCE')
)

# LiDAR 形态编码器 stem 隐藏通道（model/multimodal.py LidarMorphEncoder）
LIDAR_PROJ_HIDDEN_CFG = 48
# stem 之后在 feat_ch 上追加的空间残差块数（见 model/multimodal.py _LidarSpatialResidualBlock）
LIDAR_EXTRA_BLOCKS_CFG = 2
HSI_RESIDUAL_BLOCKS_CFG = 5
HSI_CONV_HIDDEN_CFG = 96
HSI_SE_RATIO_CFG = 32  # 0=关闭 SE；>0=开启并使用对应 squeeze ratio
# HSI 空间聚合：mean | attn_pool | multi_token
HSI_AGG_MODE_CFG = 'multi_token'
_EFFECTIVE_ABLATION_AXIS = None
_EFFECTIVE_ABLATION_INDEX = None


def _apply_multimodal_ablation():
    """按轴覆盖 LOSS_WEIGHT_GLOBAL（CENTER 由 _sync_loss_weights_from_global() 推导）。"""
    global LOSS_WEIGHT_GLOBAL, _EFFECTIVE_ABLATION_AXIS, _EFFECTIVE_ABLATION_INDEX
    axis = (os.environ.get('MMDIFF_ABLATION_AXIS') or '').strip() or (MULTIMODAL_ABLATION_AXIS or '')
    axis = axis.strip() or None
    idx_raw = (os.environ.get('MMDIFF_ABLATION_INDEX') or '').strip()
    try:
        idx = int(idx_raw) if idx_raw else int(MULTIMODAL_ABLATION_INDEX)
    except ValueError:
        idx = int(MULTIMODAL_ABLATION_INDEX)
    if axis is None:
        _EFFECTIVE_ABLATION_AXIS = None
        _EFFECTIVE_ABLATION_INDEX = None
        return
    _EFFECTIVE_ABLATION_AXIS = axis
    if axis == 'center_global':
        idx = max(0, min(idx, len(CENTER_GLOBAL_ABLATION) - 1))
        row = CENTER_GLOBAL_ABLATION[idx]
        LOSS_WEIGHT_GLOBAL = float(row[0] if isinstance(row, (tuple, list)) else row)
    else:
        raise ValueError(
            f'未知 MULTIMODAL_ABLATION_AXIS / MMDIFF_ABLATION_AXIS={axis!r}，应为 "center_global"'
        )
    _EFFECTIVE_ABLATION_INDEX = idx


_apply_multimodal_ablation()
_sync_loss_weights_from_global()

# 与 opt['model'] 中 SR3/UNet 占位配置一致的历史字段（分类主流程不再加载扩散教师）
STUDENT_SIZE = 256
STUDENT_IN_CHANNELS = 3
STUDENT_CHANNELS = (128, 256, 512, 512)
STUDENT_LAYERS_PER_BLOCK = 2

# HR 严格视野（唯一 RGB 空间对齐）：需 data_prepare 生成 rgb_hr.npy + rgb_hr.meta.json
TRAIN_RGB_HR_PATH = DATA_DIR / 'rgb_hr.npy'
RGB_HR_META_PATH = DATA_DIR / 'rgb_hr.meta.json'

# RGB：仅 LightweightRgbEncoder（patch CNN）；不再支持扩散 /离线 teacher token
RGB_SOURCE = 'student'
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_RGB_STUDENT_CKPT = _REPO_ROOT / 'model' / 'rgb_student_distill.pt'
RGB_STUDENT_CHECKPOINT = (os.environ.get('MMDIFF_RGB_STUDENT_CHECKPOINT') or '').strip()
if not RGB_STUDENT_CHECKPOINT:
    RGB_STUDENT_CHECKPOINT = str(_DEFAULT_RGB_STUDENT_CKPT)
CLS_INIT_TYPE = 'kaiming'
CLS_INIT_SCALE = 0.1
CLS_OUTPUT_CM_SIZE = 3
# 多模态 Transformer 分类头（见 model/multimodal.py）
CLS_TOKEN_DIM = 192
CLS_TRANSFORMER_HEADS = 4
CLS_TRANSFORMER_LAYERS = 1
CLS_TRANSFORMER_FF_DIM = 384
CLS_HEAD_HIDDEN = 192
# center query cross-attention logit bias：alpha * exp(-dist / tau)（固定指数核）
# dist 为 11×11 网格上的空间欧氏距离；alpha、tau 为标量超参。
CENTER_DISTANCE_BIAS_ALPHA = 3.5
CENTER_DISTANCE_BIAS_TAU = 2.0


def _train_scheduler_dict():
    """按 SCHEDULER_NAME 只写入当前形态所需字段，避免 opt 里一堆无关键。"""
    if SCHEDULER_NAME.lower() in ('cosine', 'cosine_annealing'):
        return OrderedDict(
            [
                ('name', 'cosine'),
                ('eta_min_ratio', SCHEDULER_COSINE_ETA_MIN_RATIO),
                ('warmup_ratio', SCHEDULER_COSINE_WARMUP_RATIO),
                ('warmup_steps', SCHEDULER_COSINE_WARMUP_STEPS),
            ]
        )
    return OrderedDict(
        [
            ('name', 'piecewise_two_step'),
            ('step_ratios', list(SCHED_STEP_RATIOS)),
            ('gammas', list(SCHED_GAMMAS)),
            ('constant_ratio', 0.8),
            ('gamma', 0.1),
        ]
    )


def build_opt():
    """完整训练配置 """
    inner = STUDENT_CHANNELS[0]
    mult = [c // inner for c in STUDENT_CHANNELS]
    sched = _train_scheduler_dict()
    ds = OrderedDict(
        [
            ('name', 'Houston2018'),
            ('dataroot', str(DATA_DIR)),
            ('modalities', list(ENABLED_MODALITIES)),
            ('resolution', 3),
            ('patch_size', PATCH_WINDOW_SIZE),
            ('batch_size', BATCH_SIZE),
            ('num_workers', NUM_WORKERS),
            ('use_shuffle', True),
            ('data_len', -1),
            ('n_cls', NUM_CLASSES),
            ('hsi_channels', HSI_CHANNELS),
            ('lidar_channel', LIDAR_CHANNELS),
        ]
    )
    unet = OrderedDict(
        [
            ('in_channel', STUDENT_IN_CHANNELS),
            ('out_channel', STUDENT_IN_CHANNELS),
            ('inner_channel', inner),
            ('channel_multiplier', mult),
            ('attn_res', [16]),
            ('res_blocks', STUDENT_LAYERS_PER_BLOCK),
            ('dropout', 0.2),
        ]
    )
    beta = OrderedDict(
        [
            ('schedule', 'cosine'),
            ('n_timestep', 2000),
            ('linear_start', 1e-6),
            ('linear_end', 1e-2),
        ]
    )
    lora = OrderedDict(
        [
            ('enable', False),
            ('r', 8),
            ('alpha', 8),
            ('dropout', 0.1),
            ('target_module_names', ['attn', 'noise_level_mlp', 'noise_func']),
        ]
    )
    model = OrderedDict(
        [
            ('which_model_G', 'sr3'),
            ('finetune_norm', False),
            ('image_size', STUDENT_SIZE),
            ('channels', STUDENT_IN_CHANNELS),
            ('conditional', False),
            ('unet', unet),
            ('beta_schedule', beta),
            ('lora', lora),
            ('loss', 'l2'),
        ]
    )
    train = OrderedDict(
        [
            ('n_epoch', NUM_EPOCHS),
            ('train_print_freq', 1),
            ('save_checkpoint_freq', 100),
            ('save_epoch_start', 200),
            ('test_freq', 5),
            ('val_print_freq', 1),
            ('save_checkpoint_threshold', 0.8),
            (
                'optimizer',
                OrderedDict(
                    [
                        ('type', 'adamw'),
                        ('lr', LEARNING_RATE),
                        ('weight_decay', WEIGHT_DECAY),
                        ('betas', OPTIMIZER_BETAS),
                    ]
                ),
            ),
            ('scheduler', sched),
        ]
    )
    model_cls = OrderedDict(
        [
            ('feat_scales', []),
            ('init_type', CLS_INIT_TYPE),
            ('scale', CLS_INIT_SCALE),
            ('out_channels', NUM_CLASSES),
            ('output_cm_size', CLS_OUTPUT_CM_SIZE),
            ('t', []),
            ('token_dim', CLS_TOKEN_DIM),
            ('transformer_heads', CLS_TRANSFORMER_HEADS),
            ('transformer_layers', CLS_TRANSFORMER_LAYERS),
            ('transformer_ff_dim', CLS_TRANSFORMER_FF_DIM),
            ('transformer_dropout', CLS_TRANSFORMER_DROPOUT),
            ('head_hidden', CLS_HEAD_HIDDEN),
            ('resume_state', None),
            ('enabled_modalities', list(ENABLED_MODALITIES)),
            ('modality_combo', MODALITY_COMBO),
            ('rgb_source', 'student'),
            ('rgb_student_checkpoint', RGB_STUDENT_CHECKPOINT or None),
            ('rgb_to_lidar_guidance_mode', RGB_TO_LIDAR_GUIDANCE_MODE),
            ('center_distance_bias_alpha', CENTER_DISTANCE_BIAS_ALPHA),
        ]
    )
    path = OrderedDict(
        [
            ('log', 'logs'),
            ('tb_logger', 'tb_logger'),
            ('results', 'results'),
            ('checkpoint', 'checkpoint'),
        ]
    )
    cast3 = OrderedDict(
        [
            ('lidar_hidden', LIDAR_PROJ_HIDDEN_CFG),
            ('lidar_extra_blocks', LIDAR_EXTRA_BLOCKS_CFG),
            ('hsi_residual_blocks', HSI_RESIDUAL_BLOCKS_CFG),
            ('hsi_conv_hidden', HSI_CONV_HIDDEN_CFG),
            ('hsi_se_ratio', HSI_SE_RATIO_CFG),
            ('hsi_agg_mode', HSI_AGG_MODE_CFG),
        ]
    )

    return OrderedDict(
        [
            ('name', 'cls'),
            ('phase', 'train'),
            ('autodl', True),
            ('gpu_ids', [0]),
            ('checkpoints_dir', ''),
            ('path', path),
            ('dataset', ds),
            ('resume_state', None),
            ('train', train),
            ('model', model),
            ('module_cast3', cast3),
            ('model_cls', model_cls),
        ]
    )


opt = Logger.dict_to_nonedict(build_opt())
opt = Logger.dict_to_nonedict(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# 覆盖 student 相关维度（与 build_opt 一致，便于单处改 STUDENT_*）
opt['model']['image_size'] = STUDENT_SIZE
opt['model']['unet']['channel_multiplier'] = [c // STUDENT_CHANNELS[0] for c in STUDENT_CHANNELS]
opt['train']['optimizer']['lr'] = LEARNING_RATE
opt['train']['n_epoch'] = NUM_EPOCHS
if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
    opt['train']['clip_grad_norm'] = CLIP_GRAD_NORM
opt['train']['optimizer']['weight_decay'] = WEIGHT_DECAY

modal = list(opt['dataset'].get('modalities') or ['hsi', 'lidar'])
opt['dataset']['modalities'] = list(ENABLED_MODALITIES)

opt['model_cls']['use_center_loss'] = USE_CENTER_LOSS
opt['model_cls']['loss_weight_global'] = LOSS_WEIGHT_GLOBAL
opt['model_cls']['loss_weight_center'] = LOSS_WEIGHT_CENTER
opt['model_cls']['center_distance_bias_alpha'] = float(CENTER_DISTANCE_BIAS_ALPHA)
opt['model_cls']['center_distance_bias_tau'] = float(CENTER_DISTANCE_BIAS_TAU)


def _apply_mmdiff_env_overrides():
    """
    在 build_opt 之后覆盖 loss 与 LiDAR 投影宽度（与 LIDAR_PROJ_HIDDEN_CFG 一致）。
    MMDIFF_LOSS_WEIGHT_GLOBAL（CENTER 始终为 1-GLOBAL；不再读取 MMDIFF_LOSS_WEIGHT_CENTER）
    MMDIFF_LIDAR_HIDDEN → LIDAR_PROJ_HIDDEN_CFG
    MMDIFF_LIDAR_EXTRA_BLOCKS → LIDAR_EXTRA_BLOCKS_CFG（LiDAR stem 后空间残差块数）
    MMDIFF_NUM_EPOCHS → NUM_EPOCHS 与 opt['train']['n_epoch']
    MMDIFF_HSI_RESIDUAL_BLOCKS → HSI_RESIDUAL_BLOCKS_CFG 与 opt['module_cast3']
    MMDIFF_HSI_CONV_HIDDEN → HSI_CONV_HIDDEN_CFG 与 opt['module_cast3']
    MMDIFF_HSI_SE_RATIO → HSI_SE_RATIO_CFG 与 opt['module_cast3']
    MMDIFF_HSI_AGG_MODE → HSI_AGG_MODE_CFG 与 opt['module_cast3']（mean | attn_pool | multi_token）
    MMDIFF_BATCH_SIZE → BATCH_SIZE 与 opt['dataset']['batch_size']
    MMDIFF_LEARNING_RATE → LEARNING_RATE 与 opt['train']['optimizer']['lr']
    MMDIFF_WEIGHT_DECAY → WEIGHT_DECAY 与 opt['train']['optimizer']['weight_decay']
    MMDIFF_CLS_TOKEN_DIM → CLS_TOKEN_DIM 与 opt['model_cls']['token_dim']（须能被 transformer_heads 整除）
    MMDIFF_CLS_HEAD_HIDDEN → CLS_HEAD_HIDDEN 与 opt['model_cls']['head_hidden']
    MMDIFF_CLS_TRANSFORMER_LAYERS → CLS_TRANSFORMER_LAYERS 与 opt['model_cls']['transformer_layers']
    MMDIFF_CLS_TRANSFORMER_FF_DIM → CLS_TRANSFORMER_FF_DIM 与 opt['model_cls']['transformer_ff_dim']
    MMDIFF_CENTER_DISTANCE_BIAS_ALPHA → CENTER_DISTANCE_BIAS_ALPHA 与 opt['model_cls']['center_distance_bias_alpha']
    MMDIFF_CENTER_DISTANCE_BIAS_TAU → CENTER_DISTANCE_BIAS_TAU 与 opt['model_cls']['center_distance_bias_tau']（指数衰减尺度，>0）
    MMDIFF_EARLY_STOPPING_PATIENCE → EARLY_STOPPING_PATIENCE（0=关闭早停）
    MMDIFF_RESUME_CHECKPOINT → 覆盖 RESUME_CHECKPOINT
    MMDIFF_RGB_TO_LIDAR_GUIDANCE → RGB_TO_LIDAR_GUIDANCE_MODE 与 opt['model_cls']['rgb_to_lidar_guidance_mode']（none|film）
    MMDIFF_SCHEDULER_LR_TOTAL_STEPS → opt['scheduler_lr_total_steps']（续训边界；旧 checkpoint 无该字段时手动设）
    MMDIFF_SCHEDULER_NAME / MMDIFF_SCHED_STEP_RATIOS / MMDIFF_SCHED_GAMMAS / MMDIFF_SCHED_COSINE_* → 见文件头 SCHEDULER_*
    MMDIFF_FREEZE_RGB_STUDENT=1 → 冻结轻量 RGB 编码器并重建优化器（续训 resume 时不走该分支）
    MMDIFF_RANDOM_SEED → RANDOM_SEED（torch/np 与划分等）
    MMDIFF_FORWARD_TRACE / MMDIFF_LOG_DATAFLOW=1 → model.log 中按前向打印子模块 in/out 形状（动态数据流）
    MMDIFF_FORWARD_TRACE_DEPTH（默认 3）MMDIFF_FORWARD_TRACE_MAX_FORWARDS（默认 1）
    Memory 压缩（由 model/multimodal.py 读取环境变量，不改 opt）：MMDIFF_MEMORY_COMPRESS_MODE=none|grid|linear|latent，
    MMDIFF_MEMORY_GRID_SIZE，MMDIFF_MEMORY_COMPRESS_TOKENS，MMDIFF_MEMORY_KEEP_CENTER_TOKEN=0|1
    """
    g = globals()

    def _float(name, key):
        v = os.environ.get(name)
        if v is None or v.strip() == '':
            return
        g[key] = float(v)

    def _int(name, key):
        v = os.environ.get(name)
        if v is None or v.strip() == '':
            return
        g[key] = int(v)

    def _bool_env(name, key):
        v = os.environ.get(name)
        if v is None or v.strip() == '':
            return
        g[key] = v.strip().lower() in ('1', 'true', 'yes', 'y')

    def _str_env(name, key):
        v = os.environ.get(name)
        if v is None or v.strip() == '':
            return
        g[key] = v.strip()

    _float('MMDIFF_LOSS_WEIGHT_GLOBAL', 'LOSS_WEIGHT_GLOBAL')
    _sync_loss_weights_from_global()
    _float('MMDIFF_LEARNING_RATE', 'LEARNING_RATE')
    _float('MMDIFF_WEIGHT_DECAY', 'WEIGHT_DECAY')
    _float('MMDIFF_CENTER_DISTANCE_BIAS_ALPHA', 'CENTER_DISTANCE_BIAS_ALPHA')
    _float('MMDIFF_CENTER_DISTANCE_BIAS_TAU', 'CENTER_DISTANCE_BIAS_TAU')
    _int('MMDIFF_LIDAR_HIDDEN', 'LIDAR_PROJ_HIDDEN_CFG')
    _int('MMDIFF_LIDAR_EXTRA_BLOCKS', 'LIDAR_EXTRA_BLOCKS_CFG')
    _int('MMDIFF_NUM_EPOCHS', 'NUM_EPOCHS')
    _int('MMDIFF_HSI_RESIDUAL_BLOCKS', 'HSI_RESIDUAL_BLOCKS_CFG')
    _int('MMDIFF_HSI_CONV_HIDDEN', 'HSI_CONV_HIDDEN_CFG')
    _int('MMDIFF_HSI_SE_RATIO', 'HSI_SE_RATIO_CFG')
    _str_env('MMDIFF_HSI_AGG_MODE', 'HSI_AGG_MODE_CFG')
    _int('MMDIFF_BATCH_SIZE', 'BATCH_SIZE')
    _int('MMDIFF_RANDOM_SEED', 'RANDOM_SEED')
    _int('MMDIFF_EARLY_STOPPING_PATIENCE', 'EARLY_STOPPING_PATIENCE')
    _int('MMDIFF_SCHEDULER_LR_TOTAL_STEPS', 'SCHEDULER_LR_TOTAL_STEPS')
    _int('MMDIFF_CLS_TOKEN_DIM', 'CLS_TOKEN_DIM')
    _int('MMDIFF_CLS_HEAD_HIDDEN', 'CLS_HEAD_HIDDEN')
    _int('MMDIFF_CLS_TRANSFORMER_LAYERS', 'CLS_TRANSFORMER_LAYERS')
    _int('MMDIFF_CLS_TRANSFORMER_FF_DIM', 'CLS_TRANSFORMER_FF_DIM')

    lh = int(g['LIDAR_PROJ_HIDDEN_CFG'])
    if lh < 1:
        raise ValueError(f'LiDAR 投影 lidar_hidden 须 >= 1，当前 {lh}')
    opt['module_cast3']['lidar_hidden'] = lh
    leb = max(0, int(g['LIDAR_EXTRA_BLOCKS_CFG']))
    opt['module_cast3']['lidar_extra_blocks'] = leb
    rb = max(0, int(g['HSI_RESIDUAL_BLOCKS_CFG']))
    hh = int(g['HSI_CONV_HIDDEN_CFG'])
    if hh < 1:
        raise ValueError(f'HSI_CONV_HIDDEN_CFG / MMDIFF_HSI_CONV_HIDDEN 须 >= 1，当前 {hh}')
    sr = int(g['HSI_SE_RATIO_CFG'])
    if sr < 0:
        raise ValueError(f'HSI_SE_RATIO_CFG / MMDIFF_HSI_SE_RATIO 须 >= 0，当前 {sr}')
    agg = str(g.get('HSI_AGG_MODE_CFG') or 'multi_token').strip().lower()
    if agg not in ('mean', 'attn_pool', 'multi_token'):
        raise ValueError(
            f'HSI_AGG_MODE_CFG / MMDIFF_HSI_AGG_MODE 须为 mean|attn_pool|multi_token，当前 {agg!r}'
        )
    opt['module_cast3']['hsi_residual_blocks'] = rb
    opt['module_cast3']['hsi_conv_hidden'] = hh
    opt['module_cast3']['hsi_se_ratio'] = sr
    opt['module_cast3']['hsi_agg_mode'] = agg
    g['HSI_AGG_MODE_CFG'] = agg
    opt['model_cls']['loss_weight_global'] = float(g['LOSS_WEIGHT_GLOBAL'])
    opt['model_cls']['loss_weight_center'] = float(g['LOSS_WEIGHT_CENTER'])

    ne = int(g['NUM_EPOCHS'])
    if ne < 1:
        raise ValueError(f'NUM_EPOCHS / MMDIFF_NUM_EPOCHS 须 >= 1，当前 {ne}')
    opt['train']['n_epoch'] = ne

    bs = int(g['BATCH_SIZE'])
    if bs < 1:
        raise ValueError(f'BATCH_SIZE / MMDIFF_BATCH_SIZE 须 >= 1，当前 {bs}')
    opt['dataset']['batch_size'] = bs
    slr_steps = int(g.get('SCHEDULER_LR_TOTAL_STEPS') or 0)
    if slr_steps > 0:
        opt['scheduler_lr_total_steps'] = slr_steps

    opt['train']['optimizer']['lr'] = float(g['LEARNING_RATE'])
    opt['train']['optimizer']['weight_decay'] = float(g['WEIGHT_DECAY'])

    td = int(g['CLS_TOKEN_DIM'])
    hh_cls = int(g['CLS_HEAD_HIDDEN'])
    if td < 1:
        raise ValueError(f'CLS_TOKEN_DIM / MMDIFF_CLS_TOKEN_DIM 须 >= 1，当前 {td}')
    if hh_cls < 1:
        raise ValueError(f'CLS_HEAD_HIDDEN / MMDIFF_CLS_HEAD_HIDDEN 须 >= 1，当前 {hh_cls}')
    nhead = int(g['CLS_TRANSFORMER_HEADS'])
    if td % nhead != 0:
        raise ValueError(
            f'token_dim={td} 须能被 CLS_TRANSFORMER_HEADS={nhead} 整除（MultimodalClassifier 约束）'
        )
    opt['model_cls']['token_dim'] = td
    opt['model_cls']['head_hidden'] = hh_cls

    n_layers = int(g['CLS_TRANSFORMER_LAYERS'])
    ff_dim = int(g['CLS_TRANSFORMER_FF_DIM'])
    if n_layers < 1:
        raise ValueError(f'CLS_TRANSFORMER_LAYERS / MMDIFF_CLS_TRANSFORMER_LAYERS 须 >= 1，当前 {n_layers}')
    if ff_dim < 1:
        raise ValueError(f'CLS_TRANSFORMER_FF_DIM / MMDIFF_CLS_TRANSFORMER_FF_DIM 须 >= 1，当前 {ff_dim}')
    opt['model_cls']['transformer_layers'] = n_layers
    opt['model_cls']['transformer_ff_dim'] = ff_dim

    cba = float(g['CENTER_DISTANCE_BIAS_ALPHA'])
    if cba < 0.0:
        raise ValueError(
            f'CENTER_DISTANCE_BIAS_ALPHA / MMDIFF_CENTER_DISTANCE_BIAS_ALPHA 须 >= 0，当前 {cba}'
        )
    opt['model_cls']['center_distance_bias_alpha'] = cba

    cbt = float(g['CENTER_DISTANCE_BIAS_TAU'])
    if cbt <= 0.0:
        raise ValueError(
            f'CENTER_DISTANCE_BIAS_TAU / MMDIFF_CENTER_DISTANCE_BIAS_TAU 须 > 0，当前 {cbt}'
        )
    opt['model_cls']['center_distance_bias_tau'] = cbt

    rsc = (os.environ.get('MMDIFF_RGB_STUDENT_CHECKPOINT') or '').strip()
    if rsc:
        g['RGB_STUDENT_CHECKPOINT'] = rsc
        opt['model_cls']['rgb_student_checkpoint'] = rsc

    resume_p = (os.environ.get('MMDIFF_RESUME_CHECKPOINT') or '').strip()
    if resume_p:
        g['RESUME_CHECKPOINT'] = resume_p

    _env_r2l = (os.environ.get('MMDIFF_RGB_TO_LIDAR_GUIDANCE') or '').strip()
    r2l = _normalize_rgb_to_lidar_guidance(_env_r2l or g.get('RGB_TO_LIDAR_GUIDANCE_MODE') or 'none')
    g['RGB_TO_LIDAR_GUIDANCE_MODE'] = r2l
    opt['model_cls']['rgb_to_lidar_guidance_mode'] = r2l


_apply_mmdiff_env_overrides()

EARLY_STOPPING_PATIENCE = max(0, int(EARLY_STOPPING_PATIENCE))

MULTIMODAL_ABLATION_LOG_LINE = (
    f"multimodal_ablation: modalities={MODALITY_COMBO} axis={_EFFECTIVE_ABLATION_AXIS or 'none'} "
    f"index={_EFFECTIVE_ABLATION_INDEX if _EFFECTIVE_ABLATION_AXIS else '-'} | "
    f"lidar_hidden={LIDAR_PROJ_HIDDEN_CFG} lidar_extra_blocks={LIDAR_EXTRA_BLOCKS_CFG} "
    f"hsi_res_blocks={HSI_RESIDUAL_BLOCKS_CFG} hsi_conv_hidden={HSI_CONV_HIDDEN_CFG} "
    f"hsi_se_ratio={HSI_SE_RATIO_CFG} hsi_agg_mode={HSI_AGG_MODE_CFG} | "
    f"rgb_to_lidar_guidance={RGB_TO_LIDAR_GUIDANCE_MODE} | "
    f"loss_global/center={LOSS_WEIGHT_GLOBAL}/{LOSS_WEIGHT_CENTER} | "
    f"center_dist_bias_a={CENTER_DISTANCE_BIAS_ALPHA} tau={CENTER_DISTANCE_BIAS_TAU}"
)
