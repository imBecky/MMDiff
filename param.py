from collections import OrderedDict
import os
from pathlib import Path
import torch
import utils.logger as Logger

# ---------------------------------------------------------------------------
# 数据与训练（常用修改处）
# ---------------------------------------------------------------------------
# LR 两阶衰减（相对总 optimizer step = epoch × len(train_loader)）：
# 与 300epoch 参考对齐：约 79% 总步数 ×0.1；第二段 gamma=1.0 等价单次衰减；1.5e-3 补偿更少总步数。
# run.sh 深度对照时固定本段；HSI 结构通过 MMDIFF_HSI_* 覆盖 module_cast3。
SCHED_STEP_RATIOS = [0.6, 0.75]
SCHED_GAMMAS = [0.4, 0.2]
CLIP_GRAD_NORM = 1.0
EVAL_VAL_START_EPOCH = 10
LEARNING_RATE = 3e-3

NUM_WORKERS = 13
BATCH_SIZE = 512
NUM_EPOCHS = 300
# 续训时 LambdaLR 衰减边界用：与首次训练一致的总 step 数（通常由 checkpoint 自动写入；旧断点可 export MMDIFF_SCHEDULER_LR_TOTAL_STEPS）
SCHEDULER_LR_TOTAL_STEPS = 0

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
USE_RGB_PATCHES = TRAIN_RGB_PATCHES_PATH.is_file()
RGB_CHANNELS = 3
RUN_NAME_PREFIX = 'cls'
# 验证集最佳权重保存在本次 run 的断点目录下：{run_ckps_dir}/{BEST_MODEL_FILENAME}
BEST_MODEL_FILENAME = 'best_model.pt'
# 仅兼容旧说明；训练流程不再写入此路径
MODEL_PATH = Path('./models/model.pt')
# 训练断点根目录（每次运行会创建子目录 run_tag，内含 checkpoint-<epoch>、final、best_model.pt）
CKPS_DIR = Path('../../autodl-tmp/classifier')
# 每 N 个 epoch 保存一次断点；仅当 run_selection 为真（epoch>=EVAL_VAL_START_EPOCH、有 val、且 train_acc>=EVAL_MIN_TRAIN_ACC）时保存。0 表示不按间隔保存（仍会写 final）。若希望尽快只靠准确率门槛触发验证与断点，可将 EVAL_VAL_START_EPOCH 设为 0
SAVE_EVERY_EPOCH = 5
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

HSI_CHANNELS = 50
LIDAR_CHANNELS = 1
# Houston2018 前景地物类为 20；GT 中背景常为 0，patch 仅含 y>0 像素，标签平移后为 0....19。
NUM_CLASSES = 20

RANDOM_SEED = 42
OPTIMIZER_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
TRAIN_LOG_INTERVAL = 20
EVAL_LOG_INTERVAL = 20

# 训练时检查各模态投影（HSI/LiDAR/RGB 等）在 backward 后是否有非零梯度；用于排查 no_grad 截断或未入图。
# 为 True 时，在 backward 之后、clip 之前按间隔汇总范数并写日志（可选 TensorBoard）。
CHECK_PROJECTION_GRAD = False
CHECK_PROJECTION_GRAD_INTERVAL = 10  # 每 N 个 batch；第 1 个与每 epoch 最后一个 batch 也会记录

VAL_RATIO = 0.1
DIFFUSION_NOISE_MODE = 'deterministic'
DIFFUSION_NORMALIZE_INPUT = True

# 本 epoch 训练准确率低于该值时不跑验证/选集 eval（0~1）；目标「先过拟合训练集」
EVAL_MIN_TRAIN_ACC = 0
# 过门槛后每 N 个 epoch 跑一次 eval（test 集大时减少 eval 频率；0 表示每 epoch 都跑）
EVAL_INTERVAL_EPOCHS = 5

USE_CENTER_LOSS = True
LOSS_WEIGHT_GLOBAL = 0.2
LOSS_WEIGHT_CENTER = 0.8

# 监督对比损失 SupCon（Khosla et al.）：在 c_rep 上接 projection，训练集双视图（独立随机旋转）
# 需要 RGB+扩散特征；与 CE 联合：L = L_ce + SUPCON_WEIGHT * L_supcon
USE_SUPCON = True
SUPCON_WEIGHT = 0.1
SUPCON_TEMPERATURE = 0.07
SUPCON_PROJ_DIM = 128

# 清单实验：仅 center/global loss 权重对比；生效：MULTIMODAL_ABLATION_AXIS / INDEX 或 MMDIFF_ABLATION_* 环境变量
CENTER_GLOBAL_ABLATION = ((0.2, 0.8), (0.3, 0.7))
MULTIMODAL_ABLATION_AXIS = None  # None | 'center_global'
MULTIMODAL_ABLATION_INDEX = 0

# LiDAR 形态编码器 stem 隐藏通道（model/multimodal.py LidarMorphEncoder）
LIDAR_PROJ_HIDDEN_CFG = 16
# HSI 1D 光谱编码（model/multimodal.py HSICenterSpectralEncoder）：stem 后残差块数、通道宽、SE 比
# 更深：增大 HSI_RESIDUAL_BLOCKS_CFG（例如 4～6）；加深时可同步略增 HSI_CONV_HIDDEN_CFG。
# 更宽：再调大 HSI_CONV_HIDDEN_CFG（须 ≥32，代码里 max(32, …)）。
# 想回到接近旧版容量：设 HSI_RESIDUAL_BLOCKS_CFG = 0，并把 HSI_CONV_HIDDEN_CFG 设回 64。
HSI_RESIDUAL_BLOCKS_CFG = 5
HSI_CONV_HIDDEN_CFG = 128
HSI_SE_RATIO_CFG = 8

_EFFECTIVE_ABLATION_AXIS = None
_EFFECTIVE_ABLATION_INDEX = None


def _apply_multimodal_ablation():
    """按轴覆盖 LOSS_WEIGHT_*；未选轴时保持上方默认值。"""
    global LOSS_WEIGHT_GLOBAL, LOSS_WEIGHT_CENTER
    global _EFFECTIVE_ABLATION_AXIS, _EFFECTIVE_ABLATION_INDEX
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
        LOSS_WEIGHT_GLOBAL, LOSS_WEIGHT_CENTER = CENTER_GLOBAL_ABLATION[idx]
    else:
        raise ValueError(
            f'未知 MULTIMODAL_ABLATION_AXIS / MMDIFF_ABLATION_AXIS={axis!r}，应为 "center_global"'
        )
    _EFFECTIVE_ABLATION_INDEX = idx


_apply_multimodal_ablation()

# Student diffusion（与 ../GFDiff/train_distill.py 中学生 UNet2DModel 一致）
# 学生由 train_distill 中 UNet2DModel(block_out_channels=student_channels, ...) 构造，
# checkpoint 为 diffusers DDPMPipeline.save_pretrained；此处为与蒸馏脚本对齐的元数据。
STUDENT_CHECKPOINT = Path('../../autodl-fs/student32/final')
STUDENT_SIZE = 32
STUDENT_IN_CHANNELS = 3
STUDENT_CHANNELS = (128, 256, 512, 512)
STUDENT_LAYERS_PER_BLOCK = 2
STUDENT_NUM_TRAIN_TIMESTEPS = 1000

# 与 ../GFDiff/train_distill.py 中 DEFAULT_ALIGN_LAYERS 一致：UNet 子模块名（非整数下标）
FEAT_SCALES = [
    'mid_block',
    'down_blocks.1',
    'down_blocks.2',
    'up_blocks.0',
    'up_blocks.1',
]

def _cls_diffusion_timesteps_from_env():
    """run.sh 可 export MMDIFF_DIFFUSION_TIMESTEPS=50,100,150,200（逗号分隔）覆盖默认。"""
    raw = (os.environ.get('MMDIFF_DIFFUSION_TIMESTEPS') or '').strip()
    if not raw:
        return [50, 100]
    ts = [int(x.strip()) for x in raw.split(',') if x.strip()]
    if not ts:
        raise ValueError(
            'MMDIFF_DIFFUSION_TIMESTEPS 须为逗号分隔整数，例如 50,100,150,200'
        )
    return ts


CLS_DIFFUSION_TIMESTEPS = _cls_diffusion_timesteps_from_env()
CLS_INIT_TYPE = 'kaiming'
CLS_INIT_SCALE = 0.1
CLS_OUTPUT_CM_SIZE = 3
# 多模态 Transformer 分类头（见 model/multimodal.py）
CLS_TOKEN_DIM = 256
CLS_TRANSFORMER_HEADS = 4
CLS_TRANSFORMER_LAYERS = 2
CLS_TRANSFORMER_FF_DIM = 512
CLS_TRANSFORMER_DROPOUT = 0.1
CLS_HEAD_HIDDEN = 128


def build_opt():
    """完整训练配置 """
    inner = STUDENT_CHANNELS[0]
    mult = [c // inner for c in STUDENT_CHANNELS]
    # 两阶分段衰减，见 SCHED_STEP_RATIOS / SCHED_GAMMAS；仍支持仅 constant_ratio+gamma 的旧 dict
    sched = OrderedDict(
        [
            ('name', 'piecewise_two_step'),
            ('step_ratios', list(SCHED_STEP_RATIOS)),
            ('gammas', list(SCHED_GAMMAS)),
            ('constant_ratio', 0.8),
            ('gamma', 0.1),
        ]
    )
    ds = OrderedDict(
        [
            ('name', 'Houston2018'),
            ('dataroot', str(DATA_DIR)),
            ('modalities', ['hsi', 'lidar']),
            ('resolution', 3),
            ('batch_size', BATCH_SIZE),
            ('num_workers', 14),
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
            ('save_checkpoint_freq', 5),
            ('save_epoch_start', 5),
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
            ('feat_scales', list(FEAT_SCALES)),
            ('init_type', CLS_INIT_TYPE),
            ('scale', CLS_INIT_SCALE),
            ('out_channels', NUM_CLASSES),
            ('output_cm_size', CLS_OUTPUT_CM_SIZE),
            ('t', list(CLS_DIFFUSION_TIMESTEPS)),
            ('token_dim', CLS_TOKEN_DIM),
            ('transformer_heads', CLS_TRANSFORMER_HEADS),
            ('transformer_layers', CLS_TRANSFORMER_LAYERS),
            ('transformer_ff_dim', CLS_TRANSFORMER_FF_DIM),
            ('transformer_dropout', CLS_TRANSFORMER_DROPOUT),
            ('head_hidden', CLS_HEAD_HIDDEN),
            ('use_supcon', USE_SUPCON),
            ('supcon_weight', SUPCON_WEIGHT),
            ('supcon_temperature', SUPCON_TEMPERATURE),
            ('supcon_proj_dim', SUPCON_PROJ_DIM),
            ('resume_state', None),
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
            ('hsi_residual_blocks', HSI_RESIDUAL_BLOCKS_CFG),
            ('hsi_conv_hidden', HSI_CONV_HIDDEN_CFG),
            ('hsi_se_ratio', HSI_SE_RATIO_CFG),
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
if USE_RGB_PATCHES:
    if 'rgb' not in modal:
        modal.append('rgb')
else:
    modal = [m for m in modal if m != 'rgb']
opt['dataset']['modalities'] = modal

opt['model_cls']['use_center_loss'] = USE_CENTER_LOSS
opt['model_cls']['loss_weight_global'] = LOSS_WEIGHT_GLOBAL
opt['model_cls']['loss_weight_center'] = LOSS_WEIGHT_CENTER
opt['model_cls']['use_supcon'] = USE_SUPCON
opt['model_cls']['supcon_weight'] = SUPCON_WEIGHT
opt['model_cls']['supcon_temperature'] = SUPCON_TEMPERATURE
opt['model_cls']['supcon_proj_dim'] = SUPCON_PROJ_DIM


def _apply_mmdiff_env_overrides():
    """
    在 build_opt 之后覆盖 loss 与 LiDAR 投影宽度（与 LIDAR_PROJ_HIDDEN_CFG 一致）。
    MMDIFF_LOSS_WEIGHT_GLOBAL / MMDIFF_LOSS_WEIGHT_CENTER
    MMDIFF_LIDAR_HIDDEN → LIDAR_PROJ_HIDDEN_CFG
    MMDIFF_SUPCON_WEIGHT → SUPCON_WEIGHT
    MMDIFF_USE_SUPCON → USE_SUPCON（0/1）
    MMDIFF_NUM_EPOCHS → NUM_EPOCHS 与 opt['train']['n_epoch']
    MMDIFF_HSI_RESIDUAL_BLOCKS → HSI_RESIDUAL_BLOCKS_CFG 与 opt['module_cast3']
    MMDIFF_HSI_CONV_HIDDEN → HSI_CONV_HIDDEN_CFG 与 opt['module_cast3']
    MMDIFF_HSI_SE_RATIO → HSI_SE_RATIO_CFG 与 opt['module_cast3']
    MMDIFF_RESUME_CHECKPOINT → 覆盖 RESUME_CHECKPOINT
    MMDIFF_SCHEDULER_LR_TOTAL_STEPS → opt['scheduler_lr_total_steps']（续训边界；旧 checkpoint 无该字段时手动设）
    （扩散 t 列表由模块加载时读取 MMDIFF_DIFFUSION_TIMESTEPS，见 _cls_diffusion_timesteps_from_env）
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

    _float('MMDIFF_LOSS_WEIGHT_GLOBAL', 'LOSS_WEIGHT_GLOBAL')
    _float('MMDIFF_LOSS_WEIGHT_CENTER', 'LOSS_WEIGHT_CENTER')
    _float('MMDIFF_SUPCON_WEIGHT', 'SUPCON_WEIGHT')
    _bool_env('MMDIFF_USE_SUPCON', 'USE_SUPCON')
    _int('MMDIFF_LIDAR_HIDDEN', 'LIDAR_PROJ_HIDDEN_CFG')
    _int('MMDIFF_NUM_EPOCHS', 'NUM_EPOCHS')
    _int('MMDIFF_HSI_RESIDUAL_BLOCKS', 'HSI_RESIDUAL_BLOCKS_CFG')
    _int('MMDIFF_HSI_CONV_HIDDEN', 'HSI_CONV_HIDDEN_CFG')
    _int('MMDIFF_HSI_SE_RATIO', 'HSI_SE_RATIO_CFG')
    _int('MMDIFF_SCHEDULER_LR_TOTAL_STEPS', 'SCHEDULER_LR_TOTAL_STEPS')

    lh = int(g['LIDAR_PROJ_HIDDEN_CFG'])
    if lh < 1:
        raise ValueError(f'LiDAR 投影 lidar_hidden 须 >= 1，当前 {lh}')
    opt['module_cast3']['lidar_hidden'] = lh
    rb = max(0, int(g['HSI_RESIDUAL_BLOCKS_CFG']))
    hh = int(g['HSI_CONV_HIDDEN_CFG'])
    if hh < 1:
        raise ValueError(f'HSI_CONV_HIDDEN_CFG / MMDIFF_HSI_CONV_HIDDEN 须 >= 1，当前 {hh}')
    sr = max(1, int(g['HSI_SE_RATIO_CFG']))
    opt['module_cast3']['hsi_residual_blocks'] = rb
    opt['module_cast3']['hsi_conv_hidden'] = hh
    opt['module_cast3']['hsi_se_ratio'] = sr
    opt['model_cls']['loss_weight_global'] = float(g['LOSS_WEIGHT_GLOBAL'])
    opt['model_cls']['loss_weight_center'] = float(g['LOSS_WEIGHT_CENTER'])
    opt['model_cls']['use_supcon'] = bool(g['USE_SUPCON'])
    opt['model_cls']['supcon_weight'] = float(g['SUPCON_WEIGHT'])

    ne = int(g['NUM_EPOCHS'])
    if ne < 1:
        raise ValueError(f'NUM_EPOCHS / MMDIFF_NUM_EPOCHS 须 >= 1，当前 {ne}')
    opt['train']['n_epoch'] = ne
    slr_steps = int(g.get('SCHEDULER_LR_TOTAL_STEPS') or 0)
    if slr_steps > 0:
        opt['scheduler_lr_total_steps'] = slr_steps

    resume_p = (os.environ.get('MMDIFF_RESUME_CHECKPOINT') or '').strip()
    if resume_p:
        g['RESUME_CHECKPOINT'] = resume_p


_apply_mmdiff_env_overrides()

if USE_SUPCON and not USE_RGB_PATCHES:
    raise ValueError(
        'USE_SUPCON=True 需要 RGB 与扩散特征：请准备 train_rgb_patches.npy 并确保 USE_RGB_PATCHES，'
        '或设 USE_SUPCON=False / MMDIFF_USE_SUPCON=0'
    )

MULTIMODAL_ABLATION_LOG_LINE = (
    f"multimodal_ablation: axis={_EFFECTIVE_ABLATION_AXIS or 'none'} "
    f"index={_EFFECTIVE_ABLATION_INDEX if _EFFECTIVE_ABLATION_AXIS else '-'} | "
    f"lidar_hidden={LIDAR_PROJ_HIDDEN_CFG} "
    f"hsi_res_blocks={HSI_RESIDUAL_BLOCKS_CFG} hsi_conv_hidden={HSI_CONV_HIDDEN_CFG} "
    f"hsi_se_ratio={HSI_SE_RATIO_CFG} | "
    f"loss_global/center={LOSS_WEIGHT_GLOBAL}/{LOSS_WEIGHT_CENTER} | "
    f"use_supcon={USE_SUPCON} supcon_w={SUPCON_WEIGHT} tau={SUPCON_TEMPERATURE} | "
    f"fusion=cross"
)
