"""
train_distill.py
跨分辨率知识蒸馏训练脚本：将教师 DDPM（大分辨率）的知识蒸馏到学生 DDPM（小分辨率），
核心是多层中间特征对齐（Feature-based Knowledge Distillation）。

典型用法:
    python train_distill.py \
        --teacher_size 256 --student_size 128 \
        --teacher_checkpoint path/to/teacher_checkpoint \
        --align_layers mid_block,down_blocks.2,up_blocks.1 \
        --lambda_feat 0.5 --lambda_diff 1.0 \
        --fp16
"""
import random
import logging
import numpy as np
import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DPMSolverMultistepScheduler


log = logging.getLogger(__name__)


def setup_logging(log_dir: str = ""):
    """配置 logging：同时输出到终端和 log_dir/train.log（如提供了目录）。返回 FileHandler（若有），用于仅写文件不刷屏。"""
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 避免重复添加 handler（resume 场景下可能被调用多次）
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    fh = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return fh


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# config defaults
# ---------------------------------------------------------------------------
data_dir = "../../autodl-fs/Houston2018/rgb"
in_channels = 3
teacher_size = 64
student_size = 32
batch_size = 32
lr = 1e-4
epochs = 80
save_every = 1000
ckps_dir = "../../autodl-tmp/kd_distill"
log_dir = "../../tf-logs/kd_distill"
num_train_timesteps = 1000
seed = 42
teacher_checkpoint = '../../code/kd/ckps/student64/checkpoint-20000'
# 学生 block 通道数：与教师一致 (128,256,512,512)，学生因分辨率更低、layers_per_block 更少，总参数量仍小于教师
student_channels_default = [128, 256, 512, 512]
# 对齐层：覆盖低层（纹理/边缘）+ 中层 + 高层（语义），层数越多监督越密集但更慢
# 可选层名可通过 model.named_modules() 查看；常用：
#   少层（快）: "mid_block,down_blocks.2,up_blocks.1"
#   多层（推荐）: "mid_block,down_blocks.1,down_blocks.2,up_blocks.0,up_blocks.1"
DEFAULT_ALIGN_LAYERS = "mid_block,down_blocks.1,down_blocks.2,up_blocks.0,up_blocks.1"
use_dpm_solver = True      # 采样时是否使用 DPMSolver 加速（False 则用原始 DDPM 采样）
num_inference_steps = 100   # 仅采样且 DPM 时的推理步数


def get_latest_checkpoint_path(root_ckps_dir: str) -> str:
    """在 root_ckps_dir 下递归寻找最新 checkpoint 目录（checkpoint-<step>）。"""
    root = Path(root_ckps_dir)
    if not root.exists():
        return ""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = []
    for p in root.rglob("checkpoint-*"):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        candidates.append((step, p.stat().st_mtime, str(p)))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# 数据集（双分辨率）
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".png"}


class DualResDataset(Dataset):
    """同一张图片分别 resize 到教师/学生分辨率并返回两份 tensor。"""

    def __init__(self, root_dir: str, teacher_size: int, student_size: int,
                 in_channels: int = 3):
        self.paths = sorted(
            p for p in Path(root_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在 {root_dir} 中未找到任何受支持的图片文件")

        self.color_mode = "RGB" if in_channels == 3 else "L"
        norm = [0.5] * in_channels
        self.teacher_tf = transforms.Compose([
            transforms.Resize((teacher_size, teacher_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm, norm),
        ])
        self.student_tf = transforms.Compose([
            transforms.Resize((student_size, student_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm, norm),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert(self.color_mode)
        return self.teacher_tf(img), self.student_tf(img)


# ---------------------------------------------------------------------------
# 多层特征提取器（forward hook）
# ---------------------------------------------------------------------------
class MultiLayerFeatureExtractor:
    """通过 register_forward_hook 同时捕获模型多个中间层的输出。"""

    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        module_dict = dict(model.named_modules())
        for name in layer_names:
            if name not in module_dict:
                available = [n for n, _ in model.named_modules() if n]
                raise ValueError(f"未找到层 '{name}'。可用: {available}")
            hook = module_dict[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def fn(_module, _inp, out):
            self.features[name] = out[0] if isinstance(out, tuple) else out
        return fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# 特征对齐投影器（MLP）
# ---------------------------------------------------------------------------
class FeatureAlignMLP(nn.Module):
    """两层 1×1 卷积 MLP，将学生特征通道映射到教师通道数。
    第一层：student_ch → hidden_ch（GELU 激活），第二层：hidden_ch → teacher_ch。
    空间维度差异在 forward 中通过双线性插值动态对齐。"""

    def __init__(self, student_ch: int, teacher_ch: int, hidden_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(student_ch, hidden_ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_ch, teacher_ch, kernel_size=1, bias=False),
        )
        for m in self.mlp:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, student_feat: torch.Tensor,
                teacher_feat: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(student_feat)
        if projected.shape[-2:] != teacher_feat.shape[-2:]:
            projected = F.interpolate(
                projected, size=teacher_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return F.mse_loss(projected, teacher_feat.detach())


def probe_feature_shapes(
    model: UNet2DModel, img_size: int, in_channels: int,
    layer_names: list[str], device: torch.device,
) -> dict[str, tuple]:
    """用 dummy 数据做一次前向，获取各对齐层的特征形状 (B, C, H, W)。"""
    extractor = MultiLayerFeatureExtractor(model, layer_names)
    dummy_x = torch.randn(1, in_channels, img_size, img_size, device=device)
    dummy_t = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        model(dummy_x, dummy_t)
    shapes = {name: tuple(extractor.features[name].shape) for name in layer_names}
    extractor.remove()
    return shapes


# ---------------------------------------------------------------------------
# 保存 / 恢复
# ---------------------------------------------------------------------------
def save_student_pipeline(student, noise_scheduler, projectors, save_dir):
    """保存学生模型为 DDPMPipeline + 投影器权重。"""
    pipeline = DDPMPipeline(unet=student, scheduler=noise_scheduler)
    pipeline.save_pretrained(save_dir)
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    dpm_scheduler.save_pretrained(os.path.join(save_dir, "scheduler_dpm"))
    torch.save(projectors.state_dict(), os.path.join(save_dir, "projectors.pt"))


def save_training_state(optimizer, scaler, lr_scheduler,
                        epoch: int, global_step: int, save_dir: str,
                        run_log_dir: str = "", run_ckps_dir: str = ""):
    state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "run_log_dir": run_log_dir,
        "run_ckps_dir": run_ckps_dir,
    }
    torch.save(state, os.path.join(save_dir, "training_state.pt"))


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def sample_images_from_checkpoint(args, device):
    """仅采样模式：从学生 checkpoint 生成样本图片。
    若生成全是噪声，可检查训练时 loss/diffusion 是否在下降，以及 scheduler 的 prediction_type 是否为 epsilon。"""
    ckpt = args.resume_checkpoint
    if not ckpt or not os.path.isdir(ckpt):
        raise FileNotFoundError(f"only_sample=True 需要可用的 resume_checkpoint，当前: {ckpt}")

    log.info(f"[采样] 从学生 checkpoint 加载: {ckpt}")
    pipeline = DDPMPipeline.from_pretrained(ckpt).to(device)
    pipeline.unet.eval()
    pipeline.set_progress_bar_config(disable=False)

    # 保证与训练一致：模型预测的是噪声 epsilon
    sched_config = pipeline.scheduler.config
    if getattr(sched_config, "prediction_type", None) != "epsilon":
        sched_config = dict(sched_config)
        sched_config["prediction_type"] = "epsilon"
    if args.use_dpm_solver:
        scheduler_dpm_path = os.path.join(ckpt, "scheduler_dpm")
        if os.path.isdir(scheduler_dpm_path):
            pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(scheduler_dpm_path)
        else:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sched_config)
        log.info("[采样] 使用 DPMSolverMultistepScheduler 加速采样")
    else:
        if isinstance(sched_config, dict):
            pipeline.scheduler = DDPMScheduler.from_config(sched_config)
        log.info("[采样] 使用原始 DDPM 采样")

    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(args.seed)
    num_steps = args.num_inference_steps if args.use_dpm_solver else 1000
    images = pipeline(batch_size=5, generator=gen, num_inference_steps=num_steps).images
    ckps_str = args.resume_checkpoint.split('/')[-1]
    image_dir = os.path.join(args.log_dir, f"image_{ckps_str}")
    os.makedirs(image_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(image_dir, f"sample_{i + 1}.png")
        img.save(path)
        log.info(f"[采样] 已保存: {path}")


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train(args):
    setup_logging()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.only_sample:
        sample_images_from_checkpoint(args, device)
        return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_tag = f"distill_t{args.teacher_size}_s{args.student_size}_{timestamp}"
    run_ckps_dir = os.path.join(args.ckps_dir, run_tag)
    run_log_dir = os.path.join(args.log_dir, run_tag)

    align_layers = [s.strip() for s in args.align_layers.split(",") if s.strip()]
    log.info(f"[配置] 教师分辨率={args.teacher_size}, 学生分辨率={args.student_size}")
    log.info(f"[配置] 对齐层({len(align_layers)}层): {align_layers}")
    log.info(f"[配置] lambda_diff={args.lambda_diff}, lambda_feat={args.lambda_feat}")

    # ----------------------------------------------------------------
    # 教师模型（冻结）
    # ----------------------------------------------------------------
    log.info(f"[教师] 加载预训练 checkpoint: {args.teacher_checkpoint}")
    teacher_pipe = DDPMPipeline.from_pretrained(args.teacher_checkpoint)
    teacher: UNet2DModel = teacher_pipe.unet.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    log.info(f"[教师] 参数量: {t_params:.2f} M（已冻结）")

    # ----------------------------------------------------------------
    # 学生模型
    # ----------------------------------------------------------------
    student_channels = tuple(args.student_channels)
    n_blocks = len(student_channels)
    down_types = ["DownBlock2D"] * max(n_blocks - 2, 0) + ["AttnDownBlock2D"] * min(n_blocks, 2)
    up_types = list(reversed(
        ["UpBlock2D"] * max(n_blocks - 2, 0) + ["AttnUpBlock2D"] * min(n_blocks, 2)
    ))

    student = UNet2DModel(
        sample_size=args.student_size,
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        block_out_channels=student_channels,
        down_block_types=tuple(down_types),
        up_block_types=tuple(up_types),
        layers_per_block=args.student_layers_per_block,
        attention_head_dim=8,
    ).to(device)
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    log.info(f"[学生] block_out_channels={student_channels}, layers_per_block={args.student_layers_per_block}")
    log.info(f"[学生] 参数量: {s_params:.2f} M")

    # ----------------------------------------------------------------
    # 探测特征维度 → 创建对齐投影器
    # ----------------------------------------------------------------
    t_shapes = probe_feature_shapes(teacher, args.teacher_size, args.in_channels,
                                    align_layers, device)
    s_shapes = probe_feature_shapes(student, args.student_size, args.in_channels,
                                    align_layers, device)

    projectors = nn.ModuleDict()
    proj_key_map: dict[str, str] = {}
    for name in align_layers:
        key = name.replace(".", "_")
        proj_key_map[name] = key
        s_ch, t_ch = s_shapes[name][1], t_shapes[name][1]
        hidden_ch = args.align_hidden_dim if args.align_hidden_dim > 0 else (s_ch + t_ch) // 2
        projectors[key] = FeatureAlignMLP(s_ch, t_ch, hidden_ch)
        log.info(f"  对齐 '{name}': 学生 {s_shapes[name]} →[MLP hidden={hidden_ch}]→ 教师 {t_shapes[name]}")
    projectors = projectors.to(device)

    # ----------------------------------------------------------------
    # 注册 hook
    # ----------------------------------------------------------------
    teacher_extractor = MultiLayerFeatureExtractor(teacher, align_layers)
    student_extractor = MultiLayerFeatureExtractor(student, align_layers)

    # ----------------------------------------------------------------
    # 噪声调度器 / 数据 / 优化器
    # ----------------------------------------------------------------
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    dataset = DualResDataset(args.data_dir, args.teacher_size, args.student_size,
                             args.in_channels)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=20, pin_memory=True, drop_last=True,
    )
    log.info(f"[数据] 共 {len(dataset)} 张图片, 每批 {args.batch_size} 张")

    trainable_params = list(student.parameters()) + list(projectors.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    total_steps = args.epochs * len(dataloader)
    # 前 warmup_steps 步保持恒定学习率，之后 cosine 衰减到 min_lr
    if args.warmup_steps > 0:
        warmup_steps = min(args.warmup_steps, total_steps)
        s1 = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=warmup_steps,
        )
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=args.min_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [s1, s2], milestones=[warmup_steps],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr,
        )
    if args.warmup_steps > 0:
        log.info(f"[LR] 前 {min(args.warmup_steps, total_steps)} 步恒定 lr={args.lr}, 之后 cosine 衰减至 min_lr={args.min_lr}")
    else:
        log.info(f"[LR] Cosine 调度: T_max={total_steps}, eta_min={args.min_lr}")

    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ----------------------------------------------------------------
    # 可选恢复
    # ----------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        log.info(f"[恢复] 从 {args.resume_checkpoint} 恢复学生模型")
        stu_pipe = DDPMPipeline.from_pretrained(args.resume_checkpoint)
        student.load_state_dict(stu_pipe.unet.state_dict(), strict=True)

        proj_path = os.path.join(args.resume_checkpoint, "projectors.pt")
        if os.path.isfile(proj_path):
            projectors.load_state_dict(torch.load(proj_path, map_location=device))
            log.info("[恢复] 投影器权重已加载")

        state_path = os.path.join(args.resume_checkpoint, "training_state.pt")
        if os.path.isfile(state_path):
            ts = torch.load(state_path, map_location=device)
            if "optimizer" in ts:
                optimizer.load_state_dict(ts["optimizer"])
            if "scaler" in ts and isinstance(ts["scaler"], dict):
                scaler.load_state_dict(ts["scaler"])
            if "scheduler" in ts and isinstance(ts["scheduler"], dict):
                lr_scheduler.load_state_dict(ts["scheduler"])
            start_epoch = int(ts.get("epoch", 0))
            global_step = int(ts.get("global_step", 0))

            if ts.get("run_log_dir"):
                run_log_dir = ts["run_log_dir"]
            if ts.get("run_ckps_dir"):
                run_ckps_dir = ts["run_ckps_dir"]
            log.info(f"[恢复] 从上次断点继续: start_epoch={start_epoch}, global_step={global_step}")
            log.info(f"[恢复] Checkpoint 将继续保存至: {run_ckps_dir}")
            log.info(f"[恢复] TB 日志将继续写入原目录（保留上次曲线）: {run_log_dir}")
        else:
            # 老 checkpoint 无 training_state.pt：从目录名推断路径与步数，使 TB 和 epoch 从断点续上
            ckpt_path = Path(args.resume_checkpoint)
            m = re.match(r"^checkpoint-(\d+)$", ckpt_path.name)
            if m:
                run_ckps_dir = str(ckpt_path.parent)
                inferred_tag = ckpt_path.parent.name
                run_log_dir = os.path.join(args.log_dir, inferred_tag)
                global_step = int(m.group(1))
                # 用步数反推起始 epoch，避免重复跑已完成的 epoch
                start_epoch = min(global_step // max(len(dataloader), 1), args.epochs - 1)
                start_epoch = max(0, start_epoch)
                log.info(f"[恢复] 未找到 training_state.pt，从目录名推断: global_step={global_step}, start_epoch={start_epoch}")
                log.info(f"[恢复] Checkpoint 将继续保存至: {run_ckps_dir}")
                log.info(f"[恢复] TB 日志将继续写入: {run_log_dir}")
            else:
                log.info("[恢复] 仅恢复模型权重，epoch/step 从 0 开始")

    os.makedirs(run_ckps_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    file_handler = setup_logging(run_log_dir)
    # 配置类信息在加 FileHandler 之前就打印了，这里补写到 train.log（仅写文件，不刷终端）
    if file_handler is not None:
        file_logger = logging.getLogger("train_distill.file")
        file_logger.setLevel(logging.INFO)
        file_logger.addHandler(file_handler)
        file_logger.propagate = False
        file_logger.info("======== 运行配置 ========")
        file_logger.info(f"[配置] 教师分辨率={args.teacher_size}, 学生分辨率={args.student_size}")
        file_logger.info(f"[配置] 对齐层({len(align_layers)}层): {align_layers}")
        file_logger.info(f"[配置] lambda_diff={args.lambda_diff}, lambda_feat={args.lambda_feat}")
        file_logger.info(f"[教师] checkpoint: {args.teacher_checkpoint}")
        file_logger.info(f"[教师] 参数量: {t_params:.2f} M（已冻结）")
        file_logger.info(f"[学生] block_out_channels={student_channels}, layers_per_block={args.student_layers_per_block}")
        file_logger.info(f"[学生] 参数量: {s_params:.2f} M")
        for name in align_layers:
            file_logger.info(f"  对齐 '{name}': 学生 {s_shapes[name]} → 教师 {t_shapes[name]}")
        file_logger.info(f"[数据] 共 {len(dataset)} 张图片, 每批 {args.batch_size} 张")
        if args.warmup_steps > 0:
            file_logger.info(f"[LR] 前 {min(args.warmup_steps, total_steps)} 步恒定 lr={args.lr}, 之后 cosine 衰减至 min_lr={args.min_lr}")
        else:
            file_logger.info(f"[LR] Cosine 调度: T_max={total_steps}, eta_min={args.min_lr}")
        file_logger.info("============================")

    tb_logger = SummaryWriter(log_dir=run_log_dir)

    # ----------------------------------------------------------------
    # 训练循环
    # ----------------------------------------------------------------
    student.train()
    projectors.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_total, epoch_diff, epoch_feat = 0.0, 0.0, 0.0

        for _step, (teacher_imgs, student_imgs) in enumerate(dataloader):
            teacher_imgs = teacher_imgs.to(device)
            student_imgs = student_imgs.to(device)
            bs = teacher_imgs.shape[0]

            timestep = torch.randint(
                0, args.num_train_timesteps, (bs,), device=device, dtype=torch.long,
            )

            t_noise = torch.randn_like(teacher_imgs)
            s_noise = torch.randn_like(student_imgs)
            noisy_t = noise_scheduler.add_noise(teacher_imgs, t_noise, timestep)
            noisy_s = noise_scheduler.add_noise(student_imgs, s_noise, timestep)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                # 教师前向（无梯度）
                teacher_extractor.clear()
                with torch.no_grad():
                    teacher(noisy_t, timestep, return_dict=False)

                # 学生前向
                student_extractor.clear()
                pred_noise = student(noisy_s, timestep, return_dict=False)[0]

                # 扩散重建损失
                diff_loss = F.mse_loss(pred_noise, s_noise)

                # 多层特征对齐损失
                feat_loss = torch.tensor(0.0, device=device)
                for layer_name in align_layers:
                    t_feat = teacher_extractor.features[layer_name]
                    s_feat = student_extractor.features[layer_name]
                    feat_loss = feat_loss + projectors[proj_key_map[layer_name]](
                        s_feat, t_feat,
                    )
                feat_loss = feat_loss / max(len(align_layers), 1)

                loss = args.lambda_diff * diff_loss + args.lambda_feat * feat_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            epoch_total += loss.item()
            epoch_diff += diff_loss.item()
            epoch_feat += feat_loss.item()
            global_step += 1

            tb_logger.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            tb_logger.add_scalar("loss/total", loss.item(), global_step)
            tb_logger.add_scalar("loss/diffusion", diff_loss.item(), global_step)
            tb_logger.add_scalar("loss/feature", feat_loss.item(), global_step)

            if global_step % 100 == 0:
                log.info(
                    f"  Epoch {epoch + 1}/{args.epochs} | Step {global_step} | "
                    f"Loss {loss.item():.5f} "
                    f"(diff={diff_loss.item():.5f}, feat={feat_loss.item():.5f})"
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(run_ckps_dir, f"checkpoint-{global_step}")
                save_student_pipeline(student, noise_scheduler, projectors, ckpt_dir)
                save_training_state(optimizer, scaler, lr_scheduler,
                                    epoch + 1, global_step, ckpt_dir,
                                    run_log_dir=run_log_dir,
                                    run_ckps_dir=run_ckps_dir)
                log.info(f"  >> Checkpoint 已保存至 {ckpt_dir}")

        n = max(len(dataloader), 1)
        log.info(
            f"Epoch {epoch + 1}/{args.epochs} 完成 | "
            f"Avg Loss: {epoch_total / n:.5f} "
            f"(diff={epoch_diff / n:.5f}, feat={epoch_feat / n:.5f})"
        )
        tb_logger.add_scalar("epoch_loss/total", epoch_total / n, epoch + 1)
        tb_logger.add_scalar("epoch_loss/diffusion", epoch_diff / n, epoch + 1)
        tb_logger.add_scalar("epoch_loss/feature", epoch_feat / n, epoch + 1)

    # ---- 清理 hook ----
    teacher_extractor.remove()
    student_extractor.remove()

    # ---- 保存最终模型 ----
    final_dir = os.path.join(run_ckps_dir, "final")
    save_student_pipeline(student, noise_scheduler, projectors, final_dir)
    log.info(f"训练完成! 学生模型已保存至 {final_dir}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DDPM 跨分辨率知识蒸馏（中间特征对齐）",
    )

    # ---- 蒸馏专属参数 ----
    parser.add_argument("--teacher_size", type=int, default=teacher_size,
                        help=f"教师模型图像分辨率 (默认: {teacher_size})")
    parser.add_argument("--student_size", type=int, default=student_size,
                        help=f"学生模型图像分辨率 (默认: {student_size})")
    parser.add_argument("--teacher_checkpoint", type=str, default=teacher_checkpoint,
                        help="教师模型 DDPMPipeline checkpoint 路径")
    parser.add_argument("--student_channels", type=int, nargs="+",
                        default=student_channels_default,
                        help=f"学生 UNet block_out_channels (默认与教师一致: {student_channels_default})")
    parser.add_argument("--student_layers_per_block", type=int, default=2,
                        help="学生 UNet 每 block 的 ResNet 层数 (默认: 2)")
    parser.add_argument("--align_layers", type=str, default=DEFAULT_ALIGN_LAYERS,
                        help=f"逗号分隔的对齐层名 (默认: {DEFAULT_ALIGN_LAYERS})")
    parser.add_argument("--lambda_diff", type=float, default=1.0,
                        help="扩散损失权重 (默认: 1.0)")
    parser.add_argument("--lambda_feat", type=float, default=0.0005,
                        help="特征对齐损失权重 (默认: 1.0)")
    parser.add_argument("--align_hidden_dim", type=int, default=0,
                        help="特征对齐 MLP 的中间维度；0 表示自动取学生与教师通道数的均值 (默认: 0)")

    # ---- 通用训练参数 ----
    parser.add_argument("--data_dir", type=str, default=data_dir,
                        help=f"图片数据集文件夹路径 (默认: {data_dir})")
    parser.add_argument("--in_channels", type=int, default=in_channels,
                        help=f"图像通道数 (默认: {in_channels})")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help=f"批大小 (默认: {batch_size})")
    parser.add_argument("--lr", type=float, default=lr,
                        help=f"学习率 (默认: {lr})")
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="Cosine 调度最小学习率 eta_min (默认: 0)")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="前 N 步保持恒定学习率，之后 cosine 衰减；0 表示不使用 (默认: 0)")
    parser.add_argument("--epochs", type=int, default=epochs,
                        help=f"训练轮数 (默认: {epochs})")
    parser.add_argument("--save_every", type=int, default=save_every,
                        help=f"每 N 步保存 checkpoint (默认: {save_every})")
    parser.add_argument("--ckps_dir", type=str, default=ckps_dir,
                        help=f"checkpoint 保存目录 (默认: {ckps_dir})")
    parser.add_argument("--log_dir", type=str, default=log_dir,
                        help=f"TensorBoard 日志目录 (默认: {log_dir})")
    parser.add_argument("--fp16", action="store_true",
                        help="启用 fp16 混合精度训练")
    parser.add_argument("--num_train_timesteps", type=int, default=num_train_timesteps,
                        help=f"噪声调度步数 (默认: {num_train_timesteps})")
    parser.add_argument("--seed", type=int, default=seed,
                        help=f"随机种子 (默认: {seed})")
    parser.add_argument("--only_sample", action="store_true",
                        help="仅采样模式：加载学生 checkpoint 生成样本图片")
    parser.add_argument("--use_dpm_solver", action="store_true", default=use_dpm_solver,
                        help="采样时使用 DPMSolver 加速 (默认: True)")
    parser.add_argument("--no_dpm_solver", dest="use_dpm_solver", action="store_false",
                        help="采样时禁用 DPMSolver，使用原始 DDPM 采样")
    parser.add_argument("--num_inference_steps", type=int, default=num_inference_steps,
                        help=f"仅采样且 DPM 时的推理步数 (默认: {num_inference_steps})；DDPM 固定 1000")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="从指定学生 checkpoint 恢复训练/采样")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
