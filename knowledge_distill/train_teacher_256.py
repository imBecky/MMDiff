"""
train_teacher_256.py
从头训练 256x256 轻量级 DDPM 教师模型，并提供蒸馏特征提取接口。
"""
import random
import numpy as np
import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DPMSolverMultistepScheduler

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
data_dir = '../../autodl-fs/Houston2018/rgb'
in_channels = 3
img_size = 256
batch_size = 4
lr = 1e-4
min_lr = 1e-6
epochs = 30
save_every = 2500
ckps_dir = f'../../autodl-tmp/kd_{img_size}'    # to save
log_dir = f'../../tf-logs/kd_{img_size}'
resume_checkpoint = f'../../autodl-tmp/kd_{img_size}'   # to resume
num_train_timesteps = 1000
seed = 42
use_dpm_solver = False             # 采样时是否使用 DPMSolver 加速（False 则用原始 DDPM 采样）


def get_latest_checkpoint_path(root_ckps_dir: str) -> str:
    """
    在 root_ckps_dir 下递归寻找最新 checkpoint 目录。
    目录命名格式: checkpoint-<step>，按 step 最大优先；同 step 按修改时间最新优先。
    找不到则返回空字符串。
    """
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
# 数据集
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".png"}


class RemoteSensingDataset(Dataset):
    """读取本地文件夹中的图片，Resize -> ToTensor -> Normalize 到 [-1, 1]。"""

    def __init__(self, root_dir: str, image_size: int = img_size, in_channels: int = 3):
        self.paths = sorted(
            p for p in Path(root_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在 {root_dir} 中未找到任何受支持的图片文件")

        color_mode = "RGB" if in_channels == 3 else "L"
        self.color_mode = color_mode
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * in_channels, [0.5] * in_channels),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert(self.color_mode)
        return self.transform(img)


# ---------------------------------------------------------------------------
# 蒸馏特征提取接口
# ---------------------------------------------------------------------------
def extract_features(
    model: UNet2DModel,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    layer_name: str = "mid_block",
) -> torch.Tensor:
    """
    使用 register_forward_hook 捕获 UNet 指定中间层的特征图。

    参数:
        model      : UNet2DModel 实例
        images     : 输入图像张量 (B, C, H, W)，已加噪
        timesteps  : 时间步张量 (B,)
        layer_name : 要捕获的层名称，如 "mid_block", "down_blocks.1", "up_blocks.0"

    返回:
        捕获到的特征张量
    """
    features: dict = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            features["feat"] = output[0]
        else:
            features["feat"] = output

    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        available = [n for n, _ in model.named_modules() if n]
        raise ValueError(
            f"未找到层 '{layer_name}'。可用层: {available}"
        )

    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(images, timesteps)
    finally:
        handle.remove()

    return features["feat"]


# ---------------------------------------------------------------------------
# 训练主函数
# ---------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    run_ckps_dir = os.path.join(args.ckps_dir, f'distil256_{timestamp}')
    run_log_dir = os.path.join(args.log_dir, f'distil256_{timestamp}')

    if args.only_sample:
        sample_images_from_checkpoint(args, device)
        return

    os.makedirs(run_ckps_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    tb_logger = SummaryWriter(log_dir=run_log_dir)

    # ---- 模型 ----
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=args.in_channels,
        out_channels=args.in_channels,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D",
                          "DownBlock2D",
                          "AttnDownBlock2D",
                          "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D",
                        "AttnUpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D"),
        layers_per_block=3,
        attention_head_dim=8,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[模型] UNet2DModel 参数量: {num_params:.2f} M")

    # ---- 训练噪声调度器 (DDPMScheduler) ----
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    # ---- 数据 ----
    dataset = RemoteSensingDataset(args.data_dir, image_size=img_size, in_channels=args.in_channels)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=18,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[数据] 共 {len(dataset)} 张图片, 每批 {args.batch_size} 张")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- Cosine 学习率调度 ----
    total_steps = args.epochs * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    # ---- 混合精度 ----
    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ---- 可选: 从 checkpoint 继续训练 ----
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        print(f"[恢复] 尝试从 checkpoint 加载: {args.resume_checkpoint}")
        pipeline = DDPMPipeline.from_pretrained(args.resume_checkpoint)
        model.load_state_dict(pipeline.unet.state_dict(), strict=True)
        print("[恢复] 模型权重加载完成")

        state_path = os.path.join(args.resume_checkpoint, "training_state.pt")
        if os.path.isfile(state_path):
            train_state = torch.load(state_path, map_location=device)
            if "optimizer" in train_state:
                optimizer.load_state_dict(train_state["optimizer"])
            if "scaler" in train_state and isinstance(train_state["scaler"], dict):
                scaler.load_state_dict(train_state["scaler"])
            if "scheduler" in train_state and isinstance(train_state["scheduler"], dict):
                lr_scheduler.load_state_dict(train_state["scheduler"])
            start_epoch = int(train_state.get("epoch", 0))
            global_step = int(train_state.get("global_step", 0))
            print(f"[恢复] 训练状态加载完成, start_epoch={start_epoch}, global_step={global_step}")
        else:
            print("[恢复] 未找到 training_state.pt，仅恢复模型参数，epoch/step 从 0 开始")

    # ---- 训练循环 ----
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            clean_images = batch.to(device)
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timestep = torch.randint(
                0, args.num_train_timesteps, (bs,), device=device, dtype=torch.long
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timestep)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                pred_noise = model(noisy_images, timestep, return_dict=False)[0]
                loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]
            tb_logger.add_scalar("lr", current_lr, global_step)

            if global_step % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Step {global_step} | Loss: {loss.item():.5f}"
                )
            tb_logger.add_scalar('step_loss', loss.item(), global_step)
            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_dir = os.path.join(run_ckps_dir, f"checkpoint-{global_step}")
                save_pipeline(model, noise_scheduler, ckpt_dir)
                save_training_state(
                    optimizer=optimizer,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    save_dir=ckpt_dir,
                )
                print(f"  >> Checkpoint 已保存至 {ckpt_dir}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch+1}/{args.epochs} 完成, 平均 Loss: {avg_loss:.5f}")
        tb_logger.add_scalar('epoch_loss', avg_loss, epoch + 1)

    # ---- 保存最终模型 ----
    final_dir = os.path.join(run_ckps_dir, "final")
    save_pipeline(model, noise_scheduler, final_dir)
    print(f"训练完成! 最终模型已保存至 {final_dir}")

    # ---- 蒸馏接口演示 ----
    demo_extract_features(model, device, args)


def save_pipeline(model, noise_scheduler, save_dir):
    """保存为 DDPMPipeline（同时附带 DPMSolver 采样器配置）。"""
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.save_pretrained(save_dir)

    dpm_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    dpm_scheduler.save_pretrained(os.path.join(save_dir, "scheduler_dpm"))


def save_training_state(optimizer, scaler, lr_scheduler, epoch: int, global_step: int, save_dir: str):
    """保存继续训练所需状态。"""
    state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, os.path.join(save_dir, "training_state.pt"))


def sample_images_from_checkpoint(args, device):
    """仅采样模式：加载 checkpoint 并生成 100 张图片。"""
    if not args.resume_checkpoint:
        raise ValueError("only_sample=True 时需要可用的 resume_checkpoint 路径")
    if not os.path.isdir(args.resume_checkpoint):
        raise FileNotFoundError(f"checkpoint 路径不存在: {args.resume_checkpoint}")

    print(f"[采样] 从 checkpoint 加载: {args.resume_checkpoint}")
    pipeline = DDPMPipeline.from_pretrained(args.resume_checkpoint)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    if args.use_dpm_solver:
        scheduler_dpm_path = os.path.join(args.resume_checkpoint, "scheduler_dpm")
        if os.path.isdir(scheduler_dpm_path):
            pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(scheduler_dpm_path)
        else:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        print("[采样] 使用 DPMSolverMultistepScheduler 加速采样")
    else:
        print("[采样] 使用原始 DDPM 采样")

    if device.type == "cuda":
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    else:
        generator = torch.Generator().manual_seed(args.seed)

    images = pipeline(batch_size=100, generator=generator).images

    image_dir = os.path.join(args.log_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_path = os.path.join(image_dir, f"sample_{i+1}.png")
        img.save(save_path)
        print(f"[采样] 已保存: {save_path}")


def demo_extract_features(model, device, args):
    """演示蒸馏特征提取接口。"""
    print("\n" + "=" * 60)
    print("蒸馏特征提取接口演示")
    print("=" * 60)

    model.eval()
    dummy_images = torch.randn(2, args.in_channels, img_size, img_size, device=device)
    dummy_timesteps = torch.tensor([100, 500], device=device, dtype=torch.long)

    for layer in ["mid_block", "down_blocks.1", "up_blocks.0"]:
        try:
            feat = extract_features(model, dummy_images, dummy_timesteps, layer_name=layer)
            print(f"  层 '{layer}' -> 特征形状: {feat.shape}")
        except ValueError as e:
            print(f"  层 '{layer}' -> 跳过: {e}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=f"训练 {img_size}x{img_size} 轻量级 DDPM 教师模型")
    default_resume_checkpoint = get_latest_checkpoint_path(ckps_dir)

    parser.add_argument("--data_dir", type=str, default=data_dir, help=f"图片数据集文件夹路径 （默认{data_dir}）")
    parser.add_argument("--in_channels", type=int, default=in_channels, help=f"图像通道数 (默认: {in_channels})")
    parser.add_argument("--batch_size", type=int, default=batch_size, help=f"批大小 (默认: {batch_size})")
    parser.add_argument("--lr", type=float, default=lr, help=f"学习率 (默认: {lr})")
    parser.add_argument("--min_lr", type=float, default=min_lr, help="Cosine 调度最小学习率 eta_min (默认: 0)")
    parser.add_argument("--epochs", type=int, default=epochs, help=f"训练轮数 (默认: {epochs})")
    parser.add_argument("--save_every", type=int, default=save_every, help=f"每 N 步保存 checkpoint (默认: {save_every})")
    parser.add_argument("--ckps_dir", type=str, default=ckps_dir, help=f"日志目录(默认: {ckps_dir})")
    parser.add_argument("--log_dir", type=str, default=log_dir, help=f"输出目录(默认: {log_dir})")
    parser.add_argument("--fp16", action="store_true", help=f"启用 fp16 混合精度训练")
    parser.add_argument("--num_train_timesteps", type=int, default=num_train_timesteps,
                        help=f"噪声调度步数 (默认: {num_train_timesteps})")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--use_dpm_solver", action="store_true", default=use_dpm_solver,
                        help="采样时使用 DPMSolver 加速 (默认: True)")
    parser.add_argument("--no_dpm_solver", dest="use_dpm_solver", action="store_false",
                        help="采样时禁用 DPMSolver，使用原始 DDPM 采样")
    parser.add_argument("--only_sample", action="store_true", help="仅采样模式：加载 checkpoint 生成 100 张图片，不进行训练")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=default_resume_checkpoint,
        help=(
            "从指定 checkpoint 路径恢复训练/采样（目录需包含模型权重，可选 training_state.pt）。"
            f"默认自动选择 {ckps_dir} 下最新 checkpoint"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
