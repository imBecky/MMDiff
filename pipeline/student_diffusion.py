"""
学生扩散封装：与 ../GFDiff/pipeline/student_diffusion.py 对齐。
学生骨干为 diffusers UNet2DModel，由 ../GFDiff/train_distill.py 蒸馏得到并保存为 DDPMPipeline。
"""
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler

from param import (
    DIFFUSION_NOISE_MODE,
    DIFFUSION_NORMALIZE_INPUT,
    RANDOM_SEED,
    STUDENT_NUM_TRAIN_TIMESTEPS,
)
from utils.unet_hw import unet_sample_hw


def normalize_student_checkpoint_dir(checkpoint_path) -> str:
    """去掉路径首尾空白并转为绝对路径，避免 diffusers 误判为 Hub repo id。"""
    s = str(checkpoint_path).strip()
    if not s:
        raise ValueError('学生模型 checkpoint 路径为空（param.STUDENT_CHECKPOINT）')
    p = Path(s).expanduser()
    abs_s = os.path.abspath(str(p)) if not p.is_absolute() else str(p.resolve())
    if not os.path.isdir(abs_s):
        raise FileNotFoundError(
            f'学生 DDPMPipeline 目录不存在或不是文件夹:\n  {abs_s}'
        )
    return abs_s


class StudentDiffusionWrapper:
    """Wraps a diffusers UNet2DModel student to provide the same API as the
    project's DDPM model (feed_data / get_feats / netG)."""

    def __init__(
        self,
        checkpoint_path,
        num_train_timesteps=STUDENT_NUM_TRAIN_TIMESTEPS,
        noise_mode=None,
        noise_seed_base=None,
        normalize_diffusion_input=None,
        feat_layers=None,
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_dir = normalize_student_checkpoint_dir(checkpoint_path)
        pipeline = DDPMPipeline.from_pretrained(ckpt_dir)
        self.netG = pipeline.unet.to(device)
        self.netG.eval()
        for p in self.netG.parameters():
            p.requires_grad_(False)
        # 与 checkpoint 内保存的调度一致，避免手写 DDPMScheduler 与训练时不一致
        if getattr(pipeline, 'scheduler', None) is not None:
            self.scheduler = pipeline.scheduler
        else:
            self.scheduler = DDPMScheduler.from_pretrained(ckpt_dir)
        self.device = device
        self.data = None
        self.noise_mode = noise_mode if noise_mode is not None else DIFFUSION_NOISE_MODE
        self.noise_seed_base = int(
            noise_seed_base if noise_seed_base is not None else RANDOM_SEED
        )
        self.normalize_diffusion_input = (
            normalize_diffusion_input
            if normalize_diffusion_input is not None
            else DIFFUSION_NORMALIZE_INPUT
        )
        # 与 train_distill 一致的 UNet 子模块名列表；空列表则退回 fe/fd 整数下标
        self.feat_layers = list(feat_layers) if feat_layers else []

    def rgb_spatial_size_for_unet(self) -> tuple[int, int]:
        """兼容旧代码；与 unet_sample_hw(self.netG) 相同。"""
        return unet_sample_hw(self.netG)

    def feed_data(self, data):
        self.data = {}
        for k, v in data.items():
            if k == 'sample_indices' and isinstance(v, torch.Tensor):
                self.data[k] = v.to(self.device)
            elif isinstance(v, torch.Tensor):
                self.data[k] = v.to(self.device)
            else:
                self.data[k] = v

    def set_new_noise_schedule(self, *args, **kwargs):
        pass

    def _normalize_rgb_for_diffusion(self, rgb: torch.Tensor) -> torch.Tensor:
        """与 train_distill.DualResDataset 一致：先约 [0,1] 再 (x-0.5)/0.5 -> [-1,1]。"""
        x = rgb
        if x.max() > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        return (x - 0.5) / 0.5

    def _sample_noise(self, x: torch.Tensor, t: int, training: bool) -> torch.Tensor:
        """按 DIFFUSION_NOISE_MODE 生成与输入同形状的噪声。"""
        B = x.shape[0]
        mode = (self.noise_mode or 'deterministic').lower()
        if mode == 'random':
            return torch.randn_like(x)
        if mode == 'eval_fixed':
            if training:
                return torch.randn_like(x)
            g = torch.Generator(device=x.device)
            g.manual_seed(self.noise_seed_base + int(t) * 9973)
            return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        # deterministic：按样本全局下标可复现（需 feed_data 提供 sample_indices）
        noise = torch.zeros_like(x)
        indices = self.data.get('sample_indices') if self.data else None
        for i in range(B):
            g = torch.Generator(device=x.device)
            if indices is not None:
                seed = self.noise_seed_base + int(t) * 100000 + int(indices[i].item())
            else:
                seed = self.noise_seed_base + int(t) * 100000 + i
            g.manual_seed(int(seed % (2 ** 31)))
            noise[i] = torch.randn((1,) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype, generator=g)
        return noise

    def get_feats(self, t, training: bool = True):
        """提取扩散中间特征。training=True 时保留对输入 rgb 的 autograd 图，使冻结 UNet 仍能向投影层回传梯度。"""
        self.netG.eval()
        # 仅推理时关闭梯度以省显存；训练时 UNet 参数已 requires_grad=False，不更新骨干，但需对输入建图。
        cm = torch.enable_grad if training else torch.no_grad
        with cm():
            rgb = self.data['rgb']
            if self.normalize_diffusion_input:
                x = self._normalize_rgb_for_diffusion(rgb)
            else:
                x = rgb
            th, tw = unet_sample_hw(self.netG)
            if x.shape[-2] != th or x.shape[-1] != tw:
                x = F.interpolate(
                    x,
                    size=(th, tw),
                    mode='bilinear',
                    align_corners=False,
                )
            B = x.shape[0]
            timestep = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise = self._sample_noise(x, int(t), training)
            noisy = self.scheduler.add_noise(x, noise, timestep)
            if self.feat_layers:
                from model.diffusion_features import MultiLayerFeatureExtractor

                extractor = MultiLayerFeatureExtractor(self.netG, self.feat_layers)
                self.netG(noisy, timestep)
                out = {k: extractor.features[k].clone() for k in self.feat_layers}
                extractor.remove()
                return out
            fe, fd = self._forward_with_feats(noisy, timestep)
            return fe, fd

    def _forward_with_feats(self, sample, timestep):
        """Replicates UNet2DModel.forward but captures per-resnet decoder features."""
        model = self.netG

        t_emb = model.time_proj(timestep)
        emb = model.time_embedding(t_emb)

        sample = model.conv_in(sample)

        # ---- encoder ----
        down_block_res_samples = (sample,)
        for downsample_block in model.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        fe = list(down_block_res_samples)

        # ---- mid ----
        if model.mid_block is not None:
            sample = model.mid_block(sample, emb)

        # ---- decoder (capture every resnet output) ----
        fd_raw = []
        for upsample_block in model.up_blocks:
            n_resnets = len(upsample_block.resnets)
            res_samples = list(down_block_res_samples[-n_resnets:])
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            has_attn = (hasattr(upsample_block, 'attentions')
                        and upsample_block.attentions is not None
                        and len(upsample_block.attentions) > 0)

            for j, resnet in enumerate(upsample_block.resnets):
                res_hidden = res_samples.pop()
                sample = torch.cat([sample, res_hidden], dim=1)
                sample = resnet(sample, emb)
                if has_attn:
                    sample = upsample_block.attentions[j](sample)
                fd_raw.append(sample)

            if upsample_block.upsamplers is not None:
                for upsampler in upsample_block.upsamplers:
                    sample = upsampler(sample)

        fd = list(reversed(fd_raw))
        return fe, fd
