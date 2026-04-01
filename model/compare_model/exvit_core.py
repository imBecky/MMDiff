"""
ExViT / MViT 主干（与 jingyao16/ExViT 的 MViT_pytorch_upload.MViT 对齐）。

论文: Yao et al., TGRS 2023 — Extended Vision Transformer (ExViT).
代码参考: https://github.com/jingyao16/ExViT

双模态：HSI + LiDAR（对应原 Houston2013 的 HSI + DSM），沿通道拆成 x1/x2 两支深度可分离 CNN，
再各自 ViT 编码、拼接后融合 ViT，最后用 softmax 权重对 token 加权池化 + 分类头。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """b c h w -> b (h w) c"""
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.reshape(b, n, h, -1).permute(0, 2, 1, 3).contiguous()
            for t in qkv
        ]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(b, n, -1)
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        mode: str,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    _Residual(_PreNorm(dim, _Attention(dim, heads, dim_head, dropout))),
                    _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim, dropout))),
                ])
            )
        self.mode = mode

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == 'MViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
            return x
        raise ValueError(f'未知 mode {self.mode!r}')


class MViTBackbone(nn.Module):
    """原 MViT.forward(x1, x2) -> logits；x1=HSI，x2=LiDAR/DSM。"""

    def __init__(
        self,
        patch_size: int,
        num_patches: tuple[int, int],
        num_classes: int,
        dim: int = 64,
        depth: int = 6,
        heads: int = 4,
        mlp_dim: int = 32,
        dim_head: int = 16,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        mode: str = 'MViT',
    ):
        super().__init__()
        if depth < 4:
            raise ValueError(f'MViT 原实现假定 depth>=4（fusion depth=depth-4），当前 depth={depth}')

        nout = 16
        samesize = 1
        hsi_ch, lidar_ch = int(num_patches[0]), int(num_patches[1])

        self.separable1 = nn.Sequential(
            nn.Conv2d(hsi_ch, hsi_ch, kernel_size=3, padding=samesize, groups=hsi_ch),
            nn.Conv2d(hsi_ch, nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
            nn.Conv2d(nout * 2, nout * 2, kernel_size=3, padding=samesize, groups=nout * 2),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU(),
        )
        self.separable2 = nn.Sequential(
            nn.Conv2d(lidar_ch, lidar_ch, kernel_size=3, padding=samesize, groups=lidar_ch),
            nn.Conv2d(lidar_ch, nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout * 2, kernel_size=1),
            nn.BatchNorm2d(nout * 2),
            nn.GELU(),
            nn.Conv2d(nout * 2, nout * 2, kernel_size=3, padding=samesize, groups=nout * 2),
            nn.Conv2d(nout * 2, nout * 4, kernel_size=1),
            nn.BatchNorm2d(nout * 4),
            nn.GELU(),
        )

        grid_size = 1
        self._vit_patches = (patch_size // grid_size) ** 2
        self.to_patch_embedding2 = nn.Linear(nout * 4, dim)
        self.to_patch_embedding2c = nn.Linear(nout * 4, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self._vit_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = _Transformer(
            dim, depth - 4, heads, dim_head, mlp_dim, dropout, mode,
        )
        self.transformer1 = _Transformer(
            dim, depth - 2, heads, dim_head, mlp_dim, dropout, mode,
        )
        self.transformer2 = _Transformer(
            dim, depth - 2, heads, dim_head, mlp_dim, dropout, mode,
        )

        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1),
        )
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.separable1(x1)
        x1 = _bchw_to_bhwc(x1)
        x1 = self.to_patch_embedding2(x1)
        b, n, _ = x1.shape
        x1 = x1 + self.pos_embedding[:, :n]
        x1 = self.dropout(x1)

        x2 = self.separable2(x2)
        x2 = _bchw_to_bhwc(x2)
        x2 = self.to_patch_embedding2c(x2)
        x2 = x2 + self.pos_embedding[:, :n]
        x2 = self.dropout(x2)

        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.transformer(x)

        xs = torch.squeeze(self.mlp_head0(x), dim=-1)
        x = torch.einsum('bn,bnd->bd', xs, x)
        return self.mlp_head1(x)
