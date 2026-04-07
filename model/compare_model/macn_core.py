"""
Ke Li et al. 2023 MACN（Mixing Self-Attention and Convolution Network）PyTorch 实现。

官方仓库: https://github.com/like413/MACN
（trento/network.py 中 MixConvNet / MACT / MCGF）

HSI 输入为 (B, 1, C, H, W)，LiDAR 为 (B, Cl, H, W)；首段 Conv2d 通道数按谱维动态设为 8*(C-2)，
以适配 Houston 等与 Trento(PCA=30) 不同的谱段数。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _position_map(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """原 position()，改为 device/dtype 安全。"""
    loc_w = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).unsqueeze(0).repeat(h, 1)
    loc_h = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).unsqueeze(1).repeat(1, w)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def _stride_hw(x: torch.Tensor, stride: int) -> torch.Tensor:
    return x[:, :, ::stride, ::stride]


def _init_rate_half(tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        tensor.data.fill_(0.5)


class MACT(nn.Module):
    """Mixing self-attention and convolution Transformer 层（与官方一致）。"""

    def __init__(
        self,
        in_planes: int = 64,
        out_planes: int = 64,
        kernel_att: int = 7,
        head: int = 8,
        kernel_conv: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = nn.Parameter(torch.Tensor(1))
        self.rate2 = nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(
            self.kernel_conv * self.kernel_conv * self.head_dim,
            out_planes,
            kernel_size=self.kernel_conv,
            bias=True,
            groups=self.head_dim,
            padding=1,
            stride=stride,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_rate_half(self.rate1)
        _init_rate_half(self.rate2)
        # 官方仓库对 dep_conv 的 repeat 与分组卷积权形状不完全一致；此处用 Kaiming 初始化保持可训练
        nn.init.kaiming_uniform_(self.dep_conv.weight, a=math.sqrt(5))
        if self.dep_conv.bias is not None:
            nn.init.zeros_(self.dep_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        pe = self.conv_p(_position_map(h, w, x.device, x.dtype))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = _stride_hw(q_att, self.stride)
            q_pe = _stride_hw(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(
            b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out
        )
        unfold_rpe = self.unfold(self.pad_att(pe)).view(
            1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out
        )

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(
            b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out
        )
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(
            torch.cat(
                [
                    q.view(b, self.head, self.head_dim, h * w),
                    k.view(b, self.head, self.head_dim, h * w),
                    v.view(b, self.head, self.head_dim, h * w),
                ],
                1,
            )
        )
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        out_conv = self.dep_conv(f_conv)
        return self.rate1 * out_att + self.rate2 * out_conv


class MCGF(nn.Module):
    """Multisource cross-guided fusion（无 einops，等价于官方）。"""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, kv_include_self: bool = False) -> torch.Tensor:
        b, n1, _ = x1.shape
        n2 = x2.size(1)
        h = self.heads
        d = self.to_q.out_features // h

        q = self.to_q(x1).view(b, n1, h, d).transpose(1, 2)
        k = self.to_k(x2).view(b, n2, h, d).transpose(1, 2)
        v = self.to_v(x2).view(b, n2, h, d).transpose(1, 2)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n1, -1)
        out = self.to_out(out)
        out = x1 + out

        f_q = self.to_q(out).view(b, n1, h, d).transpose(1, 2)
        f_k = self.to_k(out).view(b, n1, h, d).transpose(1, 2)
        f_v = self.to_v(x1).view(b, n1, h, d).transpose(1, 2)

        dots = torch.einsum('b h i d, b h j d -> b h i j', f_q, f_k) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, f_v)
        out = out.transpose(1, 2).contiguous().view(b, n1, -1)
        return self.to_out(out)


class MACNBackbone(nn.Module):
    """MixConvNet：输出分类 logits。"""

    def __init__(
        self,
        hsi_channels: int,
        lidar_channels: int,
        num_classes: int,
        num_tokens: int = 4,
        dim: int = 64,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hsi_channels < 3:
            raise ValueError('MACN Conv3d 需要 hsi_channels >= 3')

        self.L = num_tokens
        self.cT = dim
        self._hsi_c = hsi_channels
        self._lidar_c = lidar_channels

        d_out = hsi_channels - 2
        merge_ch = 8 * d_out

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(merge_ch, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(lidar_channels, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim))
        nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT))
        nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.cross = MCGF(dim)
        self.mact = MACT()
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x_hsi: torch.Tensor, x_lidar: torch.Tensor) -> torch.Tensor:
        # x_hsi: (B, C, H, W) -> (B, 1, C, H, W)
        x1 = x_hsi.unsqueeze(1)
        x1 = self.conv3d_features(x1)
        # (B, 8, D', H', W') -> (B, 8*D', H', W') 与官方 einops 一致
        b, c, d, h, w = x1.shape
        x1 = x1.reshape(b, c * d, h, w)
        x1 = self.conv2d_features(x1)
        x1 = self.mact(x1)
        x1 = x1.flatten(2).transpose(1, 2)

        x2 = self.conv2d_features2(x_lidar)
        x2 = self.mact(x2)
        x2 = x2.flatten(2).transpose(1, 2)

        b = x1.size(0)
        wa = self.token_wA.transpose(1, 2)
        a1 = torch.matmul(x1, wa.expand(b, -1, -1))
        a1 = a1.transpose(1, 2).softmax(dim=-1)
        vv1 = torch.matmul(x1, self.token_wV.expand(b, -1, -1))
        t1 = torch.matmul(a1, vv1)

        a2 = torch.matmul(x2, wa.expand(b, -1, -1))
        a2 = a2.transpose(1, 2).softmax(dim=-1)
        vv2 = torch.matmul(x2, self.token_wV.expand(b, -1, -1))
        t2 = torch.matmul(a2, vv2)

        cls1 = self.cls_token.expand(b, -1, -1)
        seq1 = torch.cat((cls1, t1), dim=1) + self.pos_embedding
        seq1 = self.dropout(seq1)

        cls2 = self.cls_token.expand(b, -1, -1)
        seq2 = torch.cat((cls2, t2), dim=1) + self.pos_embedding
        seq2 = self.dropout(seq2)

        x_1 = self.cross(seq1, seq2)
        x_2 = self.cross(seq2, seq1)

        o1 = self.mlp_head(x_1[:, 0])
        o2 = self.mlp_head(x_2[:, 0])
        return o1 + o2
