"""Spatial fusion: Transformer decoder layers with additive cross-attention logit bias.

Bias applies to center-query rows only (caller provides full Lq x L_mem bias template).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionWithLogitBias(nn.Module):
    """Multi-head cross-attention; additive logit_bias broadcast over heads."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query: B,Lq,E; key,value: B,Lm,E;
        logit_bias: (1,1,Lq,Lm) or (B,1,Lq,Lm) added to pre-softmax scores for all heads.
        """
        b, lq, _ = query.shape
        _, lm, _ = key.shape

        q = self.q_proj(query).view(b, lq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, lm, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, lm, self.nhead, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale  # B,H,Lq,Lm
        if logit_bias is not None:
            scores = scores + logit_bias
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, lq, self.d_model)
        out = self.out_proj(out)
        if need_weights:
            return out, attn
        return out, None


class SpatialFusionDecoderLayer(nn.Module):
    """norm_first stack: self-attn -> biased cross-attn -> FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = CrossAttentionWithLogitBias(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = {
            "relu": F.relu,
            "gelu": F.gelu,
        }[activation.lower()]

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cross_logit_bias: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = tgt
        x = x + self._sa_block(self.norm1(x))
        y, attn = self._ca_block(
            self.norm2(x), memory, cross_logit_bias, need_attn_weights
        )
        x = x + y
        x = x + self._ff_block(self.norm3(x))
        return x, attn

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x, need_weights=False)
        return self.dropout1(out)

    def _ca_block(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cross_logit_bias: Optional[torch.Tensor],
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, attn = self.cross_attn(
            x, memory, memory, logit_bias=cross_logit_bias, need_weights=need_weights
        )
        return self.dropout2(out), attn

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SpatialFusionDecoder(nn.Module):
    def __init__(self, layers: List[SpatialFusionDecoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cross_logit_bias: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        out = tgt
        attns: List[Optional[torch.Tensor]] = []
        for layer in self.layers:
            out, attn = layer(
                out,
                memory,
                cross_logit_bias=cross_logit_bias,
                need_attn_weights=need_attn_weights,
            )
            attns.append(attn)
        return out, attns


def build_spatial_fusion_decoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    activation: str = "gelu",
) -> SpatialFusionDecoder:
    layers = [
        SpatialFusionDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation=activation
        )
        for _ in range(num_layers)
    ]
    return SpatialFusionDecoder(layers)
