#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练目录加载 MultimodalClassifier，抓取最后一层（可选：所有层平均）decoder
cross-attention 权重，绘制：
  A) global vs center 双 query（各 qk 子 token 先平均）在 memory token 上的热力图；
  B) multi_token 时：**论文优先**——三组 HSI memory token 的 Global vs Center 分组柱状图；另可选保存
   将三路权重视觉映射到 11×11 的示意位置（非密集空间分辨率）。

用法（仓库根目录，与 main.py 一致）：
  python utils/plot_dual_query_attention.py \\
      --ckpt-dir "../../autodl-tmp/classifier/0505-1929_hp_s42_loss_g07_c03/final"

注意：必须先与训练时一致的 MMDIFF_*，或使用 --modality-combo auto（默认）根据
classifier.pt 中 pos_embed_mem 列数自动设置 MMDIFF_MODALITY_COMBO（在首次 import param 之前）。
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# 必须在 import torch 之前：空串或非法 OMP_NUM_THREADS 会触发 libgomp 报错
_omp = (os.environ.get("OMP_NUM_THREADS") or "").strip()
try:
    if _omp == "" or int(_omp) < 1:
        os.environ["OMP_NUM_THREADS"] = "4"
except ValueError:
    os.environ["OMP_NUM_THREADS"] = "4"

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

# 保证仓库根在 path（用户可能从其他 cwd 调用）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    pass


class AttnGrabber:
    """临时替换 CrossAttentionWithLogitBias.forward，强制返回注意力权重。"""

    def __init__(self, cross: nn.Module):
        self.cross = cross
        self._orig = cross.forward
        self.last: Optional[torch.Tensor] = None
        cross.forward = self._wrapped  # type: ignore[assignment]

    def _wrapped(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        out = self._orig(*args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            w = out[1]
            if w is not None:
                self.last = w.detach().float().cpu()
        return out

    def restore(self) -> None:
        self.cross.forward = self._orig  # type: ignore[assignment]


def _resolve_ckpt_path(ckpt_dir: Path) -> Path:
    p = ckpt_dir.expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    f = p / "classifier.pt"
    if not f.is_file():
        raise FileNotFoundError(f"找不到权重文件: {f}")
    return f


def _extract_state_dict(blob) -> dict:
    if isinstance(blob, dict):
        for k in ("state_dict", "model", "module"):
            if k in blob and isinstance(blob[k], dict):
                return blob[k]
        if blob and any("." in str(x) for x in blob.keys()):
            return blob
        raise ValueError(f"无法从 checkpoint 解析 state_dict，键样例: {list(blob.keys())[:8]}")
    raise ValueError(f"不支持的 checkpoint 类型: {type(blob)}")


def peek_pos_embed_mem_len(ckpt_file: Path) -> Optional[int]:
    """只读 checkpoint 中 pos_embed_mem 的 memory 序列长度。"""
    blob = torch.load(ckpt_file, map_location="cpu")
    state = _extract_state_dict(blob)
    t = state.get("pos_embed_mem")
    if t is None:
        return None
    if hasattr(t, "shape"):
        return int(t.shape[1])
    return None


def _apply_modality_env_for_mem_len(mem_len: int) -> str:
    """
    根据 pos_embed_mem 列数反推 MMDIFF_MODALITY_COMBO。
    含：空间融合 121×K；旧版少量 token。
    """
    m = {
        363: "hsi+rgb+lidar",
        242: "hsi+lidar",
        121: "hsi",
        5: "hsi+rgb+lidar",
        4: "hsi+lidar",
        3: "hsi",
        2: "hsi+lidar",
        1: "hsi",
    }
    if mem_len not in m:
        raise ValueError(
            f"无法为 pos_embed_mem 列数 mem_len={mem_len} 自动设置模态组合；"
            f"请显式设置环境变量或 --modality-combo"
        )
    combo = m[mem_len]
    os.environ["MMDIFF_MODALITY_COMBO"] = combo
    return combo


def ensure_env_matches_checkpoint(
    ckpt_file: Path,
    modality_combo: str,
    hsi_agg_mode: str,
) -> None:
    """
    在 import param 前设置环境变量，使 build_opt() 得到的模型与 checkpoint 一致。
    modality_combo: 'auto' | 'hsi' | 'hsi+lidar' | ...
    hsi_agg_mode: '' 表示不设；否则写入 MMDIFF_HSI_AGG_MODE

    注意：--modality-combo auto 会**始终**根据 classifier.pt 里 pos_embed_mem 列数覆盖
    MMDIFF_MODALITY_COMBO，避免 shell/run 脚本里已 export 的默认组合（如 hsi+rgb+lidar）
    与旧 checkpoint 不一致却仍跳过推断。
    """
    if hsi_agg_mode.strip():
        os.environ["MMDIFF_HSI_AGG_MODE"] = hsi_agg_mode.strip()

    prev_combo = (os.environ.get("MMDIFF_MODALITY_COMBO") or "").strip()

    if modality_combo.strip().lower() != "auto":
        os.environ["MMDIFF_MODALITY_COMBO"] = modality_combo.strip()
        print(f"[env] MMDIFF_MODALITY_COMBO={os.environ['MMDIFF_MODALITY_COMBO']} (CLI)")
        return

    mem_len = peek_pos_embed_mem_len(ckpt_file)
    if mem_len is None:
        raise RuntimeError("checkpoint 中无 pos_embed_mem，无法用 auto；请指定 --modality-combo")
    combo = _apply_modality_env_for_mem_len(mem_len)
    if prev_combo and prev_combo != combo:
        print(
            f"[auto] 覆盖环境变量 MMDIFF_MODALITY_COMBO: {prev_combo!r} -> {combo!r} "
            f"(与 ckpt pos_embed_mem 列数 mem_len={mem_len} 对齐)"
        )
    else:
        print(
            f"[auto] pos_embed_mem columns={mem_len} -> MMDIFF_MODALITY_COMBO={combo} "
            f"(若训练组合并非上表常见映射，请显式传 --modality-combo / --hsi-agg-mode)"
        )


def reload_param_module():
    """param 在 import 时会锁死 opt；调整环境后必须 reload。"""
    importlib.invalidate_caches()
    if "param" in sys.modules:
        del sys.modules["param"]


def _load_state_dict_into_model(model: nn.Module, ckpt_file: Path, map_location) -> None:
    blob = torch.load(ckpt_file, map_location=map_location)
    state = _extract_state_dict(blob)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("WARN: strict load 失败，尝试过滤形状不匹配的键后加载。")
        model_sd = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k not in model_sd:
                skipped.append((k, "unexpected key in ckpt"))
                continue
            if hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
                if v.shape != model_sd[k].shape:
                    skipped.append((k, f"shape ckpt={tuple(v.shape)} model={tuple(model_sd[k].shape)}"))
                    continue
            filtered[k] = v
        if not filtered:
            raise RuntimeError("过滤后无可加载参数，请检查 MMDIFF_* 与训练一致") from e
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print("  missing_keys (first 15):", missing[:15])
        if unexpected:
            print("  unexpected_keys (first 15):", unexpected[:15])
        if skipped:
            print("  skipped_incompatible (first 10):")
            for item in skipped[:10]:
                print("   ", item)


def _memory_column_labels(model: torch.nn.Module) -> List[str]:
    """与 MultimodalClassifier._forward_tokens 拼接顺序一致。"""
    m = model
    names: List[str] = []
    if getattr(m, "use_hsi", False):
        mode = getattr(m.hsi_encoder, "agg_mode", "mean")
        if mode == "multi_token":
            names.extend(["HSI-center", "HSI-corner", "HSI-edge"])
        else:
            names.append("HSI")
    if getattr(m, "use_rgb", False):
        names.append("RGB")
    if getattr(m, "use_lidar", False):
        names.append("LiDAR")
    return names


def _hsi_mem_slice(model) -> Optional[slice]:
    if not getattr(model, "use_hsi", False):
        return None
    n = int(getattr(model.hsi_encoder, "n_output_tokens", 1))
    return slice(0, n)


def _is_spatial_fusion_model(model: torch.nn.Module) -> bool:
    ml = int(getattr(model, "mem_len", 0))
    return ml >= 121 and ml % 121 == 0


def _spatial_modality_slices(model: torch.nn.Module) -> List[Tuple[str, slice]]:
    n = 121
    parts: List[Tuple[str, slice]] = []
    off = 0
    if getattr(model, "use_hsi", False):
        parts.append(("HSI", slice(off, off + n)))
        off += n
    if getattr(model, "use_rgb", False):
        parts.append(("RGB", slice(off, off + n)))
        off += n
    if getattr(model, "use_lidar", False):
        parts.append(("LiDAR", slice(off, off + n)))
        off += n
    return parts


def _plot_spatial_modalities_grids(
    mean_g: np.ndarray,
    mean_c: np.ndarray,
    model: torch.nn.Module,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """每模态一个 11×11：左 Global 右 Center。"""
    import matplotlib.pyplot as plt

    parts = _spatial_modality_slices(model)
    nmod = len(parts)
    fig, axes = plt.subplots(nmod, 2, figsize=(7.5, 2.8 * max(nmod, 1)))
    if nmod == 1:
        axes = np.array([axes])
    for mi, (name, sl) in enumerate(parts):
        gg = mean_g[sl].reshape(11, 11)
        gc = mean_c[sl].reshape(11, 11)
        axes[mi, 0].imshow(gg, cmap="viridis", interpolation="nearest")
        axes[mi, 0].set_title(f"{name}: Global → 11×11")
        axes[mi, 0].set_xticks(np.arange(11))
        axes[mi, 0].set_yticks(np.arange(11))
        im = axes[mi, 1].imshow(gc, cmap="viridis", interpolation="nearest")
        axes[mi, 1].set_title(f"{name}: Center → 11×11")
        axes[mi, 1].set_xticks(np.arange(11))
        axes[mi, 1].set_yticks(np.arange(11))
        plt.colorbar(im, ax=axes[mi, 1], fraction=0.046)
    fig.suptitle(
        "Cross-attn over spatial memory tokens (per modality 121 cells, row-major 11×11)"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _aggregate_across_layers(
    weights_per_layer: Sequence[torch.Tensor],
) -> torch.Tensor:
    """各层 (B,H,Lq,Sm) 对层维平均。"""
    if not weights_per_layer:
        raise RuntimeError("无注意力权重")
    stacked = torch.stack([w.float() for w in weights_per_layer], dim=0)
    return stacked.mean(dim=0)


def _forward_and_capture(
    model,
    data_dict: dict,
    *,
    grabbers: Sequence[AttnGrabber],
    all_layers: bool,
) -> torch.Tensor:
    """返回最后一层或所有层平均的 cross-attn 权重 (B, H, Lq, Sm)，在 CPU float。"""

    for g in grabbers:
        g.last = None
    with torch.no_grad():
        _ = model(data_dict)

    if all_layers:
        mats = [g.last for g in grabbers if g.last is not None]
        if not mats:
            raise RuntimeError("未捕获到注意力权重（检查 PyTorch MultiheadAttention API）")
        w = _aggregate_across_layers(mats)
    else:
        w = grabbers[-1].last
        if w is None:
            raise RuntimeError("最后一层未捕获到注意力权重")
    return w


def _reduce_to_query_pair(
    w: torch.Tensor,
    qk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    w: (B, H, 2*qk, M)
    -> global_avg (B, M), center_avg (B, M)
    """
    if w.dim() != 4:
        raise ValueError(f"期望 4D 注意力权重，得到 shape={tuple(w.shape)}")
    b, h, lq, m = w.shape
    w2 = w.mean(dim=1)  # (B, 2*qk, M)
    if lq != 2 * qk:
        raise ValueError(f"query 长度 {lq} 与 2*qk={2 * qk} 不一致")
    g = w2[:, :qk, :].mean(dim=1)
    c = w2[:, qk:, :].mean(dim=1)
    return g, c


def _build_val_or_test_loader(split: str, seed: int):
    from param import BATCH_SIZE, USE_RGB_PATCHES, VAL_RATIO
    from pipeline.data import (
        build_test_loader,
        load_rgb_hr_meta,
        load_rgb_hr_volume,
        load_test_indices_shifted,
        load_train_bundle,
        split_train_val_indices,
    )

    feats_vol, rgb_vol, train_indices, label_shift = load_train_bundle()
    rgb_hr_vol = None
    hr_rh = 1
    hr_rw = 1
    if USE_RGB_PATCHES:
        rgb_hr_vol = load_rgb_hr_volume()
        meta = load_rgb_hr_meta()
        hr_rh = int(meta["rh"])
        hr_rw = int(meta["rw"])

    split_l = split.strip().lower()
    if split_l == "val":
        _, va_idx, _, va_pos = split_train_val_indices(train_indices, VAL_RATIO, seed)
        if va_idx is None or len(va_idx) == 0:
            raise RuntimeError("验证集为空（检查 VAL_RATIO）")
        loader = build_test_loader(
            feats_vol,
            rgb_vol,
            va_idx,
            BATCH_SIZE,
            global_row_indices=va_pos,
            rgb_strict_view=bool(USE_RGB_PATCHES),
            rgb_hr_vol=rgb_hr_vol,
            hr_rh=hr_rh,
            hr_rw=hr_rw,
        )
        return loader
    if split_l == "test":
        test_idx = load_test_indices_shifted(label_shift)
        gl = np.arange(len(test_idx), dtype=np.int64)
        loader = build_test_loader(
            feats_vol,
            rgb_vol,
            test_idx,
            BATCH_SIZE,
            global_row_indices=gl,
            rgb_strict_view=bool(USE_RGB_PATCHES),
            rgb_hr_vol=rgb_hr_vol,
            hr_rh=hr_rh,
            hr_rw=hr_rw,
        )
        return loader
    raise ValueError(f"split 须为 val|test，当前 {split!r}")


def _fill_patch_grid_from_hsi_tokens(
    weights_row: np.ndarray,
    hsi_slice: slice,
    agg_mode: str,
) -> np.ndarray:
    """
    weights_row: (M,) 单条 query 对 memory 的权重
    返回 (11, 11)，非 multi_token 时全 NaN。
    """
    grid = np.full((11, 11), np.nan, dtype=np.float64)
    if agg_mode != "multi_token":
        return grid
    # memory 中 HSI 三个 token 列序：center, corner, edge
    c_idx = hsi_slice.start + 0
    co_idx = hsi_slice.start + 1
    e_idx = hsi_slice.start + 2
    wc = float(weights_row[c_idx])
    wco = float(weights_row[co_idx])
    we = float(weights_row[e_idx])
    grid[5, 5] = wc
    for r, c in ((0, 0), (0, 10), (10, 0), (10, 10)):
        grid[r, c] = wco
    for r, c in ((0, 5), (5, 0), (5, 10), (10, 5)):
        grid[r, c] = we
    return grid


def _plot_modal_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    out_png: Path,
    out_pdf: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.2), 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(mat.max()), 1e-8))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Cross-attn weights (decoder): Global vs Center  ->  memory tokens")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _hsi_slice_indices(hsi_slice: slice) -> range:
    return range(hsi_slice.start, hsi_slice.stop)


def _paper_hsi_axis_labels(colnames: List[str], hsi_slice: slice) -> List[str]:
    """论文 x 轴：三路聚合 token 的直观短标签。"""
    mapping = {
        "HSI-center": "Center\nregion",
        "HSI-corner": "Corners\n(pooled)",
        "HSI-edge": "Edge mids\n(pooled)",
    }
    return [mapping.get(colnames[i], colnames[i]) for i in _hsi_slice_indices(hsi_slice)]


def _plot_hsi_multi_token_bars(
    mean_g: np.ndarray,
    mean_c: np.ndarray,
    colnames: List[str],
    hsi_slice: slice,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """
    论文主图：仅 3 个标量权重 → 分组柱状图比 11×11「复制填色」更直观。
    """
    import matplotlib.pyplot as plt

    idxs = list(_hsi_slice_indices(hsi_slice))
    labels = _paper_hsi_axis_labels(colnames, hsi_slice)
    v_g = [float(mean_g[i]) for i in idxs]
    v_c = [float(mean_c[i]) for i in idxs]
    x = np.arange(len(idxs))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - w / 2, v_g, w, label="Global query", color="#4c72b0", edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, v_c, w, label="Center query", color="#dd8452", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cross-attention weight (mean over samples)")
    ax.set_title(
        "Decoder cross-attention to HSI memory tokens (multi_token)\n"
        r"three pooled spectral–spatial aggregates, not $11{\times}11$ independent cells"
    )
    ax.legend(frameon=True, loc="upper right")
    ymax = max(v_g + v_c) * 1.18 if (v_g + v_c) else 1.0
    ax.set_ylim(0.0, max(ymax, 1e-6))
    for i, (a, b) in enumerate(zip(v_g, v_c)):
        ax.text(i - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
        ax.text(i + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_hsi_multi_token_delta(
    mean_g: np.ndarray,
    mean_c: np.ndarray,
    colnames: List[str],
    hsi_slice: slice,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Global − Center 三路差分，便于强调 query 差异。"""
    import matplotlib.pyplot as plt

    idxs = list(_hsi_slice_indices(hsi_slice))
    labels = _paper_hsi_axis_labels(colnames, hsi_slice)
    delta = np.array([float(mean_g[i] - mean_c[i]) for i in idxs])
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(idxs))
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.bar(x, delta, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$\Delta$ weight (Global $-$ Center)")
    ax.set_title("Difference of attention on the three HSI pooled tokens")
    for i, d in enumerate(delta):
        ax.text(i, d + (0.02 if d >= 0 else -0.02) * (abs(d) + 0.01), f"{d:+.4f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_patch_grids(
    g_grid: np.ndarray,
    c_grid: np.ndarray,
    out_png: Path,
    out_pdf: Path,
) -> None:
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin([np.nanmin(g_grid), np.nanmin(c_grid)]))
    vmax = float(np.nanmax([np.nanmax(g_grid), np.nanmax(c_grid)]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))

    for ax, grid, title in zip(
        axes.flat,
        [g_grid, c_grid],
        [
            "Global query (same 3 weights copied to symbolic 11×11 positions)",
            "Center query (same 3 weights copied to symbolic 11×11 positions)",
        ],
    ):
        mg = np.ma.masked_invalid(grid)
        im = ax.imshow(mg, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(np.arange(11))
        ax.set_yticks(np.arange(11))
        ax.set_title(
            title
            + "\n(gray = no extra HSI tokens; this is NOT dense spatial attention)"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Optional schematic: 11×11 layout only mirrors center / corners / edge-mids semantics;\n"
        "use attention_hsi_tokens_bar.* for the quantitative comparison in papers."
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-query decoder cross-attention heatmaps")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="../../autodl-tmp/classifier/0505-1929_hp_s42_loss_g07_c03/final",
        help="含 classifier.pt 的目录",
    )
    parser.add_argument(
        "--modality-combo",
        type=str,
        default="auto",
        help="auto：根据 ckpt 的 pos_embed_mem 列数覆盖 MMDIFF_MODALITY_COMBO（忽略 shell 里已 export 的旧值）；"
        "或显式 hsi|hsi+lidar|hsi+rgb+lidar",
    )
    parser.add_argument(
        "--hsi-agg-mode",
        type=str,
        default="",
        help="覆盖 MMDIFF_HSI_AGG_MODE（如 mean 对应三模态单 HSI token + rgb + lidar =3）",
    )
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"))
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None, help="划分 val 用；首次 import param 前无效")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录；默认 ./paper_figs/attention/<父目录名>/",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="对所有 decoder 层的 cross-attn 权重取平均（默认仅用最后一层）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    ckpt_file = _resolve_ckpt_path(Path(args.ckpt_dir))
    ckpt_parent = ckpt_file.parent.name

    # 在 import param 前设置环境与（可选）reload
    ensure_env_matches_checkpoint(ckpt_file, args.modality_combo, args.hsi_agg_mode)
    reload_param_module()

    import param  # noqa: WPS433 — 必须在环境变量之后

    seed = int(args.seed) if args.seed is not None else int(param.RANDOM_SEED)

    from main import create_classifier  # noqa: WPS433
    from param import BATCH_SIZE, RANDOM_SEED, USE_RGB_PATCHES, VAL_RATIO, opt  # noqa: WPS433

    out_dir = Path(args.out_dir).expanduser() if args.out_dir.strip() else None
    if out_dir is None:
        out_dir = _REPO_ROOT / "paper_figs" / "attention" / ckpt_parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = create_classifier(opt)
    _load_state_dict_into_model(model, ckpt_file, map_location=device)
    model = model.to(device)
    model.eval()

    qk = int(getattr(model, "query_tokens_per_query", 4))
    dec = model.decoder
    grabbers: List[AttnGrabber] = []
    try:
        for layer in dec.layers:
            g = AttnGrabber(layer.cross_attn)
            grabbers.append(g)

        from pipeline.data import batch_to_dict  # noqa: WPS433

        loader = _build_val_or_test_loader(args.split, seed)

        sum_g = None
        sum_c = None
        n_samples = 0

        for bi, batch in enumerate(loader):
            if bi >= int(args.num_batches):
                break
            dd, _labels = batch_to_dict(batch, device, USE_RGB_PATCHES, use_supcon=False)
            w = _forward_and_capture(
                model,
                dd,
                grabbers=grabbers,
                all_layers=bool(args.all_layers),
            )
            g_b, c_b = _reduce_to_query_pair(w, qk)
            bs = g_b.shape[0]
            sum_g = g_b.sum(dim=0) if sum_g is None else sum_g + g_b.sum(dim=0)
            sum_c = c_b.sum(dim=0) if sum_c is None else sum_c + c_b.sum(dim=0)
            n_samples += bs

        if n_samples == 0:
            raise RuntimeError("DataLoader 无 batch，检查数据路径与 split")

        mean_g = (sum_g / float(n_samples)).cpu().numpy()
        mean_c = (sum_c / float(n_samples)).cpu().numpy()
        mat = np.stack([mean_g, mean_c], axis=0)
        spatial_fusion = _is_spatial_fusion_model(model)

        if spatial_fusion:
            parts = _spatial_modality_slices(model)
            col_agg = [p[0] + "_mean121" for p in parts]
            agg_g = np.array([mean_g[sl].mean() for _, sl in parts])
            agg_c = np.array([mean_c[sl].mean() for _, sl in parts])
            mat_agg = np.stack([agg_g, agg_c], axis=0)
            png_a = out_dir / "attention_modalities_mean.png"
            pdf_a = out_dir / "attention_modalities_mean.pdf"
            _plot_modal_heatmap(mat_agg, ["Global", "Center"], col_agg, png_a, pdf_a)
            png_sp = out_dir / "attention_spatial_modalities.png"
            pdf_sp = out_dir / "attention_spatial_modalities.pdf"
            _plot_spatial_modalities_grids(mean_g, mean_c, model, png_sp, pdf_sp)

            summary_lines = [
                f"ckpt_file={ckpt_file}",
                f"MMDIFF_MODALITY_COMBO={os.environ.get('MMDIFF_MODALITY_COMBO', '')}",
                f"MMDIFF_HSI_AGG_MODE={os.environ.get('MMDIFF_HSI_AGG_MODE', param.HSI_AGG_MODE_CFG)}",
                f"center_distance_bias_alpha={getattr(model, 'center_distance_bias_alpha', '')}",
                f"split={args.split} seed={seed}",
                f"num_batches_cap={args.num_batches} samples_used={n_samples}",
                f"all_layers={bool(args.all_layers)}",
                f"mem_len={getattr(model, 'mem_len', None)} qk={qk}",
                f"spatial_fusion=121xmodalities token_dim={getattr(model, 'd_model', '')}",
                "",
                "mean attention per modality (121 spatial tokens averaged):",
                "  Global: " + " | ".join(f"{lab}:{v:.6f}" for lab, v in zip(col_agg, agg_g)),
                "  Center: " + " | ".join(f"{lab}:{v:.6f}" for lab, v in zip(col_agg, agg_c)),
                "",
                f"figures: {png_a}, {png_sp}",
                "",
            ]
        else:
            colnames = _memory_column_labels(model)
            if mat.shape[1] != len(colnames):
                raise RuntimeError(
                    f"列数 {mat.shape[1]} 与标签数 {len(colnames)} 不一致；"
                    f"mem_len={getattr(model, 'mem_len', '?')}"
                )

            row_labels = ["Global", "Center"]
            png_a = out_dir / "attention_modal_tokens.png"
            pdf_a = out_dir / "attention_modal_tokens.pdf"
            _plot_modal_heatmap(mat, row_labels, colnames, png_a, pdf_a)

            agg_mode = str(getattr(model.hsi_encoder, "agg_mode", ""))
            hsi_sl = _hsi_mem_slice(model)

            summary_lines = [
                f"ckpt_file={ckpt_file}",
                f"MMDIFF_MODALITY_COMBO={os.environ.get('MMDIFF_MODALITY_COMBO', '')}",
                f"MMDIFF_HSI_AGG_MODE={os.environ.get('MMDIFF_HSI_AGG_MODE', param.HSI_AGG_MODE_CFG)}",
                f"split={args.split} seed={seed}",
                f"num_batches_cap={args.num_batches} samples_used={n_samples}",
                f"all_layers={bool(args.all_layers)}",
                f"mem_len={getattr(model, 'mem_len', None)} qk={qk}",
                f"hsi_agg_mode={agg_mode}",
                f"memory_columns={colnames}",
                "",
                "mean Global weights:",
                "  " + " | ".join(f"{c}:{v:.6f}" for c, v in zip(colnames, mean_g)),
                "mean Center weights:",
                "  " + " | ".join(f"{c}:{v:.6f}" for c, v in zip(colnames, mean_c)),
                "",
            ]
            if "HSI-center" in colnames:
                i = colnames.index("HSI-center")
                dg = float(mean_g[i] - mean_c[i])
                summary_lines.append(f"diff Global-Center on HSI-center column: {dg:.6f}")

            if agg_mode == "multi_token" and hsi_sl is not None:
                g_grid = _fill_patch_grid_from_hsi_tokens(mean_g, hsi_sl, agg_mode)
                c_grid = _fill_patch_grid_from_hsi_tokens(mean_c, hsi_sl, agg_mode)
                png_b = out_dir / "attention_patch_grid.png"
                pdf_b = out_dir / "attention_patch_grid.pdf"
                _plot_patch_grids(g_grid, c_grid, png_b, pdf_b)
                summary_lines.append(f"patch_grid_saved={png_b}")
            else:
                summary_lines.append(
                    "patch_grid: skipped (need hsi_agg_mode=multi_token with use_hsi)"
                )

        summary_path = out_dir / "summary.txt"
        summary_text = "\n".join(summary_lines) + "\n"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(summary_text)
        print(f"Saved figures under: {out_dir}")
    finally:
        for g in grabbers:
            g.restore()


if __name__ == "__main__":
    main()
