"""RGB 扩散 teacher token 离线缓存：读写与索引。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def default_meta(
    *,
    n_rows: int,
    n_rot: int,
    num_tokens: int,
    d_model: int,
    feat_scales: list[str],
    diffusion_ts: list[int],
    diffusion_teacher_checkpoint: str,
) -> dict[str, Any]:
    s = str(diffusion_teacher_checkpoint)
    return {
        'version': 1,
        'n_rows': int(n_rows),
        'n_rot': int(n_rot),
        'num_tokens': int(num_tokens),
        'd_model': int(d_model),
        'feat_scales': list(feat_scales),
        'diffusion_ts': [int(x) for x in diffusion_ts],
        'diffusion_teacher_checkpoint': s,
        'student_checkpoint': s,
    }


def save_meta(path: PathLike, meta: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_meta(path: PathLike) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'缺少 meta 文件: {p}')
    return json.loads(p.read_text(encoding='utf-8'))


def mmap_tokens(path: PathLike) -> np.ndarray:
    """内存映射只读 teacher token 数组。

    标准 .npy（含 magic）用 ``np.load(..., mmap_mode='r')``。
    旧版 precompute 曾用裸 ``np.memmap`` 写无头原始 float32，需同目录 ``{stem}.meta.json`` 推断 shape。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'缺少 teacher token 缓存: {p}')
    with open(p, 'rb') as f:
        magic = f.read(6)
    if magic == b'\x93NUMPY':
        return np.load(str(p), mmap_mode='r', allow_pickle=False)
    meta_path = p.parent / f'{p.stem}.meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(
            f'缓存 {p} 不是标准 .npy（无 NUMPY magic），且缺少 {meta_path}，无法 mmap'
        )
    meta = load_meta(meta_path)
    shape = (
        int(meta['n_rows']),
        int(meta['n_rot']),
        int(meta['num_tokens']),
        int(meta['d_model']),
    )
    expected = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    sz = p.stat().st_size
    if sz != expected:
        raise ValueError(
            f'缓存 {p} 字节数 {sz} 与 meta 期望 {expected} 不符，请删缓存后重新 precompute'
        )
    return np.memmap(str(p), dtype=np.float32, mode='r', shape=shape)


def gather_tokens(
    cache: np.ndarray,
    global_row: torch.Tensor,
    rot_k: torch.Tensor,
) -> torch.Tensor:
    """
    cache: (N, n_rot, num_tokens, d_model) 或 (N, 1, num_tokens, d_model)（测试集仅 rot0）
    global_row, rot_k: 1D long tensor，长度 B
    返回 float32 tensor (B, num_tokens, d_model)
    """
    if cache.ndim != 4:
        raise ValueError(f'cache 期望 4 维，当前 shape={cache.shape}')
    n_rot = cache.shape[1]
    gr = global_row.detach().cpu().numpy().astype(np.int64)
    rk = rot_k.detach().cpu().numpy().astype(np.int64)
    if n_rot == 1:
        rk = np.zeros_like(rk)
    else:
        rk = np.clip(rk, 0, n_rot - 1)
    out = np.empty((gr.shape[0], cache.shape[2], cache.shape[3]), dtype=np.float32)
    for i in range(gr.shape[0]):
        out[i] = np.asarray(cache[gr[i], rk[i]], dtype=np.float32)
    return torch.from_numpy(out)
