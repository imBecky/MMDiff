#!/usr/bin/env python3
"""
将旧版 precompute 写的「无 NUMPY magic 的裸 float32」teacher 缓存，转成标准 .npy，
避免重跑慢速扩散预计算。

依赖同目录 ``{文件名 stem}.meta.json``（与 ``utils/precompute_rgb_teacher_tokens.py`` 一致）。

用法（仓库根目录）::

  python tempt.py --input /path/to/rgb_teacher_tokens_train.npy
  python tempt.py --input .../train.npy --output .../train_std.npy
  python tempt.py --input .../train.npy --in-place   # 原文件 -> .bak，再写同名标准 .npy

若输入已是标准 .npy，会提示并退出 0。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.rgb_teacher_cache import load_meta  # noqa: E402


def _resolve_meta_path(inp: Path, meta_override: Path | None) -> Path:
    """precompute 写的是 ``<stem>.meta.json``，例如 ``rgb_teacher_tokens_train.meta.json``。"""
    if meta_override is not None:
        p = meta_override.resolve()
        if not p.is_file():
            raise FileNotFoundError(f'--meta 指定的文件不存在: {p}')
        return p
    cand = inp.parent / f'{inp.stem}.meta.json'
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        '找不到 meta。precompute 默认生成同目录 '
        f'「{inp.stem}.meta.json」（不是「{inp.name}.meta.json」）。'
        ' 若文件在别处: python tempt.py --input ... --meta /path/to/xxx.meta.json'
    )


def _is_standard_npy(p: Path) -> bool:
    with open(p, 'rb') as f:
        return f.read(6) == b'\x93NUMPY'


def _shape_from_meta(meta_path: Path) -> tuple[int, int, int, int]:
    m = load_meta(meta_path)
    return (
        int(m['n_rows']),
        int(m['n_rot']),
        int(m['num_tokens']),
        int(m['d_model']),
    )


def convert(
    inp: Path,
    out: Path,
    *,
    row_chunk: int,
    meta_override: Path | None = None,
) -> None:
    meta_path = _resolve_meta_path(inp, meta_override)
    shape = _shape_from_meta(meta_path)
    expected = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    sz = inp.stat().st_size
    if sz != expected:
        raise ValueError(
            f'文件字节 {sz} 与 meta 期望 {expected}（shape={shape}）不符，无法转换'
        )

    if _is_standard_npy(inp):
        print(f'已是标准 .npy，跳过: {inp}')
        return

    old = np.memmap(str(inp), dtype=np.float32, mode='r', shape=shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = open_memmap(str(out), mode='w+', dtype=np.float32, shape=shape)
    n0 = shape[0]
    rc = max(1, int(row_chunk))
    for s in range(0, n0, rc):
        e = min(s + rc, n0)
        new[s:e] = old[s:e]
    new.flush()
    del new, old
    print(f'已写入标准 .npy: {out} shape={shape}')


def main() -> None:
    p = argparse.ArgumentParser(description='裸 teacher token -> 标准 .npy')
    p.add_argument('--input', type=str, required=True, help='旧 .npy 路径（裸 float32）')
    p.add_argument(
        '--meta',
        type=str,
        default='',
        help='meta.json 路径（默认与 .npy 同目录的 <stem>.meta.json）',
    )
    p.add_argument('--output', type=str, default='', help='输出路径；默认 <input>.std.npy')
    p.add_argument(
        '--in-place',
        action='store_true',
        help='原地替换：先把 input 改名为 .bak，再写入与 input 同名的标准 .npy',
    )
    p.add_argument('--row-chunk', type=int, default=256, help='按行块拷贝，省峰值内存')
    args = p.parse_args()

    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f'找不到 input: {inp}')
    meta_ov = Path(args.meta).resolve() if str(args.meta).strip() else None

    if args.in_place:
        if args.output:
            raise SystemExit('不能同时指定 --in-place 与 --output')
        bak = inp.with_suffix(inp.suffix + '.bak')
        if _is_standard_npy(inp):
            print(f'已是标准 .npy: {inp}')
            return
        # 移动成 .bak 后 stem 会变成「xxx.npy」，meta 必须在移动前按原始 stem 解析
        meta_for_convert = meta_ov or (inp.parent / f'{inp.stem}.meta.json')
        if not meta_for_convert.is_file():
            raise SystemExit(
                f'缺少 meta: {meta_for_convert}\n'
                f'precompute 输出应为同目录「{inp.stem}.meta.json」，不是「{inp.name}.meta.json」。'
            )
        shutil.move(str(inp), str(bak))
        try:
            convert(bak, inp, row_chunk=args.row_chunk, meta_override=meta_for_convert)
            print(f'原文件已备份: {bak}')
        except Exception:
            shutil.move(str(bak), str(inp))
            raise
        return

    out = Path(args.output) if args.output else inp.with_suffix('.std' + inp.suffix)
    out = out.resolve()
    convert(inp, out, row_chunk=args.row_chunk, meta_override=meta_ov)


if __name__ == '__main__':
    main()
