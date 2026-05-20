#!/usr/bin/env python3
"""
快速验证论文级可复现设置（单 batch 前向+反向，不跑完整训练）。

用法（推荐与正式实验一致，先 export 再跑）：

  export MMDIFF_RANDOM_SEED=42 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  python scripts/verify_reproducibility.py

或：

  bash -c 'source <(grep -E "^export (MMDIFF_RANDOM_SEED|PYTHONHASHSEED|CUBLAS)" run.sh | head -3); python scripts/verify_reproducibility.py'

通过标准：同一进程内连续两次 trial 的 loss / logits 校验和完全一致。
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 与 main.py 一致：在 import torch/param 前整理 OMP
_omp = (os.environ.get("OMP_NUM_THREADS") or "").strip()
try:
    if _omp == "" or int(_omp) < 1:
        os.environ["OMP_NUM_THREADS"] = "4"
except ValueError:
    os.environ["OMP_NUM_THREADS"] = "4"


@dataclass
class EnvCheck:
    name: str
    ok: bool
    detail: str


def _check_env(strict: bool) -> List[EnvCheck]:
    checks: List[EnvCheck] = []

    def _one(name: str, ok: bool, detail: str) -> None:
        checks.append(EnvCheck(name, ok, detail))

    seed_raw = (os.environ.get("MMDIFF_RANDOM_SEED") or "").strip()
    _one(
        "MMDIFF_RANDOM_SEED",
        bool(seed_raw),
        seed_raw if seed_raw else "未设置（将使用 --seed 或默认 42）",
    )

    hash_raw = os.environ.get("PYTHONHASHSEED")
    _one(
        "PYTHONHASHSEED",
        hash_raw is not None and str(hash_raw).strip() != "",
        repr(hash_raw) if hash_raw is not None else "未设置（建议 export PYTHONHASHSEED=0）",
    )

    cublas = (os.environ.get("CUBLAS_WORKSPACE_CONFIG") or "").strip()
    cublas_ok = cublas == ":4096:8"
    _one(
        "CUBLAS_WORKSPACE_CONFIG",
        cublas_ok,
        repr(cublas) if cublas else "未设置（须在 python 启动前 export，见 run.sh）",
    )

    if strict:
        for c in checks:
            if c.name == "MMDIFF_RANDOM_SEED" and not c.ok:
                pass  # --seed 可补
    return checks


def _print_env_checks(checks: List[EnvCheck]) -> bool:
    print("=== 环境变量检查 ===")
    all_ok = True
    for c in checks:
        mark = "OK" if c.ok else "WARN"
        if c.name == "CUBLAS_WORKSPACE_CONFIG" and not c.ok:
            all_ok = False
        print(f"  [{mark}] {c.name}: {c.detail}")
    print()
    return all_ok


def _apply_seed_from_cli(seed: Optional[int]) -> int:
    if seed is not None:
        os.environ["MMDIFF_RANDOM_SEED"] = str(int(seed))
    raw = (os.environ.get("MMDIFF_RANDOM_SEED") or "").strip()
    if not raw:
        os.environ["MMDIFF_RANDOM_SEED"] = "42"
        raw = "42"
    return int(raw)


def _one_batch_metrics(seed: int, num_workers: int, device: str) -> Dict[str, Any]:
    import param
    import torch

    import model as Model
    from pipeline.data import (
        batch_to_dict,
        build_dataloaders,
        load_train_bundle,
        split_train_val_indices,
    )
    from pipeline.loop import compute_classification_loss
    from pipeline.runner import _seed_training_for_reproducibility

    param.NUM_WORKERS = int(num_workers)
    if "dataset" in param.opt:
        param.opt["dataset"]["num_workers"] = int(num_workers)

    _seed_training_for_reproducibility(seed)
    dev = torch.device(device)

    feats, rgb, train_ind, _ = load_train_bundle()
    tr_idx, va_idx, tr_pos, va_pos = split_train_val_indices(
        train_ind, param.VAL_RATIO, seed
    )
    train_loader, _, _ = build_dataloaders(
        feats,
        rgb,
        tr_idx,
        va_idx,
        None,
        param.BATCH_SIZE,
        defer_test=True,
        train_global_rows=tr_pos,
        val_global_rows=va_pos,
    )

    model = Model.create_multimodal_classifier(param.opt, None).to(dev)
    model.train()
    loss_fn = model.loss_func
    optimizer = model.optimizer

    batch = next(iter(train_loader))
    data_dict, labels = batch_to_dict(batch, dev, param.USE_RGB_PATCHES)

    optimizer.zero_grad(set_to_none=True)
    loss, logits, _ = compute_classification_loss(
        model, data_dict, labels, loss_fn
    )
    loss.backward()

    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            grad_sq += float((g * g).sum().item())
    grad_l2 = grad_sq**0.5

    with torch.no_grad():
        logits_sum = float(logits.detach().float().sum().item())
        hsi_sum = float(data_dict["hsi"].sum().item())

    return {
        "loss": float(loss.item()),
        "logits_sum": logits_sum,
        "hsi_sum": hsi_sum,
        "grad_l2": grad_l2,
        "batch_size": int(labels.shape[0]),
        "device": str(dev),
        "num_workers": int(num_workers),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _metrics_equal(a: Dict[str, Any], b: Dict[str, Any], rtol: float, atol: float) -> Tuple[bool, List[str]]:
    keys = ("loss", "logits_sum", "hsi_sum", "grad_l2")
    diffs: List[str] = []
    ok = True
    for k in keys:
        va, vb = float(a[k]), float(b[k])
        if va == vb:
            continue
        if abs(va - vb) <= atol + rtol * max(abs(va), abs(vb), 1.0):
            diffs.append(f"{k}: {va!r} vs {vb!r} (在 tol 内)")
            continue
        ok = False
        diffs.append(f"{k}: {va!r} vs {vb!r} (不一致)")
    return ok, diffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="快速验证可复现性（单 batch，无需完整训练）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（写入 MMDIFF_RANDOM_SEED；默认 42）",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="同一进程内重复次数（默认 2，应完全一致）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader num_workers（默认 0，加快启动；与正式训练 24 时数值仍应一致）",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="计算设备（默认 auto）",
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="仅检查环境变量，不加载数据/模型",
    )
    parser.add_argument(
        "--strict-env",
        action="store_true",
        help="缺少 CUBLAS_WORKSPACE_CONFIG 时直接失败退出",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="浮点比较相对容差（论文级建议 0）",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="浮点比较绝对容差（论文级建议 0）",
    )
    args = parser.parse_args()

    checks = _check_env(strict=args.strict_env)
    env_ok = _print_env_checks(checks)
    cublas_set = any(
        c.name == "CUBLAS_WORKSPACE_CONFIG" and c.ok for c in checks
    )
    if args.strict_env and not cublas_set:
        print("[FAIL] --strict-env：未设置 CUBLAS_WORKSPACE_CONFIG=:4096:8")
        return 2
    if not cublas_set:
        print(
            "[提示] 未在 shell 中 export CUBLAS_WORKSPACE_CONFIG；"
            "本脚本无法代替「进程启动前」设置。正式实验请用 bash run.sh。\n"
        )

    seed = _apply_seed_from_cli(args.seed)
    print(f"使用 RANDOM_SEED / MMDIFF_RANDOM_SEED = {seed}\n")

    if args.env_only:
        return 0 if env_ok or not args.strict_env else 2

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("[FAIL] 请求 cuda 但当前无 GPU")
        return 2

    trials = max(2, int(args.trials))
    results: List[Dict[str, Any]] = []
    print(f"=== 单 batch 试验（device={device}, workers={args.workers}, trials={trials}）===")
    for i in range(trials):
        print(f"  trial {i + 1}/{trials} ...", flush=True)
        results.append(_one_batch_metrics(seed, args.workers, device))

    print("\n=== 指标 ===")
    for i, m in enumerate(results):
        print(
            f"  trial {i + 1}: loss={m['loss']:.10f} logits_sum={m['logits_sum']:.10f} "
            f"hsi_sum={m['hsi_sum']:.10f} grad_l2={m['grad_l2']:.10f} "
            f"(batch={m['batch_size']}, cudnn_benchmark={m['cudnn_benchmark']})"
        )

    all_match = True
    print("\n=== 两两对比 ===")
    for i in range(1, len(results)):
        ok, diffs = _metrics_equal(results[0], results[i], args.rtol, args.atol)
        tag = "PASS" if ok else "FAIL"
        print(f"  trial 1 vs trial {i + 1}: [{tag}]")
        for d in diffs:
            print(f"    - {d}")
        all_match = all_match and ok

    if not cublas_set:
        print(
            "\n[注意] 本次未设 CUBLAS_WORKSPACE_CONFIG，即使 PASS 也不能代表论文级 GPU 复现；"
            "请用 run.sh 导出后再跑本脚本。"
        )

    if all_match:
        print("\n[PASS] 同进程内重复 trial 一致。建议再用相同环境连跑两次本脚本（模拟重启）。")
        return 0
    print("\n[FAIL] 指标不一致，请检查 run.sh 三行 export、param.py benchmark=False、CUDA/PyTorch 版本。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
