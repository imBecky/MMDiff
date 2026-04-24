#!/usr/bin/env python3
"""
批量评估单次训练 run 目录下的所有 checkpoint-*/classifier.pt。

示例（仓库根目录）:
  python eval_all_checkpoints.py --run-dir ../../autodl-tmp/classifier/<run_tag>
  python eval_all_checkpoints.py --input-path ../../autodl-tmp/classifier/<run_tag>/checkpoint-201
  python eval_all_checkpoints.py --run-dir ... --out-dir ./eval_all_out --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
_INFO_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} .* - [A-Z]+ - ")


def _discover_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    rows: List[Tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = _CKPT_RE.match(p.name)
        if m is None:
            continue
        ckpt_file = p / "classifier.pt"
        if not ckpt_file.is_file():
            continue
        
        step = int(m.group(1))
        if step < 150 and step > 210:
            continue
        
        rows.append((step, ckpt_file))
    rows.sort(key=lambda x: x[0])
    return rows


def _parse_input_path(path_str: str) -> Tuple[Path, Optional[Path]]:
    p = Path(path_str).expanduser().resolve()
    if p.is_file():
        if p.name != "classifier.pt":
            raise ValueError(f"文件输入仅支持 classifier.pt，当前: {p}")
        ckpt_dir = p.parent
        if not _CKPT_RE.match(ckpt_dir.name):
            raise ValueError(f"classifier.pt 必须位于 checkpoint-* 目录下，当前: {ckpt_dir}")
        return ckpt_dir.parent, ckpt_dir

    if not p.is_dir():
        raise ValueError(f"路径不存在或不是目录: {p}")

    if _CKPT_RE.match(p.name):
        return p.parent, p
    if p.name == "final" and p.parent.is_dir():
        return p.parent, None
    return p, None


def _load_training_state(checkpoint_dir: Path) -> Dict[str, Any]:
    ts_path = checkpoint_dir / "training_state.pt"
    if not ts_path.is_file():
        raise FileNotFoundError(f"找不到 training_state.pt: {ts_path}")
    import torch

    state = torch.load(str(ts_path), map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError(f"training_state.pt 不是字典结构: {ts_path}")
    return state


def _extract_from_line(pattern: str, line: str) -> Optional[str]:
    m = re.search(pattern, line)
    return m.group(1).strip() if m else None


def _strip_log_prefix(line: str) -> str:
    return _INFO_PREFIX_RE.sub("", line.strip())


def _parse_model_log_to_env(model_log: Path) -> Dict[str, str]:
    text = model_log.read_text(encoding="utf-8", errors="ignore")
    envs: Dict[str, str] = {}
    for raw in text.splitlines():
        line = _strip_log_prefix(raw)
        if not line:
            continue
        if "实验标识 |" in line:
            v = _extract_from_line(r"modality_combo=([^\s]+)", line)
            if v:
                envs["MMDIFF_MODALITY_COMBO"] = v
        elif "训练超参 |" in line:
            v = _extract_from_line(r"batch_size=([0-9]+)", line)
            if v:
                envs["MMDIFF_BATCH_SIZE"] = v
        elif line.startswith("scheduler="):
            v = _extract_from_line(r"scheduler=([^\s|]+)", line)
            if v:
                envs["MMDIFF_SCHEDULER_NAME"] = v
            for pat, k in (
                (r"eta_min_ratio=([0-9eE+.\-]+)", "MMDIFF_SCHED_COSINE_ETA_MIN_RATIO"),
                (r"warmup_ratio=([0-9eE+.\-]+)", "MMDIFF_SCHED_COSINE_WARMUP_RATIO"),
                (r"warmup_steps=([0-9eE+.\-]+)", "MMDIFF_SCHED_COSINE_WARMUP_STEPS"),
            ):
                vv = _extract_from_line(pat, line)
                if vv:
                    envs[k] = vv
        elif "数据与划分 |" in line:
            v = _extract_from_line(r"random_seed=([0-9]+)", line)
            if v:
                envs["MMDIFF_RANDOM_SEED"] = v
        elif "multimodal_ablation:" in line:
            for pat, k in (
                (r"modalities=([^\s|]+)", "MMDIFF_MODALITY_COMBO"),
                (r"lidar_hidden=([0-9]+)", "MMDIFF_LIDAR_HIDDEN"),
                (r"lidar_extra_blocks=([0-9]+)", "MMDIFF_LIDAR_EXTRA_BLOCKS"),
                (r"hsi_res_blocks=([0-9]+)", "MMDIFF_HSI_RESIDUAL_BLOCKS"),
                (r"hsi_conv_hidden=([0-9]+)", "MMDIFF_HSI_CONV_HIDDEN"),
                (r"hsi_se_ratio=([0-9]+)", "MMDIFF_HSI_SE_RATIO"),
                (r"hsi_agg_mode=([^\s|]+)", "MMDIFF_HSI_AGG_MODE"),
                (r"rgb_to_lidar_guidance=([^\s|]+)", "MMDIFF_RGB_TO_LIDAR_GUIDANCE"),
            ):
                vv = _extract_from_line(pat, line)
                if vv:
                    envs[k] = vv
        elif "model_cls | token_dim=" in line:
            for pat, k in (
                (r"token_dim=([0-9]+)", "MMDIFF_CLS_TOKEN_DIM"),
                (r"layers=([0-9]+)", "MMDIFF_CLS_TRANSFORMER_LAYERS"),
                (r"ff=([0-9]+)", "MMDIFF_CLS_TRANSFORMER_FF_DIM"),
                (r"head_hidden=([0-9]+)", "MMDIFF_CLS_HEAD_HIDDEN"),
            ):
                vv = _extract_from_line(pat, line)
                if vv:
                    envs[k] = vv
        elif "model_cls | rgb_to_lidar_guidance_mode=" in line:
            v = _extract_from_line(
                r"rgb_to_lidar_guidance_mode=([^\s|]+)", line
            )
            if v:
                envs["MMDIFF_RGB_TO_LIDAR_GUIDANCE"] = v
        elif "rgb student |" in line:
            v = _extract_from_line(r"MMDIFF_RGB_STUDENT_CHECKPOINT=([^\s]+)", line)
            if v:
                envs["MMDIFF_RGB_STUDENT_CHECKPOINT"] = v

    required = [
        "MMDIFF_MODALITY_COMBO",
        "MMDIFF_CLS_TOKEN_DIM",
        "MMDIFF_HSI_AGG_MODE",
    ]
    missing = [k for k in required if not (envs.get(k) or "").strip()]
    if missing:
        raise RuntimeError(
            f"model.log 关键配置缺失: {missing}（日志不足或非本仓库训练产物）"
        )
    return envs


def _resolve_model_log_from_checkpoint(
    checkpoint_dir: Path, run_dir: Path
) -> Tuple[Path, Dict[str, Any]]:
    state = _load_training_state(checkpoint_dir)
    run_log_dir = str(state.get("run_log_dir") or "").strip()
    if run_log_dir:
        model_log = Path(run_log_dir).expanduser().resolve() / "model.log"
        if model_log.is_file():
            return model_log, state
    fallback = run_dir / "model.log"
    if fallback.is_file():
        return fallback, state
    raise FileNotFoundError(
        f"无法定位 model.log（run_log_dir={run_log_dir!r}, fallback={fallback})"
    )


def _apply_env_overrides(envs: Dict[str, str]) -> None:
    for k, v in envs.items():
        if (v or "").strip():
            os.environ[k] = str(v).strip()


def _load_run_single_eval(repo_root: Path) -> Callable[..., Dict[str, Any]]:
    mod_path = repo_root / "pipeline" / "eval_test.py"
    if not mod_path.is_file():
        raise FileNotFoundError(f"找不到评估脚本: {mod_path}")
    spec = importlib.util.spec_from_file_location("mmdiff_eval_test_mod", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run_single_eval", None)
    if not callable(fn):
        raise RuntimeError(f"{mod_path} 未导出可调用 run_single_eval")
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量评估 run 目录下所有 checkpoint-*/classifier.pt"
    )
    parser.add_argument("--run-dir", type=str, default="../../autodl-tmp/classifier/previous/exp_0422-1422_exp_esmod_hsi_rgb_lidar/final", help="单次训练 run 目录")
    parser.add_argument(
        "--input-path",
        type=str,
        default="",
        help="可直接传 checkpoint-*/、final/ 或 checkpoint-*/classifier.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="批量结果根目录；默认写到 <run-dir>/eval_all_checkpoints",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要评估的 checkpoint 列表，不执行评估",
    )
    args = parser.parse_args()

    in_run = (args.run_dir or "").strip()
    in_path = (args.input_path or "").strip()
    if bool(in_run) == bool(in_path):
        print("[error] --run-dir 与 --input-path 二选一", file=sys.stderr)
        return 2

    try:
        if in_run:
            run_dir, seed_ckpt = _parse_input_path(in_run)
        else:
            run_dir, seed_ckpt = _parse_input_path(in_path)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        return 2

    found = _discover_checkpoints(run_dir)
    if not found:
        print(f"[error] 未找到 checkpoint-*/classifier.pt: {run_dir}", file=sys.stderr)
        return 2
    if seed_ckpt is None:
        seed_ckpt = found[0][1].parent

    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if (args.out_dir or "").strip()
        else (run_dir / "eval_all_checkpoints").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent
    try:
        model_log, training_state = _resolve_model_log_from_checkpoint(seed_ckpt, run_dir)
        recovered_env = _parse_model_log_to_env(model_log)
        slr_steps = training_state.get("scheduler_lr_total_steps", None)
        if slr_steps is not None:
            recovered_env["MMDIFF_SCHEDULER_LR_TOTAL_STEPS"] = str(int(slr_steps))
        _apply_env_overrides(recovered_env)
    except Exception as e:  # noqa: BLE001
        print(f"[error] 自动恢复评估环境失败: {e}", file=sys.stderr)
        return 2

    print(f"run_dir={run_dir}")
    print(f"seed_checkpoint_dir={seed_ckpt}")
    print(f"model_log={model_log}")
    print(f"out_dir={out_root}")
    print(f"共发现 {len(found)} 个 checkpoint")
    for ep, ckpt_file in found:
        print(f"  - epoch={ep}  file={ckpt_file}")
    print("自动恢复的关键 MMDIFF_*:")
    for k in sorted(recovered_env):
        print(f"  - {k}={recovered_env[k]}")
    if args.dry_run:
        return 0

    run_single_eval = _load_run_single_eval(repo_root)

    ok_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []

    for ep, ckpt_file in found:
        sub_out = out_root / f"checkpoint-{ep}"
        try:
            metrics = run_single_eval(
                str(ckpt_file),
                out_dir=str(sub_out),
                batch_size=args.batch_size,
                device_name=args.device,
            )
            row = {
                "epoch": int(ep),
                "checkpoint_dir": str(ckpt_file.parent),
                "checkpoint_path": str(ckpt_file),
                "metrics_file": str(sub_out / "test_metrics.json"),
                "overall_accuracy": metrics.get("overall_accuracy"),
                "average_accuracy": metrics.get("average_accuracy"),
                "kappa": metrics.get("kappa"),
                "test_loss": metrics.get("test_loss"),
            }
            ok_rows.append(row)
            print(
                "[ok] epoch=%d oa=%.6f aa=%.6f kappa=%.6f"
                % (
                    ep,
                    float(metrics.get("overall_accuracy", 0.0)),
                    float(metrics.get("average_accuracy", 0.0)),
                    float(metrics.get("kappa", 0.0)),
                )
            )
        except Exception as e:  # noqa: BLE001
            fail = {
                "epoch": int(ep),
                "checkpoint_dir": str(ckpt_file.parent),
                "checkpoint_path": str(ckpt_file),
                "error": repr(e),
            }
            fail_rows.append(fail)
            print(f"[fail] epoch={ep} checkpoint={ckpt_file} error={e}", file=sys.stderr)

    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_root),
        "total_checkpoints": int(len(found)),
        "success_count": int(len(ok_rows)),
        "failure_count": int(len(fail_rows)),
        "results": ok_rows,
        "failures": fail_rows,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"已写入汇总: {summary_path}")
    return 0 if not fail_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
