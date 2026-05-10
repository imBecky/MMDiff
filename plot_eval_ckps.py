#!/usr/bin/env python3
"""
收集已完成的 checkpoint 评估结果，并绘制 OA / AA / Kappa 曲线。

用法示例：
  python plot_eval_ckps.py --eval-dir ../../autodl-tmp/classifier/<run_tag>/eval_all_checkpoints
  python plot_eval_ckps.py --run-dir ../../autodl-tmp/classifier/<run_tag>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def _resolve_eval_dir(run_dir: str, eval_dir: str) -> Path:
    run_dir = (run_dir or "").strip()
    eval_dir = (eval_dir or "").strip()
    if bool(run_dir) == bool(eval_dir):
        raise ValueError("--run-dir 与 --eval-dir 必须二选一")
    if run_dir:
        return (Path(run_dir).expanduser().resolve() / "eval_all_checkpoints").resolve()
    return Path(eval_dir).expanduser().resolve()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_one_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[warn] 跳过损坏 JSON: {metrics_path} ({e})", file=sys.stderr)
        return None

    ckpt_dir = metrics_path.parent
    m = _CKPT_RE.match(ckpt_dir.name)
    if m is None:
        print(f"[warn] 跳过非 checkpoint 目录: {ckpt_dir}", file=sys.stderr)
        return None

    row = {
        "epoch": int(m.group(1)),
        "checkpoint_dir": str(ckpt_dir),
        "metrics_file": str(metrics_path),
        "overall_accuracy": _safe_float(payload.get("overall_accuracy")),
        "average_accuracy": _safe_float(payload.get("average_accuracy")),
        "kappa": _safe_float(payload.get("kappa")),
        "test_loss": _safe_float(payload.get("test_loss")),
        "n_samples": payload.get("n_samples"),
        "checkpoint": payload.get("checkpoint"),
    }
    return row


def collect_finished_results(eval_dir: Path) -> List[Dict[str, Any]]:
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"评估目录不存在: {eval_dir}")

    rows: List[Dict[str, Any]] = []
    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        if _CKPT_RE.match(sub.name) is None:
            continue
        metrics_path = sub / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        row = _load_one_metrics(metrics_path)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda x: int(x["epoch"]))
    return rows


def _build_summary(eval_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    discovered_epochs: List[int] = []
    for sub in eval_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _CKPT_RE.match(sub.name)
        if m is None:
            continue
        discovered_epochs.append(int(m.group(1)))
    discovered_epochs.sort()

    finished_epochs = [int(row["epoch"]) for row in rows]
    finished_epoch_set = set(finished_epochs)
    missing_epochs = [ep for ep in discovered_epochs if ep not in finished_epoch_set]

    def _best(metric_name: str) -> Optional[Dict[str, Any]]:
        valid = [row for row in rows if row.get(metric_name) is not None]
        if not valid:
            return None
        best = max(valid, key=lambda x: float(x[metric_name]))
        return {
            "epoch": int(best["epoch"]),
            "value": float(best[metric_name]),
            "checkpoint_dir": best["checkpoint_dir"],
        }

    return {
        "eval_dir": str(eval_dir),
        "discovered_checkpoint_count": len(discovered_epochs),
        "finished_checkpoint_count": len(rows),
        "finished_epochs": finished_epochs,
        "missing_epochs": missing_epochs,
        "best_overall_accuracy": _best("overall_accuracy"),
        "best_average_accuracy": _best("average_accuracy"),
        "best_kappa": _best("kappa"),
        "results": rows,
    }


def plot_metrics(
    rows: List[Dict[str, Any]],
    output_png: Path,
    title: str,
    include_loss: bool = False,
) -> None:
    if not rows:
        raise ValueError("没有可绘制的结果")

    epochs = [int(row["epoch"]) for row in rows]
    oa = [row.get("overall_accuracy") for row in rows]
    aa = [row.get("average_accuracy") for row in rows]
    kappa = [row.get("kappa") for row in rows]

    if include_loss:
        fig, (ax_main, ax_loss) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(10, 5))
        ax_loss = None

    ax_main.plot(epochs, oa, marker="o", linewidth=1.8, label="OA")
    ax_main.plot(epochs, aa, marker="s", linewidth=1.8, label="AA")
    ax_main.plot(epochs, kappa, marker="^", linewidth=1.8, label="Kappa")
    ax_main.set_ylabel("Metric Value")
    ax_main.set_title(title)
    ax_main.grid(True, linestyle="--", alpha=0.35)
    ax_main.legend()

    if ax_loss is not None:
        loss = [row.get("test_loss") for row in rows]
        ax_loss.plot(epochs, loss, marker="d", linewidth=1.5, color="tab:red", label="Test Loss")
        ax_loss.set_xlabel("Checkpoint Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.35)
        ax_loss.legend()
    else:
        ax_main.set_xlabel("Checkpoint Epoch")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="收集已完成 checkpoint 的 test_metrics.json，并绘制 OA / AA / Kappa 曲线"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="../../autodl-tmp/classifier/exp_0422-1422_exp_esmod_hsi_rgb_lidar",
        help="训练 run 目录，会自动使用 <run-dir>/eval_all_checkpoints",
    )
    parser.add_argument("--eval-dir", type=str, default="", help="eval_ckps.py 的输出目录")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="汇总 JSON 输出路径；默认写到 <eval-dir>/collected_ckpt_metrics.json",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="",
        help="曲线图输出路径；默认写到 <eval-dir>/ckpt_metrics_curve.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Checkpoint Evaluation Metrics",
        help="图标题",
    )
    parser.add_argument(
        "--include-loss",
        action="store_true",
        help="额外在第二个子图绘制 test_loss",
    )
    args = parser.parse_args()

    try:
        eval_dir = _resolve_eval_dir(args.run_dir, args.eval_dir)
        rows = collect_finished_results(eval_dir)
        if not rows:
            print(f"[error] 未在 {eval_dir} 下找到任何已完成的 test_metrics.json", file=sys.stderr)
            return 2
        summary = _build_summary(eval_dir, rows)
        output_json = (
            Path(args.output_json).expanduser().resolve()
            if (args.output_json or "").strip()
            else (eval_dir / "collected_ckpt_metrics.json").resolve()
        )
        output_png = (
            Path(args.output_png).expanduser().resolve()
            if (args.output_png or "").strip()
            else (eval_dir / "ckpt_metrics_curve.png").resolve()
        )

        output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        plot_metrics(rows, output_png=output_png, title=args.title, include_loss=args.include_loss)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        return 2

    print(f"评估目录: {eval_dir}")
    print(f"已收集完成的 checkpoint 数: {len(rows)}")
    print(f"已写入汇总: {output_json}")
    print(f"已写入曲线图: {output_png}")

    for metric_key, label in (
        ("best_overall_accuracy", "best OA"),
        ("best_average_accuracy", "best AA"),
        ("best_kappa", "best Kappa"),
    ):
        item = summary.get(metric_key)
        if not item:
            continue
        print(f"{label}: epoch={item['epoch']} value={item['value']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
