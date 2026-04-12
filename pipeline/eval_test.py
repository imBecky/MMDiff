#!/usr/bin/env python3
"""
在官方 test 划分（test_labels.npy）上评估单个 classifier.pt，与训练末尾 Final test 一致。

用法（仓库根目录）:
  python eval_test.py --checkpoint ../../autodl-tmp/classifier/.../checkpoint-205/classifier.pt
  python eval_test.py --checkpoint .../classifier.pt --out-dir ./out_metrics

汇总某次 sweep 下各子目录的 test_metrics.json:
  python eval_test.py --collect-sweep ../../autodl-tmp/classifier/.../eval_sweep_test

环境与结构须与训练一致（param / MMDIFF_*）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_single_eval(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    import model as Model
    from param import (
        BATCH_SIZE,
        DIFFUSION_NOISE_MODE,
        DIFFUSION_NORMALIZE_INPUT,
        FEAT_SCALES,
        NUM_CLASSES,
        RANDOM_SEED,
        RGB_DIFFUSION_TEACHER_CHECKPOINT,
        RGB_SOURCE,
        STUDENT_NUM_TRAIN_TIMESTEPS,
        USE_CENTER_LOSS,
        USE_RGB_PATCHES,
        opt,
    )
    from pipeline.classification_metrics import accuracies
    from pipeline.data import (
        build_test_loader,
        load_rgb_hr_meta,
        load_rgb_hr_volume,
        load_test_indices_shifted,
        load_train_bundle,
    )
    from pipeline.logging_utils import get_console_logger
    from pipeline.loop import evaluate
    from pipeline.student_diffusion import StudentDiffusionWrapper

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        print(f"[error] 找不到 checkpoint 文件: {ckpt}", file=sys.stderr)
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger = get_console_logger()

    feats_vol, rgb_vol, _train_indices, label_shift = load_train_bundle()
    rgb_hr_vol = None
    hr_rh = 1
    hr_rw = 1
    if USE_RGB_PATCHES:
        rgb_hr_vol = load_rgb_hr_volume()
        _hr_meta = load_rgb_hr_meta()
        hr_rh = int(_hr_meta["rh"])
        hr_rw = int(_hr_meta["rw"])

    test_idx = load_test_indices_shifted(label_shift)
    bs = int(args.batch_size) if args.batch_size is not None else int(BATCH_SIZE)

    test_loader = build_test_loader(
        feats_vol,
        rgb_vol,
        test_idx,
        bs,
        rgb_strict_view=bool(USE_RGB_PATCHES),
        rgb_hr_vol=rgb_hr_vol,
        hr_rh=hr_rh,
        hr_rw=hr_rw,
    )

    def create_classifier(diffusion):
        return Model.create_multimodal_classifier(opt, diffusion)

    diffusion = None
    rgb_src = (RGB_SOURCE or "student").strip().lower()
    if rgb_src == "diffusion":
        diffusion = StudentDiffusionWrapper(
            RGB_DIFFUSION_TEACHER_CHECKPOINT,
            STUDENT_NUM_TRAIN_TIMESTEPS,
            noise_mode=DIFFUSION_NOISE_MODE,
            noise_seed_base=RANDOM_SEED,
            normalize_diffusion_input=DIFFUSION_NORMALIZE_INPUT,
            feat_layers=FEAT_SCALES,
        )
    elif rgb_src not in ("student", "cached_teacher"):
        print(f"[error] 无效 MMDIFF_RGB_SOURCE: {rgb_src!r}", file=sys.stderr)
        return 2

    model = create_classifier(diffusion).to(device)
    loss_fn = model.loss_func

    try:
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(
            "\n[hint] 请 export 与训练一致的 MMDIFF_*，或复制 eval_env.example 为 eval_env.sh。\n",
            file=sys.stderr,
        )
        raise e

    model.eval()
    use_center = bool(USE_CENTER_LOSS)

    preds, targets, conf, eval_loss, eval_acc = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        NUM_CLASSES,
        logger,
        writer=None,
        epoch=0,
        split="test",
        use_center_logits=use_center,
    )
    ovr_acc, usr_acc, prod_acc, kappa, s_sqr, aa = accuracies(conf)

    def _scalar(x: Any) -> float:
        a = np.asarray(x).squeeze()
        if a.ndim != 0:
            raise TypeError(f"期望标量，得到 shape={a.shape}")
        return float(a.item())

    def _per_class(x: Any) -> List[float]:
        return [float(v) for v in np.asarray(x).ravel()]

    n_test = int(len(targets))
    payload: Dict[str, Any] = {
        "checkpoint": str(ckpt),
        "split": "test",
        "n_samples": n_test,
        "test_loss": float(eval_loss),
        "test_acc_sample_mean": float(eval_acc),
        "overall_accuracy": _scalar(ovr_acc),
        "average_accuracy": float(aa),
        # classification_metrics.accuracies：usr/prod 为逐类向量（UA / PA），非单个 float
        "user_accuracy_per_class": _per_class(usr_acc),
        "producer_accuracy_per_class": _per_class(prod_acc),
        "kappa": _scalar(kappa),
        "kappa_variance": _scalar(s_sqr),
        "oa_percent": float(round(100 * _scalar(ovr_acc), 4)),
        "aa_percent": float(round(100 * aa, 4)),
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "test_metrics.json").write_text(text, encoding="utf-8")
        logger.info("已写入 %s", out / "test_metrics.json")

    return 0


def _collect_sweep(sweep_dir: Path) -> int:
    """合并 sweep 根目录下各子目录中的 test_metrics.json -> sweep_test_summary.json"""
    sweep_dir = sweep_dir.expanduser().resolve()
    if not sweep_dir.is_dir():
        print(f"[error] 不是目录: {sweep_dir}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / "test_metrics.json"
        if not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[warn] 跳过损坏的 {p}: {e}", file=sys.stderr)
            continue
        obj["_subdir"] = sub.name
        rows.append(obj)

    out_path = sweep_dir / "sweep_test_summary.json"
    out_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"共 {len(rows)} 条，已写入 {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Test 集评估（与训练 Final test 一致）")
    p.add_argument(
        "--collect-sweep",
        type=str,
        default="",
        help="仅汇总：扫描该目录下各子目录的 test_metrics.json，写入 sweep_test_summary.json",
    )
    p.add_argument("--checkpoint", type=str, default="", help="classifier.pt 路径")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="写入 test_metrics.json 的目录（可选）",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if (args.collect_sweep or "").strip():
        return _collect_sweep(Path(args.collect_sweep.strip()))

    if not (args.checkpoint or "").strip():
        p.error("请提供 --checkpoint，或使用 --collect-sweep")
    return _run_single_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
