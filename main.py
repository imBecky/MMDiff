import os

# 必须在 import torch 之前：空串 OMP_NUM_THREADS 会触发 libgomp 报错
_omp = (os.environ.get("OMP_NUM_THREADS") or "").strip()
try:
    if _omp == "" or int(_omp) < 1:
        os.environ["OMP_NUM_THREADS"] = "4"
except ValueError:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import sys
from pathlib import Path


def _apply_cli_seed_before_param_import() -> None:
    """
    在 import model/pipeline/param 之前执行。
    - 若未传 ``--seed``：不写环境变量。
    - 若已设置 ``MMDIFF_RANDOM_SEED`` 且与 ``--seed`` 数值不同：报错退出。
    - 若未设置 ``MMDIFF_RANDOM_SEED`` 且传了 ``--seed``：写入 ``MMDIFF_RANDOM_SEED``，使 param 与 runner 仍为唯一归宿。
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--seed", type=int, default=None)
    args_early, _ = probe.parse_known_args()
    if args_early.seed is None:
        return
    raw = (os.environ.get("MMDIFF_RANDOM_SEED") or "").strip()
    if raw != "":
        try:
            env_seed = int(raw)
        except ValueError:
            print(
                f"MMDIFF_RANDOM_SEED={raw!r} 无法解析为整数，请先修正环境变量或使用 --seed 前 unset。",
                file=sys.stderr,
            )
            sys.exit(2)
        if env_seed != int(args_early.seed):
            print(
                '[错误] `--seed=%s` 与 `MMDIFF_RANDOM_SEED=%s`（=%d）不一致；'
                "请只保留其一或两者设为同一整数后再运行。"
                % (args_early.seed, raw, env_seed),
                file=sys.stderr,
            )
            sys.exit(2)
        return
    os.environ["MMDIFF_RANDOM_SEED"] = str(int(args_early.seed))


if __name__ == "__main__":
    _apply_cli_seed_before_param_import()
    import param
    from pipeline.runner import _seed_training_for_reproducibility

    _seed_training_for_reproducibility(param.RANDOM_SEED)

import model as Model
from pipeline import TrainingRunOptions, run_training, verify_projection_gradients
from utils.training_control_variable_summary import emit_training_control_variable_summary


def create_classifier(opt_cfg, diffusion=None):
    """diffusion 参数已废弃，保留仅为与 runner 签名兼容。"""
    return Model.create_multimodal_classifier(opt_cfg, diffusion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态分类训练 / 投影梯度验证")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "随机种子（无 MMDIFF_RANDOM_SEED 时写入该环境变量，使与 param.RANDOM_SEED 一致）；"
            "若环境中已设置 MMDIFF_RANDOM_SEED 则必须与此处数值一致，否则报错退出"
        ),
    )
    parser.add_argument(
        "--verify-projection-grad",
        action="store_true",
        help="仅跑一个 batch，检查 projections 在反向中是否有非零梯度，然后退出（退出码 0=通过）",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="不创建 TensorBoard/断点目录，不写 checkpoint、final、best_model；仅控制台日志，最佳权重保留在内存供 Final test",
    )
    parser.add_argument(
        "--no-conf-detail",
        action="store_true",
        help="不单独写入混淆矩阵/误分类对日志 conf_detail.log（默认会写）",
    )
    args = parser.parse_args()
    if args.verify_projection_grad:
        from param import LOG_PATH

        emit_training_control_variable_summary(no_artifacts=False, log_file=Path(LOG_PATH))
        verify_projection_gradients(create_classifier)
    else:
        emit_training_control_variable_summary(no_artifacts=args.no_artifacts)
        run_training(
            create_classifier,
            TrainingRunOptions(
                no_artifacts=args.no_artifacts,
                save_conf_detail=not args.no_conf_detail,
            ),
        )
