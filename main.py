import os

# 必须在 import torch 之前：空串 OMP_NUM_THREADS 会触发 libgomp 报错
_omp = (os.environ.get("OMP_NUM_THREADS") or "").strip()
try:
    if _omp == "" or int(_omp) < 1:
        os.environ["OMP_NUM_THREADS"] = "4"
except ValueError:
    os.environ["OMP_NUM_THREADS"] = "4"

import argparse
from pathlib import Path

import model as Model
from pipeline import TrainingRunOptions, run_training, verify_projection_gradients
from utils.training_control_variable_summary import emit_training_control_variable_summary


def create_classifier(opt_cfg, diffusion):
    return Model.create_multimodal_classifier(opt_cfg, diffusion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态分类训练 / 投影梯度验证")
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
