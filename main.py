import argparse
import os
import model as Model
from pipeline import TrainingRunOptions, run_training, verify_projection_gradients

os.environ.setdefault("OMP_NUM_THREADS", "4")


def create_classifier(opt_cfg, diffusion):
    return Model.create_multimodal_classifier(opt_cfg, diffusion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多模态分类训练 / 投影梯度验证')
    parser.add_argument(
        '--verify-projection-grad',
        action='store_true',
        help='仅跑一个 batch，检查 projections 在反向中是否有非零梯度，然后退出（退出码 0=通过）',
    )
    parser.add_argument(
        '--no-artifacts',
        action='store_true',
        help='不创建 TensorBoard/断点目录，不写 checkpoint、final、best_model；仅控制台日志，最佳权重保留在内存供 Final test',
    )
    args = parser.parse_args()
    if args.verify_projection_grad:
        verify_projection_gradients(create_classifier)
    else:
        run_training(
            create_classifier,
            TrainingRunOptions(no_artifacts=args.no_artifacts),
        )
