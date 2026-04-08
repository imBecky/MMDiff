"""
对比实验入口：与主训练隔离，仅通过环境变量 MMDIFF_COMPARE_RUN=1 关闭 SupCon / 双头 CE。

用法（仓库根目录）：
  python utils/main_compare.py --model fgcnn
  MMDIFF_EXPERIMENT_TAG=cmp_x python utils/main_compare.py --model fgcnn

环境变量（可选）：
  MMDIFF_COMPARE_MODEL       与 --model 等价（一般无需单独设）
  MMDIFF_COMPARE_RUN_NAME_PREFIX  默认 cmp，影响 TB run 名前缀
  MMDIFF_COMPARE_CKPS_DIR    覆盖断点根目录（默认仍用 param.CKPS_DIR）
  MMDIFF_EXPERIMENT_TAG      单次实验 tag
  MMDIFF_NUM_EPOCHS 等       与 main.py / param 一致
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 从 utils/ 子目录运行时，默认 sys.path 不含仓库根，无法 import model
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 与 main.py 一致：空或非正整数 OMP_NUM_THREADS 会触发 libgomp 报错
_omp = (os.environ.get('OMP_NUM_THREADS') or '').strip()
if not _omp.isdigit() or int(_omp) <= 0:
    os.environ['OMP_NUM_THREADS'] = '4'


def _bootstrap_env():
    os.environ['MMDIFF_COMPARE_RUN'] = '1'
    os.environ.setdefault('MMDIFF_USE_SUPCON', '0')


def main():
    parser = argparse.ArgumentParser(description='对比模型训练（复用 pipeline，不改主模型逻辑）')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help=(
            '模型注册名，如 fgcnn fusatnet exvit two_branch_cnn dfinet macn ss_mae；'
            'DFINet 官方仓库名别名：formango_dfinet / hsi_msi_multisource（'
            'https://github.com/formango/HSI_MSI_Multisource_Classification ）；'
            'SS-MAE：https://github.com/summitgao/SS-MAE'
        ),
    )
    parser.add_argument(
        '--verify-projection-grad',
        action='store_true',
        help='单 batch 检查 projections 子模块梯度（对比模型一般可跳过）',
    )
    parser.add_argument(
        '--no-artifacts',
        action='store_true',
        help='不写入 TB/断点文件，仅控制台',
    )
    parser.add_argument(
        '--no-conf-detail',
        action='store_true',
        help='不写 conf_detail.log',
    )
    args = parser.parse_args()

    _bootstrap_env()
    os.environ['MMDIFF_COMPARE_MODEL'] = args.model.strip().lower()

    # 须在 import param / pipeline 之前设好环境变量
    from model.compare_model import create_compare_classifier
    from pipeline import TrainingRunOptions, run_training, verify_projection_gradients

    def create_classifier(opt_cfg, diffusion):
        return create_compare_classifier(opt_cfg, diffusion)

    if args.verify_projection_grad:
        verify_projection_gradients(create_classifier)
    else:
        run_training(
            create_classifier,
            TrainingRunOptions(
                no_artifacts=args.no_artifacts,
                save_conf_detail=not args.no_conf_detail,
            ),
        )


if __name__ == '__main__':
    main()
