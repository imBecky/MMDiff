#!/usr/bin/env bash
# =============================================================================
# 多模态分支消融批量跑：7 个非空模态组合
# 组合由 MMDIFF_MODALITY_COMBO 控制（hsi / rgb / lidar / hsi+rgb / ...）
#
# 训练与 TensorBoard 的 run 目录由 param.TB_LOG_ROOT 决定（默认 ../../tf-logs）。
# 本脚本仅把控制台 tee 到同一根目录下的子目录，避免在仓库根目录另建日志目录。
#
# 用法
#   chmod +x run_multimodal_modality_ablation.sh
#   ./run_multimodal_modality_ablation.sh
#
# 可选环境变量：
#   MMDIFF_NUM_EPOCHS                 默认 300
#   MMDIFF_EXPERIMENT_TAG_PREFIX      默认 MODABL
#   TB_LOG_ROOT                       默认 ../../tf-logs（与 param.py 中 TB_LOG_ROOT 一致）
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-200}"

if [[ "${MMDIFF_SKIP_CONDA_ACTIVATE:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate "${CONDA_ENV_NAME:-hbq}" 2>/dev/null || true
fi

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-modality}"
# 与 param.py: TB_LOG_ROOT = Path('../../tf-logs') 对齐（在仓库根目录执行时）
TB_LOG_ROOT="${TB_LOG_ROOT:-../../tf-logs}"

COMBOS=(
  "hsi+rgb+lidar"
  "rgb+lidar"
  "hsi+lidar"
  "hsi+rgb"
  "hsi"
  "rgb"
  "lidar"
)

echo "========== 多模态分支消融批量运行 | prefix=$PREFIX | TB_LOG_ROOT=$TB_LOG_ROOT =========="

for combo in "${COMBOS[@]}"; do
  safe="${combo//[^a-zA-Z0-9_-]/_}"
  export MMDIFF_MODALITY_COMBO="$combo"
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${safe}"
  python main.py 2>&1
done

echo "========== 全部消融跑完。单次 run 产物：${TB_LOG_ROOT}/<run_tag>/（model.log、metrics_summary.json 等）=========="

curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"