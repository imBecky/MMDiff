#!/usr/bin/env bash
# =============================================================================
# 批量对比实验：依次训练各官方对照模型，控制台 + tee 到 compare_logs/
#
# 用法（仓库根目录，Git Bash / Linux）：
#   chmod +x run_compare.sh && ./run_compare.sh
#
# 环境变量：
#   MMDIFF_NUM_EPOCHS              默认 300
#   MMDIFF_EXPERIMENT_TAG_PREFIX   默认 cmp（单次 tag: ${PREFIX}_${model}）
#   COMPARE_LOG_ROOT               默认 ./compare_logs
#   COMPARE_MODELS                 空格分隔，覆盖默认列表
#   MMDIFF_SKIP_CONDA_ACTIVATE=1   不尝试 conda activate
#   CONDA_ENV_NAME                 默认 hbq
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MMDIFF_COMPARE_RUN="${MMDIFF_COMPARE_RUN:-1}"
export MMDIFF_USE_SUPCON="${MMDIFF_USE_SUPCON:-0}"

unset MMDIFF_RESUME_CHECKPOINT 2>/dev/null || true

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-cmp}"
LOG_ROOT="${COMPARE_LOG_ROOT:-./compare_logs}"
mkdir -p "$LOG_ROOT"

DEFAULT_MODELS="coupled_cnn fusatnet macn hct exvit ss_mae msfmamba dcmnet"
MODELS_ARR=(${COMPARE_MODELS:-$DEFAULT_MODELS})

if [[ "${MMDIFF_SKIP_CONDA_ACTIVATE:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate "${CONDA_ENV_NAME:-hbq}" 2>/dev/null || true
fi

echo "========== 对比实验批量运行 | PREFIX=${PREFIX} | LOG_ROOT=${LOG_ROOT} =========="

for m in "${MODELS_ARR[@]}"; do
  safe="${m//[^a-zA-Z0-9_-]/_}"
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${safe}"
  logfile="${LOG_ROOT}/${PREFIX}_${safe}.log"
  echo "---------- 模型=${m} | tag=${MMDIFF_EXPERIMENT_TAG} | tee=${logfile} ----------"
  python main_compare.py --model "$m" 2>&1 | tee "$logfile"
done

echo "========== 全部对比跑完。各次 metrics_summary.json 在 tf-logs 下对应 run 目录内 =========="
