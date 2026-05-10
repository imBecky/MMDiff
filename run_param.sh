#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ===========================
# 固定配置（按需手改这里）
# ===========================
SEEDS=("42")
MODALITY="hsi+rgb+lidar"
OMP_THREADS="4"

# 固定主干配置，避免实验间漂移
HSI_SE_RATIO="32"
HSI_RESIDUAL_BLOCKS="5"
HSI_CONV_HIDDEN="96"
HSI_AGG_MODE="mean"
LIDAR_HIDDEN="48"
LIDAR_EXTRA_BLOCKS="2"
CLS_TRANSFORMER_LAYERS="1"
CLS_TRANSFORMER_FF_DIM="384"

# 只保留 Loss 权重消融：中心权重=0.66/0.68/0.72/0.74/0.76，全局=1-中心
LOSS_CASES=(
  "loss_g034_c066|0.34|0.66"
  "loss_g032_c068|0.32|0.68"
  "loss_g028_c072|0.28|0.72"
  "loss_g026_c074|0.26|0.74"
  "loss_g024_c076|0.24|0.76"
)

clear_case_env() {
  unset \
    MMDIFF_PATCH_WINDOW_SIZE \
    MMDIFF_LOSS_WEIGHT_GLOBAL \
    MMDIFF_LOSS_WEIGHT_CENTER \
    MMDIFF_CLS_TOKEN_DIM \
    MMDIFF_CLS_HEAD_HIDDEN \
    MMDIFF_EXPERIMENT_TAG \
    MMDIFF_RUN_TIMESTAMP \
    2>/dev/null || true
}

set_fixed_env() {
  export OMP_NUM_THREADS="${OMP_THREADS}"
  export MMDIFF_MODALITY_COMBO="${MODALITY}"
  export MMDIFF_HSI_SE_RATIO="${HSI_SE_RATIO}"
  export MMDIFF_HSI_RESIDUAL_BLOCKS="${HSI_RESIDUAL_BLOCKS}"
  export MMDIFF_HSI_CONV_HIDDEN="${HSI_CONV_HIDDEN}"
  export MMDIFF_HSI_AGG_MODE="${HSI_AGG_MODE}"
  export MMDIFF_LIDAR_HIDDEN="${LIDAR_HIDDEN}"
  export MMDIFF_LIDAR_EXTRA_BLOCKS="${LIDAR_EXTRA_BLOCKS}"
  export MMDIFF_CLS_TRANSFORMER_LAYERS="${CLS_TRANSFORMER_LAYERS}"
  export MMDIFF_CLS_TRANSFORMER_FF_DIM="${CLS_TRANSFORMER_FF_DIM}"
}

do_shutdown() {
  sleep 3
  local _i
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown
    sleep 3
  done
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}

run_case() {
  local seed="$1"
  local tag="$2"
  shift 2
  clear_case_env
  set_fixed_env
  export MMDIFF_RANDOM_SEED="${seed}"
  export MMDIFF_EXPERIMENT_TAG="hp_s${seed}_${tag}"
  export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
  while [ "$#" -gt 0 ]; do
    export "$1"
    shift
  done
  echo "===> ${MMDIFF_EXPERIMENT_TAG}"
  python main.py
}

main() {
  local seed item tag wg wc
  echo "========== 仅 Loss 权重消融（全局=1-中心） =========="
  for seed in "${SEEDS[@]}"; do
    echo "==================== seed=${seed} ===================="
    for item in "${LOSS_CASES[@]}"; do
      IFS="|" read -r tag wg wc <<< "${item}"
      run_case "${seed}" "${tag}" \
        "MMDIFF_LOSS_WEIGHT_GLOBAL=${wg}" \
        "MMDIFF_LOSS_WEIGHT_CENTER=${wc}"
    done
  done
}

main
do_shutdown
