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

# 只保留三组超参数实验
PATCH_CASES=(
  "patch_p09|9"
  "patch_p11|11"
  "patch_p13|13"
)
LOSS_CASES=(
  "loss_g01_c09|0.1|0.9"
  "loss_g02_c08|0.2|0.8"
  "loss_g03_c07|0.3|0.7"
)
EMBED_CASES=(
  "embed_t256_h256|256|256"
  "embed_t320_h320|320|320"
  "embed_t384_h384|384|384"
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
  local seed item tag p wg wc td hh
  echo "========== 仅超参数实验：patch / loss / embed =========="
  for seed in "${SEEDS[@]}"; do
    echo "==================== seed=${seed} ===================="

    for item in "${PATCH_CASES[@]}"; do
      IFS="|" read -r tag p <<< "${item}"
      run_case "${seed}" "${tag}" "MMDIFF_PATCH_WINDOW_SIZE=${p}"
    done

    for item in "${LOSS_CASES[@]}"; do
      IFS="|" read -r tag wg wc <<< "${item}"
      run_case "${seed}" "${tag}" \
        "MMDIFF_LOSS_WEIGHT_GLOBAL=${wg}" \
        "MMDIFF_LOSS_WEIGHT_CENTER=${wc}"
    done

    for item in "${EMBED_CASES[@]}"; do
      IFS="|" read -r tag td hh <<< "${item}"
      run_case "${seed}" "${tag}" \
        "MMDIFF_CLS_TOKEN_DIM=${td}" \
        "MMDIFF_CLS_HEAD_HIDDEN=${hh}"
    done
  done
}

main
