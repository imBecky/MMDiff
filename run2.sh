#!/usr/bin/env bash
# Memory 压缩实验 — with center-token；末尾通知 + 关机（适合无人值守主机）
# 与 run1.sh 并行时：仅本脚本关机，run1 不关机。

set -euo pipefail

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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MMDIFF_RANDOM_SEED="${MMDIFF_RANDOM_SEED:-42}"

export MMDIFF_CLS_TOKEN_DIM="${MMDIFF_CLS_TOKEN_DIM:-192}"
export MMDIFF_CLS_HEAD_HIDDEN="${MMDIFF_CLS_HEAD_HIDDEN:-192}"
export MMDIFF_CLS_TRANSFORMER_LAYERS="${MMDIFF_CLS_TRANSFORMER_LAYERS:-1}"
export MMDIFF_CLS_TRANSFORMER_FF_DIM="${MMDIFF_CLS_TRANSFORMER_FF_DIM:-384}"

export MMDIFF_HSI_SE_RATIO="${MMDIFF_HSI_SE_RATIO:-8}"
export MMDIFF_HSI_RESIDUAL_BLOCKS="${MMDIFF_HSI_RESIDUAL_BLOCKS:-2}"
export MMDIFF_HSI_CONV_HIDDEN="${MMDIFF_HSI_CONV_HIDDEN:-64}"
export MMDIFF_HSI_AGG_MODE="${MMDIFF_HSI_AGG_MODE:-multi_token}"

export MMDIFF_LIDAR_HIDDEN="${MMDIFF_LIDAR_HIDDEN:-16}"
export MMDIFF_LIDAR_EXTRA_BLOCKS="${MMDIFF_LIDAR_EXTRA_BLOCKS:-0}"

export MMDIFF_RGB_STUDENT_CHECKPOINT="${MMDIFF_RGB_STUDENT_CHECKPOINT:-${RGB_STUDENT_CKPT:-}}"
export MMDIFF_FREEZE_RGB_STUDENT="${MMDIFF_FREEZE_RGB_STUDENT:-0}"

export MMDIFF_LOSS_WEIGHT_GLOBAL="${MMDIFF_LOSS_WEIGHT_GLOBAL:-0.25}"
export MMDIFF_MODALITY_COMBO="hsi+rgb+lidar"

export MMDIFF_CENTER_DISTANCE_BIAS_ALPHA=3.5
_bias_tau="${MMDIFF_CENTER_DISTANCE_BIAS_TAU:-2.0}"
export MMDIFF_CENTER_DISTANCE_BIAS_TAU="$_bias_tau"
_bias_tdot="${_bias_tau//./}"

export MMDIFF_COUPLING_HIDDEN_FACTOR=2

_cleanup_memory_env() {
  unset MMDIFF_MEMORY_COMPRESS_MODE MMDIFF_MEMORY_GRID_SIZE MMDIFF_MEMORY_COMPRESS_TOKENS \
    MMDIFF_MEMORY_KEEP_CENTER_TOKEN MMDIFF_EXPERIMENT_TAG MMDIFF_RUN_TIMESTAMP
}

# 1) grid 4×4 + ct
_cleanup_memory_env
export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
export MMDIFF_EXPERIMENT_TAG="a35_chf2_grid4ct_t${_bias_tdot}"
export MMDIFF_MEMORY_COMPRESS_MODE=grid
export MMDIFF_MEMORY_GRID_SIZE=4
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=1
python main.py
_cleanup_memory_env

# 2) linear 16 + ct
export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
export MMDIFF_EXPERIMENT_TAG="a35_chf2_linear16ct_t${_bias_tdot}"
export MMDIFF_MEMORY_COMPRESS_MODE=linear
export MMDIFF_MEMORY_COMPRESS_TOKENS=16
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=1
python main.py
_cleanup_memory_env

# 3) latent 16 + ct
export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
export MMDIFF_EXPERIMENT_TAG="a35_chf2_latent16ct_t${_bias_tdot}"
export MMDIFF_MEMORY_COMPRESS_MODE=latent
export MMDIFF_MEMORY_COMPRESS_TOKENS=16
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=1
python main.py
_cleanup_memory_env

do_shutdown
