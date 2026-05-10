#!/usr/bin/env bash

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

# 实验标记与种子
export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-distance_trial}"
export MMDIFF_RANDOM_SEED="${MMDIFF_RANDOM_SEED:-42}"
export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"

# 三模态融合（HSI/RGB/LiDAR）

export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"

export MMDIFF_CLS_TOKEN_DIM="${MMDIFF_CLS_TOKEN_DIM:-192}"
export MMDIFF_CLS_HEAD_HIDDEN="${MMDIFF_CLS_HEAD_HIDDEN:-192}"
export MMDIFF_CLS_TRANSFORMER_LAYERS="${MMDIFF_CLS_TRANSFORMER_LAYERS:-1}"
export MMDIFF_CLS_TRANSFORMER_FF_DIM="${MMDIFF_CLS_TRANSFORMER_FF_DIM:-384}"

export MMDIFF_CENTER_DISTANCE_BIAS_ALPHA="${MMDIFF_CENTER_DISTANCE_BIAS_ALPHA:-0.2}"

# encoder / HSI branch（可调；与 spatial 并行使用，不改变融合用 121 格 memory）
export MMDIFF_HSI_SE_RATIO="${MMDIFF_HSI_SE_RATIO:-8}"
export MMDIFF_HSI_RESIDUAL_BLOCKS="${MMDIFF_HSI_RESIDUAL_BLOCKS:-2}"
export MMDIFF_HSI_CONV_HIDDEN="${MMDIFF_HSI_CONV_HIDDEN:-64}"
export MMDIFF_HSI_AGG_MODE="${MMDIFF_HSI_AGG_MODE:-multi_token}"

export MMDIFF_LIDAR_HIDDEN="${MMDIFF_LIDAR_HIDDEN:-16}"
export MMDIFF_LIDAR_EXTRA_BLOCKS="${MMDIFF_LIDAR_EXTRA_BLOCKS:-0}"

export MMDIFF_RGB_STUDENT_CHECKPOINT="${MMDIFF_RGB_STUDENT_CHECKPOINT:-$RGB_STUDENT_CKPT}"
export MMDIFF_FREEZE_RGB_STUDENT="${MMDIFF_FREEZE_RGB_STUDENT:-0}"

export MMDIFF_LOSS_WEIGHT_GLOBAL="${-0.2}"
export MMDIFF_EXPERIMENT_TAG="g${WG/./}"
python main.py
do_shutdown
