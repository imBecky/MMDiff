#!/usr/bin/env bash
# 无人值守训练：通知 + 关机。训练数值默认来自 param.py；此处仅 TB 命名与环境清理。
#
# Cross-attention 中心距离 bias 固定为 alpha*exp(-dist/tau)，见 model/multimodal.MultimodalClassifier._build_cross_attn_logit_bias

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

export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"

unset MMDIFF_COUPLING_HIDDEN_FACTOR
unset MMDIFF_GLOBAL_ANTICENTER_BIAS
unset MMDIFF_CLS_HEAD_LAYERS
unset MMDIFF_GLOBAL_QUERY_TOKENS
unset MMDIFF_CENTER_QUERY_TOKENS

export MMDIFF_MEMORY_COMPRESS_MODE=none
unset MMDIFF_MEMORY_GRID_SIZE MMDIFF_MEMORY_COMPRESS_TOKENS MMDIFF_MEMORY_KEEP_CENTER_TOKEN

# tag 后缀仅用于日志命名；默认与 param.CENTER_DISTANCE_BIAS_TAU=2.0 一致（可通过环境预先覆盖以供展示）
_bias_tau="${MMDIFF_CENTER_DISTANCE_BIAS_TAU:-2.0}"
_bias_tdot="${_bias_tau//./}"

export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
export MMDIFF_EXPERIMENT_TAG="a35_expDist_t${_bias_tdot}_resTrue"

python main.py

unset MMDIFF_EXPERIMENT_TAG

do_shutdown
