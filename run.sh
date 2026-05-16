#!/usr/bin/env bash
# 串行四组训练：`MMDIFF_EXPERIMENT_TAG` + `MMDIFF_RUN_TIMESTAMP` 区分 TB。
# （1）不压缩 memory，增大 cross-attention 的 query token 数；
# （2）linear：121→K 可学习线性聚合，几何通过距离矩阵 softmax 聚合保留（无 center memory token，tag …_nct）；
# （3）grid：每模态 G×G token，仍为规则空间栅格（较 none121 快许多，`…_nct`）；
# （4）latent：K 个 query + MHA 自 121 空间 token 聚合，另保留 center patch（`MMDIFF_MEMORY_KEEP_CENTER_TOKEN=1`，tag …_ct）；完成后关机（内含 webhook）。

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
export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"

unset MMDIFF_COUPLING_HIDDEN_FACTOR
unset MMDIFF_GLOBAL_ANTICENTER_BIAS
unset MMDIFF_CLS_HEAD_LAYERS
unset MMDIFF_GLOBAL_QUERY_TOKENS
unset MMDIFF_CENTER_QUERY_TOKENS
unset MMDIFF_MODALITY_EMBED
unset MMDIFF_DISTANCE_BIAS_HSI_ONLY

# tag 前缀里的 tau 码；与 param.CENTER_DISTANCE_BIAS_TAU 默认 2.0 对齐（可事先 export 覆盖）
_bias_tau="${MMDIFF_CENTER_DISTANCE_BIAS_TAU:-2.0}"
_bias_tdot="${_bias_tau//./}"

_cleanup_memory_case() {
  unset MMDIFF_MEMORY_COMPRESS_MODE MMDIFF_MEMORY_GRID_SIZE MMDIFF_MEMORY_COMPRESS_TOKENS MMDIFF_MEMORY_KEEP_CENTER_TOKEN
}

_cleanup_query_case() {
  unset MMDIFF_GLOBAL_QUERY_TOKENS MMDIFF_CENTER_QUERY_TOKENS
}

_run_one() {
  local _suffix=$1
  export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
  export MMDIFF_EXPERIMENT_TAG="a36_mdemb_hsibias_t${_bias_tdot}_${_suffix}"
  python main.py
  unset MMDIFF_EXPERIMENT_TAG
  _cleanup_memory_case
}

# --- ① none：121 空间 token ---
_cleanup_memory_case
_cleanup_query_case
export MMDIFF_MEMORY_COMPRESS_MODE=none
unset MMDIFF_MEMORY_GRID_SIZE MMDIFF_MEMORY_COMPRESS_TOKENS MMDIFF_MEMORY_KEEP_CENTER_TOKEN
export MMDIFF_GLOBAL_QUERY_TOKENS="${MMDIFF_GLOBAL_QUERY_TOKENS:-4}"
export MMDIFF_CENTER_QUERY_TOKENS="${MMDIFF_CENTER_QUERY_TOKENS:-4}"
_run_one "none121_gl${MMDIFF_GLOBAL_QUERY_TOKENS}_ce${MMDIFF_CENTER_QUERY_TOKENS}"
_cleanup_query_case

# --- ② linear：K 维压缩，无 center-token；query 回归默认 ---
_cleanup_memory_case
_cleanup_query_case
export MMDIFF_MEMORY_COMPRESS_MODE=linear
export MMDIFF_MEMORY_COMPRESS_TOKENS="${MMDIFF_MEMORY_COMPRESS_TOKENS:-16}"
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=0
unset MMDIFF_MEMORY_GRID_SIZE
_run_one "linear${MMDIFF_MEMORY_COMPRESS_TOKENS}_nct"
_cleanup_memory_case

# --- ③ grid：每分支空间 token 收窄为 G×G，位置保持为规则网格（优于「丢空间」的全局 pooling）---
_cleanup_memory_case
_cleanup_query_case
export MMDIFF_MEMORY_COMPRESS_MODE=grid
export MMDIFF_MEMORY_GRID_SIZE="${MMDIFF_MEMORY_GRID_SIZE:-4}"
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=0
unset MMDIFF_MEMORY_COMPRESS_TOKENS
_run_one "grid${MMDIFF_MEMORY_GRID_SIZE}_nct"
_cleanup_memory_case

# --- ④ latent：K token MHA 压缩 + center memory token（几何距离行与 linear/latent 的 dist logits 对齐）---
_cleanup_memory_case
_cleanup_query_case
export MMDIFF_MEMORY_COMPRESS_MODE=latent
export MMDIFF_MEMORY_COMPRESS_TOKENS="${MMDIFF_MEMORY_COMPRESS_TOKENS:-16}"
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=1
unset MMDIFF_MEMORY_GRID_SIZE
_run_one "latent${MMDIFF_MEMORY_COMPRESS_TOKENS}_ct"
_cleanup_memory_case

do_shutdown