#!/usr/bin/env bash
# g6 超参 **第一轮（粗网格）**：**λ × α** 一起扫。
# Memory：grid G=6，无 center memory token。
# 默认：λ ∈ {0.25, 0.30, 0.35}，α ∈ {2, 2.5, 3} → 共 9 跑，tag `g6_l030_a25` 等。
# 覆盖：`export MMDIFF_LAMBDA_SWEEP="..."` / `export MMDIFF_ALPHA_SWEEP="..."`。
#
set -euo pipefail

do_shutdown() {
  sleep 10
  local _i
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown
    sleep 3
  done
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# ---------------------------------------------------------------------------
# 【重要 · 勿删】论文级可复现 — 改 run.sh 时务必保留下面三行 export
# - MMDIFF_RANDOM_SEED      → param.RANDOM_SEED / runner 固定 torch/np/sklearn 划分
# - PYTHONHASHSEED=0        → 固定 dict/set 迭代顺序（Python 3.3+ 默认会随机化哈希）
# - CUBLAS_WORKSPACE_CONFIG → Ampere+ cuBLAS GEMM（Linear/Attention）须在进程启动前固定
# 详见 agent-collab/README.md「可复现性（随机种子）」
# ---------------------------------------------------------------------------
export MMDIFF_RANDOM_SEED="${MMDIFF_RANDOM_SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"

unset MMDIFF_COUPLING_HIDDEN_FACTOR
unset MMDIFF_GLOBAL_ANTICENTER_BIAS
unset MMDIFF_CLS_HEAD_LAYERS
unset MMDIFF_GLOBAL_QUERY_TOKENS
unset MMDIFF_CENTER_QUERY_TOKENS
unset MMDIFF_MODALITY_EMBED
unset MMDIFF_DISTANCE_BIAS_HSI_ONLY
unset MMDIFF_CENTER_DISTANCE_BIAS_ALPHA
unset MMDIFF_LOSS_WEIGHT_GLOBAL

export MMDIFF_TB_SIMPLE_RUN_DIR="${MMDIFF_TB_SIMPLE_RUN_DIR:-1}"

_run_one() {
  local _suffix=$1
  export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
  export MMDIFF_EXPERIMENT_TAG="${_suffix}"
  python main.py
  unset MMDIFF_EXPERIMENT_TAG
}

_g="${MMDIFF_MEMORY_GRID_SIZE:-6}"

export MMDIFF_MEMORY_COMPRESS_MODE=grid
export MMDIFF_MEMORY_GRID_SIZE="${_g}"
export MMDIFF_MEMORY_KEEP_CENTER_TOKEN=0
unset MMDIFF_MEMORY_COMPRESS_TOKENS

# shellcheck disable=SC2086
for _lam in ${MMDIFF_LAMBDA_SWEEP:-0.25 0.30 0.35}; do
  export MMDIFF_LOSS_WEIGHT_GLOBAL="${_lam}"
  _ldot="${_lam//./}"
  for _alpha in ${MMDIFF_ALPHA_SWEEP:-2 2.5 3}; do
    export MMDIFF_CENTER_DISTANCE_BIAS_ALPHA="${_alpha}"
    _adot="${_alpha//./}"
    _run_one "g${_g}_l${_ldot}_a${_adot}"
    unset MMDIFF_CENTER_DISTANCE_BIAS_ALPHA
  done
  unset MMDIFF_LOSS_WEIGHT_GLOBAL
done

unset MMDIFF_MEMORY_COMPRESS_MODE MMDIFF_MEMORY_GRID_SIZE MMDIFF_MEMORY_KEEP_CENTER_TOKEN

do_shutdown
