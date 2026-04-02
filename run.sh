#!/usr/bin/env bash
# 用法见文末「run.sh help」；无参数 = 主训练（BS64 / SupCon / gate_before_pool）+ 关机。
# TB/checkpoint 目录默认短名：{时间戳}_e{NN}_lr{标签}（见 logging_utils.prepare_tb_run_dir）；长目录名：MMDIFF_TB_LONG_TAG=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

_safe_combo() {
  local s="${1:-}"
  echo "${s//[^a-zA-Z0-9_-]/_}"
}

setup_common() {
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-200}"
  export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"
  export MMDIFF_HSI_RESIDUAL_BLOCKS=6
  export MMDIFF_HSI_CONV_HIDDEN=96
  export MMDIFF_HSI_SE_RATIO=16
}

usage() {
  cat <<'EOF'
用法: bash run.sh [子命令]

  无参数     主训练 exp1（BS64 / SupCon / gate_before_pool），结束后关机（与旧版一致）
  exp1       BS64 + SupCon（与无参数相同，不关机）
  exp2       BS512 + wd=2e-4 + lr=1e-4（调度同 param 默认 piecewise）
  exp3       BS512 + wd=2e-4 + lr=2e-3（高 lr 对照）
  sanity     HSI 分支自检（秒级，需时手动跑）；可传参，如: bash run.sh sanity --batch 4
  all        串行 exp1 → exp2 → exp3（不含 sanity）；失败不中断；终端+日志双写
  help       本说明

  可选环境变量: MMDIFF_*（见 param.py）、MMDIFF_OVERNIGHT_LOG、MMDIFF_SHUTDOWN_AT_END=1（all 结束后关机）
  余弦调度: MMDIFF_SCHEDULER_NAME=cosine（默认 piecewise，与 best_model.log 一致）

示例:
  bash run.sh all
  MMDIFF_OVERNIGHT_LOG=./my.log bash run.sh all
EOF
}

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-multimodal}"
COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"
SC="$(_safe_combo "$COMBO")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_step() {
  local name="$1"
  shift
  log "========== START $name =========="
  "$@"
  local ec=$?
  if [ "$ec" -eq 0 ]; then
    log "========== END $name OK =========="
  else
    log "========== END $name FAILED (exit $ec) =========="
  fi
  return "$ec"
}

do_shutdown() {
  sleep 3
  local _i
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown 2>/dev/null || true
    sleep 3
  done
}

# 单次训练：统一时间戳（便于 overnight 流水线对齐）；日志目录见 prepare_tb_run_dir
_export_run_tag_exp1() {
  export MMDIFF_USE_SUPCON="${MMDIFF_USE_SUPCON:-1}"
  export MMDIFF_BATCH_SIZE=64
  export MMDIFF_EXPERIMENT_NUM=1
  export MMDIFF_LR_TAG=6e-4
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool"
}

# ---------------------------------------------------------------------------
# 无参数：旧版 = exp1 + 关机
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
  setup_common
  export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
  _export_run_tag_exp1
  echo "run: ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG}"
  python main.py
  do_shutdown
  exit 0
fi

case "$1" in
  help|-h|--help)
    usage
    ;;
  exp1)
    setup_common
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    _export_run_tag_exp1
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  exp2)
    setup_common
    export MMDIFF_USE_SUPCON=1
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=2e-4
    export MMDIFF_LEARNING_RATE=1e-4
    export MMDIFF_EXPERIMENT_NUM=2
    export MMDIFF_LR_TAG=1e-4
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd2e4_gate_before_pool"
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  exp3)
    setup_common
    export MMDIFF_USE_SUPCON=1
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=2e-4
    export MMDIFF_LEARNING_RATE=2e-3
    export MMDIFF_EXPERIMENT_NUM=3
    export MMDIFF_LR_TAG=2e-3
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd2e4_gate_before_pool"
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  sanity)
    echo "=== utils/hsi_branch_sanity.py ==="
    python utils/hsi_branch_sanity.py "${@:2}"
    ;;
  all)
    LOG="${MMDIFF_OVERNIGHT_LOG:-$ROOT/overnight_$(date +%Y%m%d_%H%M%S).log}"
    PIPE_TS="$(date +%Y%m%d-%H%M%S)"
    (
      set +e
      log "pipeline | log: $LOG | PIPE_TS=$PIPE_TS (export MMDIFF_RUN_TIMESTAMP=$PIPE_TS for each step for aligned dirs)"
      FAILED=0
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step exp1 bash "$ROOT/run.sh" exp1 || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step exp2 bash "$ROOT/run.sh" exp2 || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step exp3 bash "$ROOT/run.sh" exp3 || FAILED=$((FAILED + 1))
      log "pipeline finished | failed_steps=$FAILED (0=all ok)"
      if [ "${MMDIFF_SHUTDOWN_AT_END:-0}" = "1" ]; then
        log "MMDIFF_SHUTDOWN_AT_END=1 -> shutdown"
        do_shutdown
      fi
      if [ "$FAILED" -gt 0 ]; then
        exit 1
      fi
      exit 0
    ) 2>&1 | tee "$LOG"
    ;;
  *)
    echo "未知子命令: $1" >&2
    usage >&2
    exit 1
    ;;
esac
