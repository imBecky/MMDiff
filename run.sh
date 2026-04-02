#!/usr/bin/env bash
# 用法见文末「run.sh help」；无参数 = 主训练（BS64 / SupCon / lr=1e-3 / 余弦+warmup5% / gate_before_pool）+ 关机。
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
  # 主实验（exp1 / 无参）：lr_max=1e-3，余弦退火，warmup=总步数 5%（≈总轮数 5%）；exp5 在子命令内改为 piecewise
  export MMDIFF_SCHEDULER_NAME="${MMDIFF_SCHEDULER_NAME:-cosine}"
  export MMDIFF_LEARNING_RATE="${MMDIFF_LEARNING_RATE:-1e-3}"
  export MMDIFF_SCHED_COSINE_WARMUP_RATIO="${MMDIFF_SCHED_COSINE_WARMUP_RATIO:-0.05}"
}

usage() {
  cat <<'EOF'
用法: bash run.sh [子命令]

  无参数     主训练 exp1（BS64 / SupCon / lr=1e-3 / cosine+warmup5% / gate_before_pool），结束后关机
  exp1       同上，不关机
  exp2       BS64 + cosine+warmup5% + lr=1e-3 + wd=1e-4 + SupCon=OFF（与 exp1 对照 SupCon）
  exp3       BS512 + cosine+warmup5% + lr=4e-3 + wd=5e-4 + SupCon=OFF（大 batch + 较高 WD）
  exp4       BS512 + cosine+warmup5% + lr=4e-3 + wd=2e-4 + SupCon=OFF（与 exp3 仅 WD 不同）
  exp5       BS64 + piecewise + lr=6e-4 + wd=1e-4 + SupCon=OFF（贴近 best_model.log，无 SupCon）
  sanity     HSI 分支自检（秒级，需时手动跑）；可传参，如: bash run.sh sanity --batch 4
  all        串行 exp1 → exp2 → exp3 → exp4 → exp5（不含 sanity）；失败不中断；终端+日志双写
  help       本说明

  可选环境变量: MMDIFF_*（见 param.py）、MMDIFF_OVERNIGHT_LOG、MMDIFF_SHUTDOWN_AT_END=1（all 结束后关机）
  exp1 默认: MMDIFF_SCHEDULER_NAME=cosine, MMDIFF_LEARNING_RATE=1e-3, MMDIFF_SCHED_COSINE_WARMUP_RATIO=0.05

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
  export MMDIFF_LR_TAG=1e-3
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool_cos1e3_w5"
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
    export MMDIFF_USE_SUPCON=0
    export MMDIFF_BATCH_SIZE=64
    export MMDIFF_WEIGHT_DECAY=1e-4
    export MMDIFF_LEARNING_RATE=1e-3
    export MMDIFF_EXPERIMENT_NUM=2
    export MMDIFF_LR_TAG=1e-3
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool_cos1e3_w5_nosupcon"
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  exp3)
    setup_common
    export MMDIFF_USE_SUPCON=0
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=5e-4
    export MMDIFF_LEARNING_RATE=4e-3
    export MMDIFF_EXPERIMENT_NUM=3
    export MMDIFF_LR_TAG=4e-3
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd5e4_cos4e3_w5_nosupcon"
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  exp4)
    setup_common
    export MMDIFF_USE_SUPCON=0
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=2e-4
    export MMDIFF_LEARNING_RATE=4e-3
    export MMDIFF_EXPERIMENT_NUM=4
    export MMDIFF_LR_TAG=4e-3
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd2e4_cos4e3_w5_nosupcon"
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} | exp=${MMDIFF_EXPERIMENT_NUM} lr=${MMDIFF_LR_TAG} ==="
    python main.py
    ;;
  exp5)
    setup_common
    export MMDIFF_SCHEDULER_NAME=piecewise_two_step
    export MMDIFF_USE_SUPCON=0
    export MMDIFF_BATCH_SIZE=64
    export MMDIFF_WEIGHT_DECAY=1e-4
    export MMDIFF_LEARNING_RATE=6e-4
    export MMDIFF_EXPERIMENT_NUM=5
    export MMDIFF_LR_TAG=6e-4
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool_pw6e4_nosupcon"
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
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step exp4 bash "$ROOT/run.sh" exp4 || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step exp5 bash "$ROOT/run.sh" exp5 || FAILED=$((FAILED + 1))
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
