#!/usr/bin/env bash
# 用法见文末「run.sh help」；无参数 = 旧版行为（主训练 + 关机）。
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

  无参数     主训练（BS64 / SupCon / gate_before_pool），结束后执行关机（与旧版一致）
  baseline   同上主训练，不关机
  exp1       关 SupCon，BS64
  exp2       BS512 + weight_decay=2e-4
  exp3       BS512 + wd=2e-4 + 线性 LR=4.8e-3
  sanity     HSI 分支自检（秒级）；可传参，如: bash run.sh sanity --batch 4
  all        串行 sanity → baseline → exp1 → exp2 → exp3；失败不中断；终端+日志双写
  help       本说明

  可选环境变量: MMDIFF_*（见 param.py）、MMDIFF_OVERNIGHT_LOG（汇总日志路径）、
  MMDIFF_SHUTDOWN_AT_END=1（仅 bash run.sh all 全部结束后关机）

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

# ---------------------------------------------------------------------------
# 无参数：旧版 = 主训练 + 关机
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
  setup_common
  export MMDIFF_USE_SUPCON="${MMDIFF_USE_SUPCON:-1}"
  export MMDIFF_BATCH_SIZE=64
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool"
  echo "run: ${MMDIFF_EXPERIMENT_TAG}"
  python main.py
  # curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  do_shutdown
  # curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
  exit 0
fi

case "$1" in
  help|-h|--help)
    usage
    ;;
  baseline)
    setup_common
    export MMDIFF_USE_SUPCON=1
    export MMDIFF_BATCH_SIZE=64
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} ==="
    python main.py
    ;;
  exp1)
    setup_common
    export MMDIFF_USE_SUPCON=0
    export MMDIFF_BATCH_SIZE=64
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS64_gate_before_pool_nosupcon"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} ==="
    python main.py
    ;;
  exp2)
    setup_common
    export MMDIFF_USE_SUPCON=1
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=2e-4
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd2e4_gate_before_pool"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} ==="
    python main.py
    ;;
  exp3)
    setup_common
    export MMDIFF_USE_SUPCON=1
    export MMDIFF_BATCH_SIZE=512
    export MMDIFF_WEIGHT_DECAY=2e-4
    export MMDIFF_LEARNING_RATE=4.8e-3
    export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${SC}_B6_H96_SE16_BS512_wd2e4_lr4p8e3_gate_before_pool"
    echo "=== ${MMDIFF_EXPERIMENT_TAG} ==="
    python main.py
    ;;
  sanity)
    echo "=== utils/hsi_branch_sanity.py ==="
    python utils/hsi_branch_sanity.py "${@:2}"
    ;;
  all)
    LOG="${MMDIFF_OVERNIGHT_LOG:-$ROOT/overnight_$(date +%Y%m%d_%H%M%S).log}"
    (
      set +e
      log "pipeline | log: $LOG"
      FAILED=0
      run_step sanity bash "$ROOT/run.sh" sanity || FAILED=$((FAILED + 1))
      run_step baseline bash "$ROOT/run.sh" baseline || FAILED=$((FAILED + 1))
      run_step exp1 bash "$ROOT/run.sh" exp1 || FAILED=$((FAILED + 1))
      run_step exp2 bash "$ROOT/run.sh" exp2 || FAILED=$((FAILED + 1))
      run_step exp3 bash "$ROOT/run.sh" exp3 || FAILED=$((FAILED + 1))
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
