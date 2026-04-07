#!/usr/bin/env bash
# 主实验网格：HSI SE + LiDAR 基线宽；不含 base / lidar_wide。
# SE 默认 24；exp3 略增 wd；exp4 更大 classifier（320/160）+ 再略增 wd。
# 须使用 Unix 换行（LF）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
RUN_SH="$ROOT/run.sh"

: "${MMDIFF_GRID_SE:=24}"
: "${MMDIFF_COMBO_LIDAR_BASE:=64}"
: "${MMDIFF_RGB_STUDENT_CHECKPOINT:=$ROOT/model/rgb_student_distill.pt}"
# exp3 / exp4 的 weight_decay（勿过大；可按需 export 覆盖）
: "${MMDIFF_GRID_WD_EXP3:=1.5e-4}"
: "${MMDIFF_GRID_WD_EXP4:=2e-4}"

export_for_train() {
  export MMDIFF_RGB_SOURCE=student
  export MMDIFF_RGB_STUDENT_CHECKPOINT
  export MMDIFF_FREEZE_RGB_STUDENT=0
  export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
}

apply_exp() {
  case "$1" in
    exp1)
      export MMDIFF_HSI_SE_RATIO="$MMDIFF_GRID_SE"
      export MMDIFF_LIDAR_HIDDEN="$MMDIFF_COMBO_LIDAR_BASE"
      export MMDIFF_LIDAR_EXTRA_BLOCKS=3
      export MMDIFF_WEIGHT_DECAY=1e-4
      unset MMDIFF_CLS_TOKEN_DIM
      unset MMDIFF_CLS_HEAD_HIDDEN
      ;;
    exp2)
      export MMDIFF_HSI_SE_RATIO="$MMDIFF_GRID_SE"
      export MMDIFF_LIDAR_HIDDEN="$MMDIFF_COMBO_LIDAR_BASE"
      export MMDIFF_LIDAR_EXTRA_BLOCKS=4
      export MMDIFF_WEIGHT_DECAY=1e-4
      unset MMDIFF_CLS_TOKEN_DIM
      unset MMDIFF_CLS_HEAD_HIDDEN
      ;;
    exp3)
      export MMDIFF_HSI_SE_RATIO="$MMDIFF_GRID_SE"
      export MMDIFF_LIDAR_HIDDEN="$MMDIFF_COMBO_LIDAR_BASE"
      export MMDIFF_LIDAR_EXTRA_BLOCKS=3
      export MMDIFF_WEIGHT_DECAY="$MMDIFF_GRID_WD_EXP3"
      unset MMDIFF_CLS_TOKEN_DIM
      unset MMDIFF_CLS_HEAD_HIDDEN
      ;;
    exp4)
      export MMDIFF_HSI_SE_RATIO="$MMDIFF_GRID_SE"
      export MMDIFF_LIDAR_HIDDEN="$MMDIFF_COMBO_LIDAR_BASE"
      export MMDIFF_LIDAR_EXTRA_BLOCKS=3
      export MMDIFF_WEIGHT_DECAY="$MMDIFF_GRID_WD_EXP4"
      export MMDIFF_CLS_TOKEN_DIM=320
      export MMDIFF_CLS_HEAD_HIDDEN=160
      ;;
    *)
      echo "未知 exp: $1（exp1|exp2|exp3|exp4）" >&2
      exit 1
      ;;
  esac
  export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-${MMDIFF_EXPERIMENT_TAG_PREFIX:-exp}_$1}"
}

run_train_exp() {
  export_for_train
  apply_exp "$1"
  exec python main.py
}

run_grid() {
  local ts="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
  local e failed=0
  for e in exp1 exp2 exp3 exp4; do
    MMDIFF_RUN_TIMESTAMP="$ts" MMDIFF_EXPERIMENT_TAG="" bash "$RUN_SH" train_exp "$e" || failed=$((failed + 1))
  done
  exit "$failed"
}

do_shutdown() {
  sleep 3
  local _i
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown 2>/dev/null || true
    sleep 3
  done
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}

usage() {
  cat <<'EOF'
用法: bash run.sh <子命令>

  train_exp <exp1|exp2|exp3|exp4>   单组实验
  grid                              串行 exp1→exp4
  all                               等同 grid，结束可配合 MMDIFF_SHUTDOWN_AT_END
  sanity [args...]

网格摘要:
  exp1  HSI_SE=24 lidar=64 blk=3 wd=1e-4 cls=默认
  exp2  同 exp1 但 LiDAR 残差块=4（轻微加深）
  exp3  同 exp1 结构 wd=MMDIFF_GRID_WD_EXP3（默认 1.5e-4）
  exp4  同 exp3  cls token=320 head=160 wd=MMDIFF_GRID_WD_EXP4（默认 2e-4）

可调: MMDIFF_GRID_SE MMDIFF_COMBO_LIDAR_BASE MMDIFF_GRID_WD_EXP3 MMDIFF_GRID_WD_EXP4
      MMDIFF_RGB_STUDENT_CHECKPOINT

若 Linux 报 pipefail: sed -i 's/\r$//' run.sh 或 dos2unix run.sh
EOF
}

[ $# -eq 0 ] && set -- grid

case "${1:-}" in
  help|-h|--help) usage ;;
  train_exp) run_train_exp "${2:-exp1}" ;;
  grid) run_grid ;;
  all)
    ts="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
    f=0
    MMDIFF_RUN_TIMESTAMP="$ts" MMDIFF_EXPERIMENT_TAG="" bash "$RUN_SH" grid || f=$((f + 1))
    if [ "${MMDIFF_SHUTDOWN_AT_END:-0}" = 1 ]; then
      sleep 2
      /usr/bin/shutdown 2>/dev/null || true
    fi
    do_shutdown
    exit "$f"
    ;;
  sanity) python utils/hsi_branch_sanity.py "${@:2}" ;;
  # 兼容旧入口（已废弃 combo）
  train_combo)
    echo "train_combo 已改为 train_exp；请: bash run.sh train_exp ${2:-exp1}" >&2
    exit 1
    ;;
  *) echo "未知: $1" >&2; usage >&2; exit 1 ;;
esac
