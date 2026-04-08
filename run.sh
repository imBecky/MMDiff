#!/usr/bin/env bash
# 主实验：强 HSI SE + 较大分类头；骨干刻意变浅（HSI/LiDAR/融合），其余 lr/wd 等见 param 默认。
# 须使用 Unix 换行（LF）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
RUN_SH="$ROOT/run.sh"

# --- 结构：SE + 大分类头（可 export 覆盖）---
: "${MMDIFF_GRID_SE:=24}"
: "${MMDIFF_CLS_TOKEN_DIM:=320}"
: "${MMDIFF_CLS_HEAD_HIDDEN:=160}"

# --- 骨干变浅（可 export 覆盖）---
: "${MMDIFF_HSI_RESIDUAL_BLOCKS:=3}"
: "${MMDIFF_HSI_CONV_HIDDEN:=64}"
: "${MMDIFF_HSI_AGG_MODE:=mean}"
: "${MMDIFF_LIDAR_HIDDEN:=48}"
: "${MMDIFF_LIDAR_EXTRA_BLOCKS:=1}"
: "${MMDIFF_CLS_TRANSFORMER_LAYERS:=1}"
: "${MMDIFF_CLS_TRANSFORMER_FF_DIM:=384}"

: "${MMDIFF_RGB_STUDENT_CHECKPOINT:=$ROOT/model/rgb_student_distill.pt}"

export_for_train() {
  # 容器里常见 OMP_NUM_THREADS=空串，libgomp 会报错；:- 在空串时也会用默认值
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MMDIFF_RGB_SOURCE=student
  export MMDIFF_RGB_STUDENT_CHECKPOINT
  export MMDIFF_FREEZE_RGB_STUDENT=0
  export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
}

run_train() {
  export_for_train
  export MMDIFF_HSI_SE_RATIO="$MMDIFF_GRID_SE"
  export MMDIFF_CLS_TOKEN_DIM
  export MMDIFF_CLS_HEAD_HIDDEN
  export MMDIFF_HSI_RESIDUAL_BLOCKS
  export MMDIFF_HSI_CONV_HIDDEN
  export MMDIFF_HSI_AGG_MODE
  export MMDIFF_LIDAR_HIDDEN
  export MMDIFF_LIDAR_EXTRA_BLOCKS
  export MMDIFF_CLS_TRANSFORMER_LAYERS
  export MMDIFF_CLS_TRANSFORMER_FF_DIM
  export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-${MMDIFF_EXPERIMENT_TAG_PREFIX:-exp}_se_cls_shallow}"
  exec python main.py
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

  train     浅骨干 + SE + 大分类头（见脚本顶部 MMDIFF_* 默认值）
  grid      同 train
  all       train 结束后 do_shutdown
  sanity [args...]

默认变浅：HSI 残差块 3、conv_hidden 64、agg=mean；LiDAR hidden 48、extra_blocks=1；
融合 Decoder 层数 1、ff_dim=384。SE 与 cls token/head 仍由 MMDIFF_GRID_SE / MMDIFF_CLS_* 控制。

若 Linux 报 pipefail: sed -i 's/\r$//' run.sh 或 dos2unix run.sh
EOF
}

[ $# -eq 0 ] && set -- train

case "${1:-}" in
  help|-h|--help) usage ;;
  train) run_train ;;
  grid)
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
    unset MMDIFF_EXPERIMENT_TAG 2>/dev/null || true
    run_train
    ;;
  all)
    ts="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
    f=0
    MMDIFF_RUN_TIMESTAMP="$ts" bash "$RUN_SH" train || f=$((f + 1))
    if [ "${MMDIFF_SHUTDOWN_AT_END:-0}" = 1 ]; then
      sleep 2
      /usr/bin/shutdown 2>/dev/null || true
    fi
    do_shutdown
    exit "$f"
    ;;
  sanity) python utils/hsi_branch_sanity.py "${@:2}" ;;
  train_exp|train_combo)
    echo "已简化为单组 train；请: bash run.sh train" >&2
    exit 1
    ;;
  *) echo "未知: $1" >&2; usage >&2; exit 1 ;;
esac
