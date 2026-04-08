#!/usr/bin/env bash
# RGB(student)→LiDAR FiLM 消融：设置 MMDIFF_RGB_TO_LIDAR_GUIDANCE=film 后调用 run.sh train_exp。
# 须 rgb_source=student（由 run.sh export_for_train）且模态含 rgb+lidar。
# 须使用 Unix 换行（LF）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
RUN_SH="$ROOT/run.sh"
FUS_SH="$ROOT/fus.sh"

: "${MMDIFF_FUS_EXP:=exp1}"

usage() {
  cat <<'EOF'
用法: bash fus.sh <子命令>

  train_fus [exp1|exp2|exp3|exp4]
      启用 RGB→LiDAR FiLM（film），再调用 run.sh train_exp；参数默认 MMDIFF_FUS_EXP（默认 exp1）。

  grid [exp1|exp2|exp3|exp4]
      与 train_fus 相同（本脚本仅一种引导，保留 grid 便于与旧习惯一致）。

  all [exp1|exp2|exp3|exp4]
      等同 grid；可选 MMDIFF_SHUTDOWN_AT_END=1 后尝试关机，并发送与 run.sh all 相同的通知。

环境变量:
  MMDIFF_RGB_TO_LIDAR_GUIDANCE  默认由本脚本设为 film；手动可设为 none 关闭
  MMDIFF_FUS_EXP                默认 exp1
  MMDIFF_EXPERIMENT_TAG_PREFIX  默认 fus_rgb2l；tag 形如 fus_rgb2l_film_exp1_mmdd-hhmm

若 Linux 报 pipefail: sed -i 's/\r$//' fus.sh 或 dos2unix fus.sh
EOF
}

run_train_fus() {
  local exp="${1:-$MMDIFF_FUS_EXP}"
  export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
  export MMDIFF_RGB_TO_LIDAR_GUIDANCE=film
  export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG_PREFIX:-fus_rgb2l}_film_${exp}_${MMDIFF_RUN_TIMESTAMP}"
  bash "$RUN_SH" train_exp "$exp"
}

[ $# -eq 0 ] && set -- train_fus

case "${1:-}" in
  help|-h|--help) usage ;;
  train_fus) run_train_fus "${2:-}" ;;
  grid) run_train_fus "${2:-}" ;;
  all)
    ts="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"
    f=0
    MMDIFF_RUN_TIMESTAMP="$ts" MMDIFF_EXPERIMENT_TAG="" bash "$FUS_SH" train_fus "${2:-}" || f=$((f + 1))
    if [ "${MMDIFF_SHUTDOWN_AT_END:-0}" = 1 ]; then
      sleep 2
      /usr/bin/shutdown 2>/dev/null || true
    fi
    sleep 3
    curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=fus.sh_rgb2lidar已完成channel=9" 2>/dev/null || true
    exit "$f"
    ;;
  *) echo "未知: $1" >&2; usage >&2; exit 1 ;;
esac
