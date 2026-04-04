#!/usr/bin/env bash
# 轻量 RGB 消融：随机初始化 / 蒸馏权重+冻结 / 蒸馏权重+微调
# 无参数 = all：precompute → ablate_all（内含 distill + 三种主训练，仅终端输出）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

_safe_combo() {
  local s="${1:-}"
  echo "${s//[^a-zA-Z0-9_-]/_}"
}

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-multimodal}"
COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"
SC="$(_safe_combo "$COMBO")"

RGB_STUDENT_CKPT="${MMDIFF_RGB_STUDENT_CHECKPOINT:-$ROOT/rgb_student_distill.pt}"

setup_common() {
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-200}"
  export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"
  export MMDIFF_HSI_RESIDUAL_BLOCKS="${MMDIFF_HSI_RESIDUAL_BLOCKS:-6}"
  export MMDIFF_HSI_CONV_HIDDEN="${MMDIFF_HSI_CONV_HIDDEN:-96}"
  export MMDIFF_HSI_SE_RATIO="${MMDIFF_HSI_SE_RATIO:-16}"
  export MMDIFF_SCHEDULER_NAME="${MMDIFF_SCHEDULER_NAME:-cosine}"
  export MMDIFF_LEARNING_RATE="${MMDIFF_LEARNING_RATE:-1e-3}"
  export MMDIFF_SCHED_COSINE_WARMUP_RATIO="${MMDIFF_SCHED_COSINE_WARMUP_RATIO:-0.05}"
  export MMDIFF_BATCH_SIZE="${MMDIFF_BATCH_SIZE:-64}"
  export MMDIFF_WEIGHT_DECAY="${MMDIFF_WEIGHT_DECAY:-1e-4}"
  export MMDIFF_USE_SUPCON="${MMDIFF_USE_SUPCON:-0}"
}

# 主训练：rgb_source 默认见 param；此处只设消融（random|freeze|finetune）
setup_rgb_ablate() {
  local a="${1:?}"
  setup_common
  export MMDIFF_EXPERIMENT_NUM="${MMDIFF_EXPERIMENT_NUM:-1}"
  export MMDIFF_LR_TAG="${MMDIFF_LR_TAG:-$MMDIFF_LEARNING_RATE}"
  case "$a" in
    random)
      export MMDIFF_RGB_STUDENT_CHECKPOINT=""
      export MMDIFF_FREEZE_RGB_STUDENT=0
      export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-${PREFIX}_rgb_student_${SC}_rand}"
      ;;
    freeze)
      export MMDIFF_RGB_STUDENT_CHECKPOINT="$RGB_STUDENT_CKPT"
      export MMDIFF_FREEZE_RGB_STUDENT=1
      export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-${PREFIX}_rgb_student_${SC}_freeze}"
      ;;
    finetune|ft)
      export MMDIFF_RGB_STUDENT_CHECKPOINT="$RGB_STUDENT_CKPT"
      export MMDIFF_FREEZE_RGB_STUDENT=0
      export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-${PREFIX}_rgb_student_${SC}_ft}"
      ;;
    *)
      echo "内部错误: setup_rgb_ablate $a" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
用法: bash run.sh [子命令]

  无参数 | all   串行：precompute →（distill + 三种主训练），其中 ablate_all 会先 distill 再消融；默认结束后关机（MMDIFF_SHUTDOWN_AT_END=0 则不关机）
  precompute     离线预计算 RGB teacher token（train + test）
  distill        蒸馏 student：默认最多 100 epoch、早停 15；TensorBoard 见 train_rgb_distill 输出路径
  ablate_all     先 distill（重新蒸馏）再串行 random → freeze → finetune
  train_random   消融：RGB student 随机初始化（不加载 MMDIFF_RGB_STUDENT_CHECKPOINT）
  train_freeze   消融：加载蒸馏权重 + 冻结 rgb_student（MMDIFF_FREEZE_RGB_STUDENT=1）
  train_finetune|train|main  消融：加载蒸馏权重 + 微调 rgb_student（默认）
  sanity         HSI 分支自检；额外参数: bash run.sh sanity --batch 4
  help           本说明

环境变量（节选）:
  扩散教师路径 / 默认 rgb_source / HR 严格视野：见 param.py（需 data_prepare 生成 rgb_hr.npy）
  MMDIFF_RGB_STUDENT_CHECKPOINT   轻量 RGB 编码器权重路径，默认 <仓库根>/rgb_student_distill.pt
  MMDIFF_FREEZE_RGB_STUDENT       1/true 冻结 rgb_student（runner 内重建优化器）
  MMDIFF_PRECOMPUTE_BATCH / MMDIFF_DISTILL_BATCH
  MMDIFF_DISTILL_EPOCHS  蒸馏最大 epoch，默认 100
  MMDIFF_DISTILL_EARLY_STOP  验证 loss 早停 patience，默认 15（0=关闭）
  MMDIFF_SHUTDOWN_AT_END=0 取消关机

示例:
  bash run.sh ablate_all
  bash run.sh all
  bash run.sh train_finetune
EOF
}

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
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown 2>/dev/null || true
    sleep 3
  done
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}

_run_train_ablate() {
  local tag="$1"
  echo "=== ${tag} | MMDIFF_EXPERIMENT_TAG=${MMDIFF_EXPERIMENT_TAG} | FREEZE=${MMDIFF_FREEZE_RGB_STUDENT:-0} | ckpt=${MMDIFF_RGB_STUDENT_CHECKPOINT:-<empty>} ==="
  python main.py
}

if [ $# -eq 0 ]; then
  set -- all
fi

case "$1" in
  help|-h|--help)
    usage
    ;;
  precompute)
    _pcb="${MMDIFF_PRECOMPUTE_BATCH:-32}"
    echo "=== precompute train+test (batch=$_pcb) HR 严格视野 -> param 中 teacher token 路径 ==="
    python utils/precompute_rgb_teacher_tokens.py --split train --batch-size "$_pcb"
    python utils/precompute_rgb_teacher_tokens.py --split test --batch-size "$_pcb"
    ;;
  distill)
    _de="${MMDIFF_DISTILL_EPOCHS:-100}"
    _db="${MMDIFF_DISTILL_BATCH:-0}"
    _esp="${MMDIFF_DISTILL_EARLY_STOP:-15}"
    echo "=== distill LightweightRgbEncoder -> $RGB_STUDENT_CKPT (max_epochs=$_de early_stop_patience=$_esp) ==="
    python utils/train_rgb_distill.py --epochs "$_de" --batch-size "$_db" --early-stopping-patience "$_esp" --out "$RGB_STUDENT_CKPT"
    ;;
  train_random)
    setup_rgb_ablate random
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    _run_train_ablate train_random
    ;;
  train_freeze)
    setup_rgb_ablate freeze
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    _run_train_ablate train_freeze
    ;;
  train_finetune|train|main)
    setup_rgb_ablate finetune
    export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
    _run_train_ablate train_finetune
    ;;
  ablate_all)
    PIPE_TS="$(date +%Y%m%d-%H%M%S)"
    (
      set +e
      log "ablate_all | PIPE_TS=$PIPE_TS（先 distill 再三种主训练）"
      FAILED=0
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step distill bash "$ROOT/run.sh" distill || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" RGB_STUDENT_CKPT="$RGB_STUDENT_CKPT" run_step train_random bash "$ROOT/run.sh" train_random || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" RGB_STUDENT_CKPT="$RGB_STUDENT_CKPT" run_step train_freeze bash "$ROOT/run.sh" train_freeze || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" RGB_STUDENT_CKPT="$RGB_STUDENT_CKPT" run_step train_finetune bash "$ROOT/run.sh" train_finetune || FAILED=$((FAILED + 1))
      log "ablate_all finished | failed_steps=$FAILED (0=all ok)"
      if [ "${MMDIFF_SHUTDOWN_AT_END:-1}" != "0" ]; then
        log "MMDIFF_SHUTDOWN_AT_END default/on -> shutdown (set MMDIFF_SHUTDOWN_AT_END=0 to skip)"
        do_shutdown
      fi
      if [ "$FAILED" -gt 0 ]; then
        exit 1
      fi
      exit 0
    )
    ;;
  all)
    PIPE_TS="$(date +%Y%m%d-%H%M%S)"
    (
      set +e
      log "all | PIPE_TS=$PIPE_TS（precompute → ablate_all 内含 distill+消融）"
      FAILED=0
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" run_step precompute bash "$ROOT/run.sh" precompute || FAILED=$((FAILED + 1))
      MMDIFF_RUN_TIMESTAMP="$PIPE_TS" RGB_STUDENT_CKPT="$RGB_STUDENT_CKPT" run_step ablate_all bash "$ROOT/run.sh" ablate_all || FAILED=$((FAILED + 1))
      log "all finished | failed_steps=$FAILED (0=all ok)"
      if [ "${MMDIFF_SHUTDOWN_AT_END:-1}" != "0" ]; then
        log "MMDIFF_SHUTDOWN_AT_END default/on -> shutdown (set MMDIFF_SHUTDOWN_AT_END=0 to skip)"
        do_shutdown
      fi
      if [ "$FAILED" -gt 0 ]; then
        exit 1
      fi
      exit 0
    )
    ;;
  sanity)
    echo "=== utils/hsi_branch_sanity.py ==="
    python utils/hsi_branch_sanity.py "${@:2}"
    ;;
  *)
    echo "未知子命令: $1" >&2
    usage >&2
    exit 1
    ;;
esac
