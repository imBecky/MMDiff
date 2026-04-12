#!/usr/bin/env bash
# 主入口：映射为 MMDIFF_* 后运行 main.py；默认跑满模态消融 7 组。
# 须使用 Unix 换行（LF）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# =============================================================================
# 实验设置（只改这里）：短名 ≠ 环境变量名
# 若已在 shell 中 export 了同名 MMDIFF_*，则以环境变量为准
# =============================================================================
EXP_TAG="exp_"

# 与 param.RANDOM_SEED 对齐；可通过 export MMDIFF_RANDOM_SEED=… 覆盖
RANDOM_SEED="42"

RGB_SRC="student"
FREEZE_RGB_STUDENT="1"
RGB_STUDENT_CKPT="${ROOT}/model/rgb_student_distill.pt"

# HSI：SE 挤压比、残差深度、conv 宽、空间聚合（对齐 exp_se_cls_shallow：5×96+SE32+mean）
HSI_SE_RATIO="32"
HSI_RESIDUAL_BLOCKS="5"
HSI_CONV_HIDDEN="96"
HSI_AGG_MODE="mean"

# LiDAR 分支（48+2 残差）
LIDAR_HIDDEN="48"
LIDAR_EXTRA_BLOCKS="2"

# 分类头 Transformer + token（320/320/384）
CLS_TOKEN_DIM="320"
CLS_HEAD_HIDDEN="320"
CLS_TRANSFORMER_LAYERS="1"
CLS_TRANSFORMER_FF_DIM="384"

# 模态消融顺序：单模态 → 两两 → 三模态（与 param.DEFAULT_MODALITY_COMBO 的 + 连接一致）
MODALITY_ABLATION_COMBOS=(
  hsi+rgb+lidar
  hsi+rgb
  hsi+lidar
  rgb+lidar
  hsi
  rgb
  lidar
)

# 仅跑一组或自定义列表：export MMDIFF_MODALITY_COMBO=hsi+rgb 且 RUN_MODALITY_ABLATION=0
# 跳过整段消融、只训一次（默认全模态）：RUN_MODALITY_ABLATION=0 bash run.sh
RUN_MODALITY_ABLATION="${RUN_MODALITY_ABLATION:-1}"

# =============================================================================
# apply
# =============================================================================
apply_experiment_env() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

  export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-$EXP_TAG}"
  export MMDIFF_RANDOM_SEED="${MMDIFF_RANDOM_SEED:-$RANDOM_SEED}"
  export MMDIFF_RGB_SOURCE="${MMDIFF_RGB_SOURCE:-$RGB_SRC}"
  export MMDIFF_RGB_STUDENT_CHECKPOINT="${MMDIFF_RGB_STUDENT_CHECKPOINT:-$RGB_STUDENT_CKPT}"
  export MMDIFF_FREEZE_RGB_STUDENT="${MMDIFF_FREEZE_RGB_STUDENT:-$FREEZE_RGB_STUDENT}"
  export MMDIFF_RUN_TIMESTAMP="${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"

  # 与 param.py ENABLED_MODALITIES 一致；未设时由 param 内部默认 hsi+rgb+lidar
  export MMDIFF_MODALITY_COMBO="${MMDIFF_MODALITY_COMBO:-}"

  # HSI SE：优先 MMDIFF_HSI_SE_RATIO，其次兼容旧名 MMDIFF_GRID_SE，最后用文件内 HSI_SE_RATIO
  export MMDIFF_HSI_SE_RATIO="${MMDIFF_HSI_SE_RATIO:-${MMDIFF_GRID_SE:-$HSI_SE_RATIO}}"
  export MMDIFF_CLS_TOKEN_DIM="${MMDIFF_CLS_TOKEN_DIM:-$CLS_TOKEN_DIM}"
  export MMDIFF_CLS_HEAD_HIDDEN="${MMDIFF_CLS_HEAD_HIDDEN:-$CLS_HEAD_HIDDEN}"
  export MMDIFF_HSI_RESIDUAL_BLOCKS="${MMDIFF_HSI_RESIDUAL_BLOCKS:-$HSI_RESIDUAL_BLOCKS}"
  export MMDIFF_HSI_CONV_HIDDEN="${MMDIFF_HSI_CONV_HIDDEN:-$HSI_CONV_HIDDEN}"
  export MMDIFF_HSI_AGG_MODE="${MMDIFF_HSI_AGG_MODE:-$HSI_AGG_MODE}"
  export MMDIFF_LIDAR_HIDDEN="${MMDIFF_LIDAR_HIDDEN:-$LIDAR_HIDDEN}"
  export MMDIFF_LIDAR_EXTRA_BLOCKS="${MMDIFF_LIDAR_EXTRA_BLOCKS:-$LIDAR_EXTRA_BLOCKS}"
  export MMDIFF_CLS_TRANSFORMER_LAYERS="${MMDIFF_CLS_TRANSFORMER_LAYERS:-$CLS_TRANSFORMER_LAYERS}"
  export MMDIFF_CLS_TRANSFORMER_FF_DIM="${MMDIFF_CLS_TRANSFORMER_FF_DIM:-$CLS_TRANSFORMER_FF_DIM}"
}

_combo_to_tag_suffix() {
  echo "$1" | tr '+/' '__'
}

run_modality_ablation_series() {
  local combo suffix final_ec ec
  final_ec=0
  for combo in "${MODALITY_ABLATION_COMBOS[@]}"; do
    suffix="$(_combo_to_tag_suffix "$combo")"
    export MMDIFF_MODALITY_COMBO="$combo"
    export MMDIFF_EXPERIMENT_TAG="${EXP_TAG}_mod_${suffix}"
    export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
    echo "========== 模态消融: MMDIFF_MODALITY_COMBO=${combo}  TAG=${MMDIFF_EXPERIMENT_TAG} =========="
    apply_experiment_env
    ec=0
    python main.py || ec=$?
    if [ "$ec" -ne 0 ]; then
      final_ec="$ec"
      echo "========== 本组训练非零退出: ${ec}（继续下一组） ==========" >&2
    fi
  done
  return "$final_ec"
}

run_single_train() {
  apply_experiment_env
  ec=0
  python main.py || ec=$?
  return "$ec"
}

do_shutdown() {
  sleep 3
  local _i
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9" || true
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown 2>/dev/null || true
    sleep 3
  done
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9" || true
}

train_ec=0
if [ "$RUN_MODALITY_ABLATION" = 1 ]; then
  run_modality_ablation_series || train_ec=$?
else
  run_single_train || train_ec=$?
fi

# 成功/失败均关机（省实例费）；本地不关机: MMDIFF_SKIP_SHUTDOWN=1 bash run.sh
[ "${MMDIFF_SKIP_SHUTDOWN:-0}" = 1 ] || do_shutdown
exit "$train_ec"
