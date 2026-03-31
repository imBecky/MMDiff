#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-150}"

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-smaller2_bs}"
# 与 param.py: TB_LOG_ROOT = Path('../../tf-logs') 对齐（在仓库根目录执行时）
TB_LOG_ROOT="${TB_LOG_ROOT:-../../tf-logs}"

# 模态：与 param.py DEFAULT_MODALITY_COMBO 一致，默认全模态；单模态消融可 export MMDIFF_MODALITY_COMBO=hsi
COMBO="${MMDIFF_MODALITY_COMBO:-hsi+rgb+lidar}"

# 与 param.py 默认对齐的「当前」基准：仅在本脚本中逐步加深 / 加大 SE / 加大 batch
# 基准 batch 可用 MMDIFF_BASE_BATCH 覆盖（默认与 param.BATCH_SIZE 一致）
BASE_BATCH="${MMDIFF_BASE_BATCH:-64}"
BASE_HIDDEN=96
BASE_SE=8
BASE_BLOCKS=4

# 更深 classifier：在 BASE_BLOCKS 上增加残差块数
DEEP_BLOCKS=6
# 更大 HSI SE（squeeze 比，数值更大 = 中间通道更宽，与 param 中 hsi_se_ratio 一致）
LARGE_SE=16

safe_combo="${COMBO//[^a-zA-Z0-9_-]/_}"

echo "========== 顺序实验（每次在前一步基础上叠加设置）=========="
echo "PREFIX=$PREFIX | TB_LOG_ROOT=$TB_LOG_ROOT"
echo "combo=$COMBO | 基准: blocks=$BASE_BLOCKS hidden=$BASE_HIDDEN SE=$BASE_SE batch=$BASE_BATCH"
echo "步骤: (1)更深 blocks=$DEEP_BLOCKS → (2)+更大SE=$LARGE_SE → (3–5) batch 128/256/512"
echo "=========================================================="

total_runs=0

run_train() {
  local tag_suffix="$1"
  local blocks="$2"
  local hidden="$3"
  local se="$4"
  local bs="$5"
  total_runs=$((total_runs + 1))
  local exp_tag="${PREFIX}_${safe_combo}_${tag_suffix}"

  echo ""
  echo ">>> [Run ${total_runs}] ${exp_tag}"
  echo "    blocks=${blocks} hidden=${hidden} se_ratio=${se} batch=${bs}"

  export MMDIFF_MODALITY_COMBO="$COMBO"
  export MMDIFF_EXPERIMENT_TAG="$exp_tag"
  export MMDIFF_HSI_RESIDUAL_BLOCKS="$blocks"
  export MMDIFF_HSI_CONV_HIDDEN="$hidden"
  export MMDIFF_HSI_SE_RATIO="$se"
  export MMDIFF_BATCH_SIZE="$bs"

  python main.py 2>&1
  echo "    [Run ${total_runs} 完成]"
}

# (1) 仅加深 backbone（相对 param 默认 4 块 → 6 块）
run_train "B${DEEP_BLOCKS}_H${BASE_HIDDEN}_SE${BASE_SE}_BS${BASE_BATCH}" \
  "$DEEP_BLOCKS" "$BASE_HIDDEN" "$BASE_SE" "$BASE_BATCH"

# (2) 保持 (1) 的深度，加大 SE
run_train "B${DEEP_BLOCKS}_H${BASE_HIDDEN}_SE${LARGE_SE}_BS${BASE_BATCH}" \
  "$DEEP_BLOCKS" "$BASE_HIDDEN" "$LARGE_SE" "$BASE_BATCH"

# (3–5) 保持 (2) 的结构，依次增大 batch
for bs in 128 256 512; do
  run_train "B${DEEP_BLOCKS}_H${BASE_HIDDEN}_SE${LARGE_SE}_BS${bs}" \
    "$DEEP_BLOCKS" "$BASE_HIDDEN" "$LARGE_SE" "$bs"
done

echo ""
echo "========== 全部 ${total_runs} 组实验完成 =========="

echo "========== 10 秒后关机 =========="
sleep 10
if command -v shutdown >/dev/null 2>&1; then
  shutdown -h now
elif command -v poweroff >/dev/null 2>&1; then
  poweroff
else
  echo "未找到 shutdown/poweroff，请手动关机。"
fi
