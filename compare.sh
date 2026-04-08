#!/usr/bin/env bash
# 一键串行跑对比模型；每个模型可单独改 lr（MMDIFF_LEARNING_RATE）。
# 用法：在仓库根目录 bash compare.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 格式 model:lr（改数字即可）
MODELS=(
  "fgcnn:1e-3"
  "fusatnet:2e-5"
  "exvit:1e-3"
  "two_branch_cnn:1e-3"
  # DFINet = formango/HSI_MSI_Multisource_Classification（与 dfinet / formango_dfinet 同一实现）
  "hsi_msi_multisource:1e-3"
  "macn:1e-3"
  # SS-MAE（TGRS 2023）微调默认 lr 1e-4、wd 0.05 见官方 README；此处仅传 lr，wd 仍可用 MMDIFF_SSMAE_WEIGHT_DECAY
  "ss_mae:1e-4"
)

for entry in "${MODELS[@]}"; do
  IFS=":" read -r model lr <<< "$entry"
  echo "===> $model  LR=$lr"
  MMDIFF_LEARNING_RATE="$lr" python utils/main_compare.py --model "$model"
done
