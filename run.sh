#!/usr/bin/env bash
# =============================================================================
# MMDiff：按三组扩散时间步列表依次各跑一遍完整训练（每组一次 python main.py）
#   [50,100]、[50,100,150]、[20,50]
#
# 说明：
# - 每组通过 MMDIFF_DIFFUSION_TIMESTEPS 传入 param；MMDIFF_EXPERIMENT_TAG 自动带 t 列表便于区分目录。
# - 可改 MMDIFF_EXPERIMENT_TAG_PREFIX（默认 B）或下方 TS_LIST。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

unset MMDIFF_RESUME_CHECKPOINT 2>/dev/null || true

export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-300}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-B}"
# 三组 t 组合（逗号分隔，与 param 中 MMDIFF_DIFFUSION_TIMESTEPS 一致）
TS_LIST=(
  "50,100"
  "50,100,150"
  "20,50"
)

for ts in "${TS_LIST[@]}"; do
  export MMDIFF_DIFFUSION_TIMESTEPS="$ts"
  safe="${ts//,/_}"
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_multiT_${safe}"
  echo "========== MMDIFF_DIFFUSION_TIMESTEPS=${ts}  MMDIFF_EXPERIMENT_TAG=${MMDIFF_EXPERIMENT_TAG} =========="
  python main.py
done

unset MMDIFF_RESUME_CHECKPOINT

curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 3
/usr/bin/shutdown
sleep 10
/usr/bin/shutdown
sleep 3
curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
