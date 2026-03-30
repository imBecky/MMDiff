#!/usr/bin/env bash
# =============================================================================
# HSI 消融：D1 → D2 → A → B → C（单步扩散 t=50，与 param 中 MMDIFF_* 一致）
#
# D1 attn_pool   : 3×96×8，可学习 9 位置加权聚合
# D2 multi_token : 3×96×8，中心/四角/四边 三 token
# A mean（轻量） : 3×96×8，算术平均（原 baseline）
# B 大容量       : 5×128×8，算术平均
# C 放宽 SE      : 5×128×4，算术平均
#
# 用法（仓库根目录）：
#   chmod +x run.sh && ./run.sh
#
# 可选环境变量：
#   MMDIFF_NUM_EPOCHS              默认 300
#   MMDIFF_EXPERIMENT_TAG_PREFIX   默认 HSI（run tag: cls_${PREFIX}_...）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

unset MMDIFF_RESUME_CHECKPOINT 2>/dev/null || true

export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-300}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 单步扩散，与其它 HSI 对照表一致
export MMDIFF_DIFFUSION_TIMESTEPS="50"
export MMDIFF_LIDAR_EXTRA_BLOCKS="${MMDIFF_LIDAR_EXTRA_BLOCKS:-0}"

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-HSI}"

run_one() {
  local tag="$1"
  export MMDIFF_EXPERIMENT_TAG="${PREFIX}_${tag}"
  echo "========== ${MMDIFF_EXPERIMENT_TAG} | HSI rb=${MMDIFF_HSI_RESIDUAL_BLOCKS:-?} hidden=${MMDIFF_HSI_CONV_HIDDEN:-?} se=${MMDIFF_HSI_SE_RATIO:-?} agg=${MMDIFF_HSI_AGG_MODE:-?} =========="
  python main.py
}

# ---- D1：可学习空间加权（轻量 3×96×8）----
export MMDIFF_HSI_RESIDUAL_BLOCKS=3
export MMDIFF_HSI_CONV_HIDDEN=96
export MMDIFF_HSI_SE_RATIO=8
export MMDIFF_HSI_AGG_MODE=attn_pool
run_one "D1_attn_pool"

# ---- D2：三 token（轻量 3×96×8）----
export MMDIFF_HSI_AGG_MODE=multi_token
run_one "D2_multi_token"

unset MMDIFF_RESUME_CHECKPOINT 2>/dev/null || true
echo "========== 全部 HSI 消融跑完 =========="

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
curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"