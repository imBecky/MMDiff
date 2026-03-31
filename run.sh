#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MMDIFF_NUM_EPOCHS="${MMDIFF_NUM_EPOCHS:-200}"

PREFIX="${MMDIFF_EXPERIMENT_TAG_PREFIX:-modality}"
# 与 param.py: TB_LOG_ROOT = Path('../../tf-logs') 对齐（在仓库根目录执行时）
TB_LOG_ROOT="${TB_LOG_ROOT:-../../tf-logs}"

# HSI网络结构配置组合
# 更深：HSI_RESIDUAL_BLOCKS_CFG 从 4 增加到 5, 6
# 加深时同步略增 HSI_CONV_HIDDEN_CFG；更宽时进一步调大
HSI_RESIDUAL_BLOCKS_CFGS=(4 5 6)
HSI_CONV_HIDDEN_CFGS=(96 128 160)
HSI_SE_RATIO_CFGS=(8 16)

COMBOS=(
  # "hsi+rgb+lidar"
  # "rgb+lidar"
  # "hsi+lidar"
  # "hsi+rgb"
  "hsi"
  # "rgb"
  # "lidar"
)

echo "========== 多模态分支消融 + HSI结构搜索批量运行 =========="
echo "PREFIX=$PREFIX | TB_LOG_ROOT=$TB_LOG_ROOT"
echo "HSI_RESIDUAL_BLOCKS_CFGS: ${HSI_RESIDUAL_BLOCKS_CFGS[*]}"
echo "HSI_CONV_HIDDEN_CFGS: ${HSI_CONV_HIDDEN_CFGS[*]}"
echo "HSI_SE_RATIO_CFGS: ${HSI_SE_RATIO_CFGS[*]}"
echo "=========================================================="

total_runs=0
for combo in "${COMBOS[@]}"; do
  for blocks in "${HSI_RESIDUAL_BLOCKS_CFGS[@]}"; do
    for hidden in "${HSI_CONV_HIDDEN_CFGS[@]}"; do
      for se_ratio in "${HSI_SE_RATIO_CFGS[@]}"; do
        ((++total_runs))
        # 构建安全的标识符（替换特殊字符）
        safe_combo="${combo//[^a-zA-Z0-9_-]/_}"
        
        # 实验标签：包含模态组合和网络结构参数
        # 格式: {PREFIX}_{safe_combo}_B{blocks}_H{hidden}_SE{se_ratio}
        exp_tag="${PREFIX}_${safe_combo}_B${blocks}_H${hidden}_SE${se_ratio}"
        
        echo ""
        echo ">>> [Run $total_runs] 组合=$combo | Blocks=$blocks | Hidden=$hidden | SE=$se_ratio"
        echo "    实验标签: $exp_tag"
        
        # 导出环境变量供Python代码读取
        export MMDIFF_MODALITY_COMBO="$combo"
        export MMDIFF_EXPERIMENT_TAG="$exp_tag"
        export HSI_RESIDUAL_BLOCKS_CFG="$blocks"
        export HSI_CONV_HIDDEN_CFG="$hidden"
        export HSI_SE_RATIO_CFG="$se_ratio"
        
        # 运行训练
        python main.py 2>&1
        
        echo "    [Run $total_runs 完成]"
      done
    done
  done
done

echo ""
echo "========== 全部 $total_runs 组实验完成 =========="
# echo "========== 全部消融跑完。单次 run 产物：${TB_LOG_ROOT}/<run_tag>/（model.log、metrics_summary.json 等）=========="

# curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# /usr/bin/shutdown
# sleep 3
# curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"