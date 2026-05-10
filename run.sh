#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# =============================================================================
# 实验设置
# =============================================================================
# arch_then_compare：架构消融（可选）→ 对比试验按 SEEDS 各跑一遍（main_compare）
# modality：主模型 main.py 仅 ABLATION_SEED（单种子模态消融）
# modality_multiseed：主模型 main.py 对 SEEDS 中每个种子各跑一遍模态消融（论文五种子）
# arch_ablation | modality | modality_multiseed | compare | all | arch_then_compare | patience_sweep
RUN_MODE="${RUN_MODE:-compare}"

EXP_TAG="Szutree_"
# 与 param.RANDOM_SEED 对齐；架构消融固定为 ABLATION_SEED
RANDOM_SEED="42"
ABLATION_SEED="${ABLATION_SEED:-42}"

# 对比试验：42、42×2、×4、×8、×16（与论文五种子一致）；架构消融不使用本列表
BASE_SEED="${BASE_SEED:-42}"
SEEDS=(
  "42"
  # "13"
  # "2026"
  # "4399"
)
PATIENCE_VALUES=(15 20 25 30)
FREEZE_RGB_STUDENT="0"
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

MODALITY_ABLATION_COMBOS=(
  hsi+rgb+lidar
  # hsi+rgb
  # hsi+lidar
  # rgb+lidar
  # hsi
  # rgb
  # lidar
)

MODELS=(
  "fgcnn:5e-4"
  "fusatnet:2e-5"
  "exvit:1e-3"
  "two_branch_cnn:1e-3"
  "hsi_msi_multisource:1e-3"
  "macn:1e-3"
  "ss_mae:1e-4"
)

# =============================================================================
# apply
# =============================================================================
apply_experiment_env() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

  export MMDIFF_EXPERIMENT_TAG="${MMDIFF_EXPERIMENT_TAG:-$EXP_TAG}"
  export MMDIFF_RANDOM_SEED="${MMDIFF_RANDOM_SEED:-$RANDOM_SEED}"
  export MMDIFF_RGB_STUDENT_CHECKPOINT="${MMDIFF_RGB_STUDENT_CHECKPOINT:-$RGB_STUDENT_CKPT}"
  export MMDIFF_FREEZE_RGB_STUDENT="${MMDIFF_FREEZE_RGB_STUDENT:-$FREEZE_RGB_STUDENT}"
  export MMDIFF_RUN_TIMESTAMP="exp_${MMDIFF_RUN_TIMESTAMP:-$(date +%m%d-%H%M)}"

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
  export MMDIFF_ARCH_VARIANT="${MMDIFF_ARCH_VARIANT:-}"
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
    export MMDIFF_EXPERIMENT_TAG="${EXP_TAG}mod_${suffix}"
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

# 主模型（main.py）模态消融：对 SEEDS 列表中每个种子各跑完整套 MODALITY_ABLATION_COMBOS
run_modality_ablation_multiseed() {
  local seed final_ec ec
  final_ec=0
  for seed in "${SEEDS[@]}"; do
    echo "#####################################################################"
    echo "# 主模型训练（main.py）多种子  SEED=${seed}  SEEDS=(${SEEDS[*]})"
    echo "#####################################################################"
    RANDOM_SEED="${seed}"
    export MMDIFF_RANDOM_SEED="${seed}"
    EXP_TAG="exp_es_s${seed}_"
    ec=0
    run_modality_ablation_series || ec=$?
    if [ "$ec" -ne 0 ]; then
      final_ec="$ec"
      echo "========== seed=${seed} 主模型训练非零退出: ${ec}（继续下一颗种子） ==========" >&2
    fi
  done
  return "$final_ec"
}

run_patience_sweep() {
  local batch_ts autodl_base p seed ec final_ec p_dir summary_csv
  batch_ts="$(date +%Y%m%d_%H%M%S)"
  autodl_base="../../autodl-tf-logs/${batch_ts}"
  summary_csv="${autodl_base}/patience_summary.csv"
  final_ec=0

  mkdir -p "${autodl_base}"
  echo "record_type,patience,seed,tag,oa,aa,kappa,oa_mean,oa_std,aa_mean,aa_std,kappa_mean,kappa_std,N" > "${summary_csv}"

  echo "#####################################################################"
  echo "# Patience Sweep: PATIENCE_VALUES=(${PATIENCE_VALUES[*]})  SEEDS=(${SEEDS[*]})"
  echo "# 输出目录: ${autodl_base}"
  echo "#####################################################################"

  for p in "${PATIENCE_VALUES[@]}"; do
    p_dir="${autodl_base}/p${p}"
    mkdir -p "${p_dir}"
    echo "========== 开始 patience=${p}，日志根目录=${p_dir} =========="
    for seed in "${SEEDS[@]}"; do
      export MMDIFF_TB_LOG_ROOT="${p_dir}"
      export MMDIFF_EARLY_STOPPING_PATIENCE="${p}"
      export MMDIFF_RANDOM_SEED="${seed}"
      export MMDIFF_EXPERIMENT_TAG="psweep_p${p}_s${seed}"
      export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
      RANDOM_SEED="${seed}"
      EXP_TAG=""
      apply_experiment_env
      echo "===> patience=${p} seed=${seed} TAG=${MMDIFF_EXPERIMENT_TAG}"
      ec=0
      python main.py || ec=$?
      if [ "$ec" -ne 0 ]; then
        final_ec="$ec"
        echo "========== patience=${p} seed=${seed} 非零退出: ${ec}（继续） ==========" >&2
      fi
    done

    python - "${p_dir}" "${p}" "${summary_csv}" <<'PYEOF'
import json
import math
import statistics
import sys
from pathlib import Path

p_dir = Path(sys.argv[1])
patience = sys.argv[2]
summary_csv = Path(sys.argv[3])
json_paths = sorted(p_dir.glob("**/metrics_summary.json"))

rows = []
oas, aas, kappas = [], [], []
for jp in json_paths:
    try:
        data = json.loads(jp.read_text(encoding="utf-8"))
    except Exception:
        continue
    tag = str(data.get("experiment_tag", "")).strip()
    if f"psweep_p{patience}_" not in tag:
        continue
    try:
        oa = float(data["oa"])
        aa = float(data["aa"])
        kappa = float(data["kappa"])
    except Exception:
        continue
    seed = ""
    marker = "_s"
    if marker in tag:
        seed = tag.rsplit(marker, 1)[-1]
    rows.append((seed, tag, oa, aa, kappa))
    oas.append(oa)
    aas.append(aa)
    kappas.append(kappa)

with summary_csv.open("a", encoding="utf-8", newline="") as f:
    if not rows:
        f.write(f"aggregate,{patience},,,,,,NA,NA,NA,NA,NA,NA,0\n")
        raise SystemExit(0)

    oa_mean = statistics.fmean(oas)
    aa_mean = statistics.fmean(aas)
    kappa_mean = statistics.fmean(kappas)
    oa_std = math.sqrt(statistics.pvariance(oas))
    aa_std = math.sqrt(statistics.pvariance(aas))
    kappa_std = math.sqrt(statistics.pvariance(kappas))

    for seed, tag, oa, aa, kappa in rows:
        f.write(
            f"seed_detail,{patience},{seed},{tag},{oa:.6f},{aa:.6f},{kappa:.6f},,,,,,,\n"
        )
    f.write(
        f"aggregate,{patience},,,,,,{oa_mean:.6f},{oa_std:.6f},"
        f"{aa_mean:.6f},{aa_std:.6f},{kappa_mean:.6f},{kappa_std:.6f},{len(rows)}\n"
    )
PYEOF
  done

  echo "===== patience sweep 完成，汇总文件: ${summary_csv} ====="
  return "$final_ec"
}

do_shutdown() {
  sleep 3
  local _i
  curl "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=训练完成&desp=Python脚本已执行完毕channel=9"
  for _i in 1 2 3 4 5 6 7 8 9; do
    /usr/bin/shutdown
    sleep 3
  done
  curl -fsS "https://sctapi.ftqq.com/SCT313662TGZ7JRPbisBQfDZbabO1Kmmdt.send?title=服务器关闭失败&desp=服务器关闭失败channel=9"
}

run_compare() {
  local s="${MMDIFF_RANDOM_SEED:-${RANDOM_SEED:-42}}"
  export MMDIFF_RANDOM_SEED="${s}"
  RANDOM_SEED="${s}"
  EXP_TAG=""
  echo "========== 对比试验  SEED=${s}  =========="
  for entry in "${MODELS[@]}"; do
      export MMDIFF_RUN_TIMESTAMP="$(date +%m%d_%H%M)"
      IFS=":" read -r model lr <<< "$entry"
      echo "===> seed=${s}  ${model}  LR=$lr"
      export MMDIFF_EXPERIMENT_TAG="${model}_s${s}"
      MMDIFF_LEARNING_RATE="$lr" python utils/main_compare.py --model "$model"
  done
}

# 架构消融：固定三模态与同协议；每组前清除易残留的环境变量
run_arch_ablation_series() {
  local final_ec ec
  final_ec=0
  export MMDIFF_MODALITY_COMBO="hsi+rgb+lidar"

  _arch_one() {
    local tag="$1"
    local variant="$2"
    local ec
    export MMDIFF_EXPERIMENT_TAG="${EXP_TAG}${tag}"
    export MMDIFF_RUN_TIMESTAMP="$(date +%m%d-%H%M)"
    unset MMDIFF_ARCH_VARIANT MMDIFF_LOSS_WEIGHT_GLOBAL MMDIFF_LOSS_WEIGHT_CENTER MMDIFF_USE_CENTER_LOSS 2>/dev/null || true
    export MMDIFF_ARCH_VARIANT="$variant"
    case "$3" in
      center_only_loss)
        export MMDIFF_LOSS_WEIGHT_GLOBAL=0
        export MMDIFF_LOSS_WEIGHT_CENTER=1
        export MMDIFF_USE_CENTER_LOSS=1
        ;;
      single_ce)
        export MMDIFF_USE_CENTER_LOSS=0
        ;;
      *)
        ;;
    esac
    echo "========== 架构消融: tag=${MMDIFF_EXPERIMENT_TAG} ARCH=${MMDIFF_ARCH_VARIANT} extra=${3:-none} =========="
    apply_experiment_env
    ec=0
    python main.py || ec=$?
    if [ "$ec" -ne 0 ]; then
      final_ec="$ec"
      echo "========== 本组训练非零退出: ${ec}（继续下一组） ==========" >&2
    fi
  }

  _arch_one "arch_full" "full" ""
  _arch_one "arch_no_dual_query" "single_query" "single_ce"
  _arch_one "arch_no_decoder_fusion" "concat_mlp" "single_ce"
  _arch_one "arch_no_dual_head_loss" "full" "center_only_loss"

  return "$final_ec"
}

# 默认流水线：架构消融仅 ABLATION_SEED；对比试验按 SEEDS 各跑一遍（param 读 MMDIFF_RANDOM_SEED）
run_arch_then_compare_multiseed() {
  # echo "#####################################################################"
  # echo "# 1/2 架构消融  仅 seed=${ABLATION_SEED}"
  # echo "#####################################################################"
  # RANDOM_SEED="${ABLATION_SEED}"
  # export MMDIFF_RANDOM_SEED="${ABLATION_SEED}"
  # EXP_TAG="s${ABLATION_SEED}_"
  # run_arch_ablation_series || true
  EXP_TAG=""

  local seed
  echo "#####################################################################"
  echo "# 2/2 对比试验  多种子 SEEDS=(${SEEDS[*]})"
  echo "#####################################################################"
  for seed in "${SEEDS[@]}"; do
    RANDOM_SEED="${seed}"
    export MMDIFF_RANDOM_SEED="${seed}"
    run_compare || true
  done
}

main() {
  case "${RUN_MODE}" in
    arch_then_compare)
      run_arch_then_compare_multiseed
      ;;
    arch_ablation)
      RANDOM_SEED="${ABLATION_SEED}"
      export MMDIFF_RANDOM_SEED="${ABLATION_SEED}"
      EXP_TAG="${EXP_TAG:-s${ABLATION_SEED}_}"
      run_arch_ablation_series
      ;;
    modality)
      RANDOM_SEED="${ABLATION_SEED}"
      export MMDIFF_RANDOM_SEED="${ABLATION_SEED}"
      # EXP_TAG="${EXP_TAG:-s${ABLATION_SEED}_}"
      EXP_TAG="exp_es"
      run_modality_ablation_series
      ;;
    modality_multiseed)
      run_modality_ablation_multiseed
      ;;
    compare)
      run_compare
      ;;
    patience_sweep)
      run_patience_sweep
      ;;
    all)
      RANDOM_SEED="${ABLATION_SEED}"
      export MMDIFF_RANDOM_SEED="${ABLATION_SEED}"
      EXP_TAG="${EXP_TAG:-s${ABLATION_SEED}_}"
      run_arch_ablation_series || true
      run_modality_ablation_series || true
      EXP_TAG=""
      run_compare
      ;;
    *)
      echo "未知 RUN_MODE=${RUN_MODE}，应为 arch_then_compare|arch_ablation|modality|modality_multiseed|compare|patience_sweep|all" >&2
      return 2 ;;
  esac
}

main
# python eval_ckps.py
do_shutdown
