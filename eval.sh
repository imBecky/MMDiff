#!/usr/bin/env bash
# 遍历某次训练 run 目录下所有 checkpoint-*，在官方 test 集（test_labels.npy）上评估各 classifier.pt。
# 与训练末尾 Final test 一致；不使用 inference.py（那是验证集）。
# 须使用 Unix 换行（LF）。在仓库根目录执行: bash eval.sh
#
#   export EVAL_RUN_DIR=/path/to/0409-1150_exp_se_cls_shallow
# 与训练结构一致时可复制 eval_env.example -> eval_env.sh（若存在则自动 source）
# 额外参数传给 eval_test.py，例如: bash eval.sh --batch-size 32
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/eval_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/eval_env.sh"
  echo "[eval] 已 source $ROOT/eval_env.sh"
  echo ""
fi

# 单次实验根目录（其下应有 checkpoint-<epoch>/classifier.pt，可选 final/classifier.pt）
: "${EVAL_RUN_DIR:=$ROOT/../../autodl-tmp/classifier/0409-1150_exp_se_cls_shallow}"

RUN_DIR="$(cd "$EVAL_RUN_DIR" 2>/dev/null && pwd)" || {
  echo "目录不存在: $EVAL_RUN_DIR" >&2
  exit 1
}

OUT_ROOT="${RUN_DIR}/eval_sweep_test"
mkdir -p "$OUT_ROOT"

echo "RUN_DIR=$RUN_DIR"
echo "OUT_ROOT=$OUT_ROOT  （各子目录内 test_metrics.json；最后 sweep_test_summary.json）"
echo ""

_n=0
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  name="$(basename "$d")"
  pt="$d/classifier.pt"
  if [[ ! -f "$pt" ]]; then
    echo "[skip] $name 无 classifier.pt"
    continue
  fi
  _n=$((_n + 1))
  sub="$OUT_ROOT/$name"
  mkdir -p "$sub"
  echo "======== $name （test）========"
  python eval_test.py --checkpoint "$pt" --out-dir "$sub" "$@"
  echo ""
done < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

if [[ "$_n" -eq 0 ]]; then
  echo "未找到可用 checkpoint（需存在 classifier.pt）: $RUN_DIR/checkpoint-*" >&2
  exit 1
fi

if [[ -f "$RUN_DIR/final/classifier.pt" ]]; then
  echo "======== final （test）========"
  mkdir -p "$OUT_ROOT/final"
  python eval_test.py --checkpoint "$RUN_DIR/final/classifier.pt" --out-dir "$OUT_ROOT/final" "$@"
fi

python eval_test.py --collect-sweep "$OUT_ROOT"

echo "全部完成。汇总: $OUT_ROOT/sweep_test_summary.json"
