# 会话日志：2026-04-02 — run.sh 主模型实验网格（五组）

## 元信息

- **触发**：依据 `best_model.log` 与用户经验（大 batch 略增 WD、SupCon 可能拖累、新 HSI 分支欠拟合），重设 exp1–exp3 并新增 exp4/exp5。
- **结论**：已写入 [`run.sh`](../../run.sh)；`bash run.sh all` 串行 exp1→exp5；对照链见下。
- **涉及路径**：`run.sh`（未改 `param.py` 调度默认值；cosine+warmup 仅经 `MMDIFF_*` 由脚本注入）。

## 做了什么

- **exp1** / 无参：BS64、cosine+warmup5%、lr=1e-3、wd=param 默认 1e-4、SupCon **ON**；tag `…_cos1e3_w5`。
- **exp2**：与 exp1 同调度同 BS/lr/wd，**SupCon OFF**；tag `…_nosupcon`。
- **exp3**：BS512、lr=4e-3、wd=**5e-4**、SupCon OFF、cosine+warmup（与 exp4 仅 WD 不同）。
- **exp4**：BS512、lr=4e-3、wd=**2e-4**、SupCon OFF。
- **exp5**：BS64、**piecewise**、lr=6e-4、wd=1e-4、SupCon OFF（贴近 best_model，无 SupCon）。
- **`all`**：上述五步顺序执行；`MMDIFF_SHUTDOWN_AT_END=1` 仍可关机。

## 未决 / 后续

- 跑完五组后对比 OA/AA/Kappa 与 `best_model.log`（OA 92.6% 等）；若需把某档固化为默认，再改 `setup_common` 或 `param`。

## 给下一个 Agent

- 实验含义表：见本会话用户附带的「实验网格重设计」计划或 [`INDEX.md`](../INDEX.md) 首行链到本文。
- 本目录默认被根目录 `.gitignore` 忽略；若需纳入版本库，对新增 md 使用 `git add -f`。
