# Agent 工作日志索引（先读 [`README`](./README.md) 背景，再读本页）

> **渐进式暴露**：本文件只保留「最近几条 + 一行摘要」。完整过程写在 `logs/` 下对应文件，**不要**把长文堆进本索引。

| 最近更新 | 主题 | 一句话结果 | 详情 |
|---------|------|------------|------|
| 2026-04-04 | **param 瘦身 + RGB 扩散教师** | 256 DDPM 默认路径、`rgb_source=student`、仅 HR+`*_strict` 缓存、减 env/减 `run.sh` 重复；原则见日志 | [logs/2026-04-04-param-slim-rgb-teacher.md](./logs/2026-04-04-param-slim-rgb-teacher.md) |
| 2026-04-03 | **RGB 轻量 student：蒸馏 TB+早停、teacher 缓存与 run.sh 消融** | `rgb_teacher_cache`/`open_memmap`/`tempt.py`；`train_rgb_distill` 默认 100ep、**早停 15**、TensorBoard；**`MMDIFF_FREEZE_RGB_STUDENT`**；**`ablate_all`=先 distill 再 random/freeze/ft**；**`all`=precompute→ablate_all**（不重复 distill）；`run.sh` 不写本地 tee 日志 | [logs/2026-04-03-rgb-student-distill-ablation-and-cache.md](./logs/2026-04-03-rgb-student-distill-ablation-and-cache.md) |
| 2026-04-03 | **run.sh 续训 exp3/4/5** | `exp3r`/`exp4r`/`exp5r` 从 `final` 续训；`resume345` 串行；`MMDIFF_RESUME_EXP*`/`RESUME_RUN_TS` 可覆盖；**exp5=piecewise** 对齐 best、非 cosine；runner 续训后 lr×0.5 | [logs/2026-04-03-runsh-resume-exp345-and-exp5-scheduler.md](./logs/2026-04-03-runsh-resume-exp345-and-exp5-scheduler.md) |
| 2026-04-02 | **run.sh 主模型五组实验网格** | exp1=cosine+SupCon；exp2 关 SupCon；exp3/4=BS512 仅 WD 5e-4 vs 2e-4；exp5=piecewise 6e-4 无 SupCon；`all` 串行 1→5 | [logs/2026-04-02-runsh-exp-grid-five.md](./logs/2026-04-02-runsh-exp-grid-five.md) |
| 2026-04-02 | **对比模型可复现 / 学术诚信（强制）** | 除统一数据+epoch/lr 等基础项外，对比须对齐**原论文方法**；Two-branch 三阶段；**DFINet**（联合损失+SGD，`dfinet_protocol`）与 **MACN**（Focal+Adam+无调度）已接；本机跑 Python 先 **`conda activate hbq`**（见 [`README`](./README.md)） | [logs/2026-04-02-compare-models-repro-integrity.md](./logs/2026-04-02-compare-models-repro-integrity.md) |
| 2026-04-02 | 双分支 CNN 对比模型（BUCT Xu 2017） | PyTorch 接入；**已纠偏为三阶段协议**；旧日志中「端到端」表述过时，以 repro-integrity + `two_branch_protocol` 为准 | [logs/2026-04-02-two-branch-cnn-compare.md](./logs/2026-04-02-two-branch-cnn-compare.md) |
| 2026-04-01 | `run.sh` 合并、`param` 环境覆盖、HSI sanity | 单脚本 baseline/exp1–3/all；`MMDIFF_LR/WD`；删 `run_experiments`/`run_all_overnight`；sanity 用 `no_grad` | [logs/2026-04-01-runsh-param-sanity.md](./logs/2026-04-01-runsh-param-sanity.md) |
| 2026-04-01 | HSI `HSICenterSpectralEncoder` 顺序 | SE 前移到 pool 前：stem→res→gate→pool→spatial_agg→proj；见 `multimodal.py` | [logs/2026-04-01-hsi-se-gate-before-pool.md](./logs/2026-04-01-hsi-se-gate-before-pool.md) |
| 2026-03-31 | F-GCN 对比、早停、run.sh 与全模态 | 仅保留 F-GCN；param 须含早停与 batch 覆盖；run.sh 默认 hsi+rgb+lidar | [logs/2026-03-31-fgcn-compare-earlystop-runsh.md](./logs/2026-03-31-fgcn-compare-earlystop-runsh.md) |
| 2026-03-31 | 合并 stash + 服务器 HSI 配置 | 提交 a6d405a；stash 已 drop；保留 run.sh 网格与 param | [logs/2026-03-31-merge-stash-server.md](./logs/2026-03-31-merge-stash-server.md) |

---

## 给其他 Agent 的阅读顺序（省 token）

1. **已读** [`README.md`](./README.md) 后，再读本页 → 知道最近谁在做什么、是否与自己相关。
2. **若相关** → 打开上表「详情」列中的 **单个** `logs/*.md`。
3. **仍不够** → 再看该 session 里写的「仓库路径 / commit / 服务器命令」，必要时才读源码。

不要一次性读完整个 `logs/` 目录。

---

## 写入规则（每次会话结束前）

1. 在 `logs/` 新建 `YYYY-MM-DD-简短主题.md`（同一天多条可加后缀 `-2`）。
2. 按 `logs/_TEMPLATE.md` 填写；长讨论、大段输出只放详情文件。
3. **回到本文件**：在表格**顶部插入一行**（最新在上），主题仍为一行摘要。
4. **控制索引体积**：表格超过 **12 行**时，把最旧的几行剪切到 `logs/archive-index.md`（只保留日期+主题+文件链接一行）。

---

## 快速链接

- [项目静态上下文（环境与关键路径）](./README.md)
- [会话日志目录](./logs/)
