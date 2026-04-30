# Agent 工作日志索引（先读 [`README`](./README.md) 背景，再读本页）

> **渐进式暴露**：本文件只保留「最近几条 + 一行摘要」。完整过程写在 `logs/` 下对应文件，**不要**把长文堆进本索引。

| 最近更新 | 主题 | 一句话结果 | 详情 |
|---------|------|------------|------|
| 2026-04-26 | **MSDQ 论文重写 + 写作宝典** | 重构 Intro/Related Work/实验说明/表格分析；新增 `WRITING_GUIDE_MSDQ.md` 记录用户论文偏好、领域知识和后续写作规则 | [logs/2026-04-26-msdq-paper-rewrite-writing-guide.md](./logs/2026-04-26-msdq-paper-rewrite-writing-guide.md) |
| 2026-04-26 | **基线 LR 依据 + 实验公平与科研严谨** | Two-branch/MACN 等与主模型调度差异的可核查依据（BUCT `models.py`）；全员提醒：设置公平、披露协议、可复现 | [logs/2026-04-26-baseline-lr-fairness-research-rigor.md](./logs/2026-04-26-baseline-lr-fairness-research-rigor.md) |
| 2026-04-25 | **patience 扫描 + 论文草稿 + references 重排** | `MMDIFF_TB_LOG_ROOT`；`run.sh` 的 `patience_sweep` 与 `../../autodl-tf-logs/<ts>/`；`MSDQ_Draft.md` 中英稿；`references.md` 分节+表 | [logs/2026-04-25-patience-sweep-paper-drafts-references.md](./logs/2026-04-25-patience-sweep-paper-drafts-references.md) |
| 2026-04-24 | **README：主模型结构与方法要点** | `README.md` 纠偏「扩散+主模型」表述；新增 `MultimodalClassifier` 三分支、融合、配置入口摘要 | [logs/2026-04-24-readme-multimodal-structure.md](./logs/2026-04-24-readme-multimodal-structure.md) |
| 2026-04-24 | **严格 HR / eval 体检 / 数据报告脚本 / 全库风险 / 论文草稿目录** | `rh==rw` 与 strict+旋转；`eval_ckps` 过滤写死&危险默认；`verify_prepared_data`→`autodl-tmp`；param 依 cwd、VAL_RATIO=0 用 test 选优；`paper_drafts`+ignore | [logs/2026-04-24-cursor-tooling-strict-hr-eval-audit-drafts.md](./logs/2026-04-24-cursor-tooling-strict-hr-eval-audit-drafts.md) |
| 2026-04-10 | **HSI：数据 patch vs 模型内 3×3** | `PATCH_WINDOW_SIZE` 管外层块；`HSICenterSpectralEncoder` 内 `_crop_center_3x3` 再取中心 3×3 做光谱编码；LiDAR 用整 patch | [logs/2026-04-10-multimodal-hsi-center-3x3-crop.md](./logs/2026-04-10-multimodal-hsi-center-3x3-crop.md) |
| 2026-04-09 | **RGB(student)→LiDAR FiLM + fus.sh 覆盖** | 移除 LiDAR→HSI A/B/C；`MMDIFF_RGB_TO_LIDAR_GUIDANCE=none|film`；RGB token 均值 FiLM→`lidar_g/c`；`fus.sh` 仅 film+`train_exp` | [logs/2026-04-09-rgb-student-to-lidar-film-fus.md](./logs/2026-04-09-rgb-student-to-lidar-film-fus.md) |
| 2026-04-07 | **对比基线 SS-MAE（TGRS 2023）** | 迁入 [summitgao/SS-MAE](https://github.com/summitgao/SS-MAE) `VisionTransfromers`；注册 `ss_mae`/`ssmae`；`crop_size`/PCA 环境变量；`compare.sh` 示例 `ss_mae:1e-4` | [logs/2026-04-07-compare-ss-mae.md](./logs/2026-04-07-compare-ss-mae.md) |
| 2026-04-08 | **LiDAR→HSI 引导 A/B/C + fus.sh**（**已废弃**，见 04-09） | 历史记录 | [logs/2026-04-08-lidar-guidance-abc-fus-sh.md](./logs/2026-04-08-lidar-guidance-abc-fus-sh.md) |
| 2026-04-08 | **run.sh：SE24 四组网格** | 去掉 base/lidar_wide；`train_exp exp1–4`；exp3 略增 wd；exp4 cls 320/160 + wd；`MMDIFF_CLS_*` 入 param | [logs/2026-04-08-runsh-grid-se24-wd-cls.md](./logs/2026-04-08-runsh-grid-se24-wd-cls.md) |
| 2026-04-07 | **DFINet = formango 官方仓** | 已接 DFINet；新增注册别名 `formango_dfinet`/`hsi_msi_multisource`，`DFINET_PROTOCOL_COMPARE_NAMES` 统一走 `dfinet_protocol` | [logs/2026-04-07-dfinet-formango-registry-aliases.md](./logs/2026-04-07-dfinet-formango-registry-aliases.md) |
| 2026-04-06 | **run.sh 精简 + HSI 48 + 融合仅 cross** | 根 `run.sh`：`bash "$RUN_SH"`、LF、**distill→grid**、无 precompute；`HSI_CHANNELS=48`/`MMDIFF_HSI_CHANNELS`；`load_train_bundle` 维校验；去掉 concat 与 `FUSION_MODE`；`mmdd-hhmm` | [logs/2026-04-06-runsh-slim-hsi48-cross-only.md](./logs/2026-04-06-runsh-slim-hsi48-cross-only.md) |
| 2026-04-04 | **param 瘦身 + RGB 扩散教师** | 256 DDPM 默认路径、`rgb_source=student`、仅 HR+`*_strict` 缓存、减 env/减 `run.sh` 重复；原则见日志 | [logs/2026-04-04-param-slim-rgb-teacher.md](./logs/2026-04-04-param-slim-rgb-teacher.md) |

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
