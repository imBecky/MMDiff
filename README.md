# MSDQ — Houston2018 多模态地物分类

基于 **HSI（高光谱）+ RGB + LiDAR** 的多模态分类实验仓库，主干为自定义 **MultimodalClassifier**（空间 cross-attention 融合，`model/multimodal.py`）。支持模态消融、中心距离 bias 等超参，以及对比基线入口（见下文）。

---

## 快速开始

1. **准备数据**  
   - 放置 `houston2018.mat`（键名约定见 [`utils/Houston_mat_convention.md`](utils/Houston_mat_convention.md)）。  
   - 根据实际情况修改 [`data_prepare.py`](data_prepare.py) 顶部 `DATA_PATH` 与输出 `DATA_DIR`（默认可与 [`param.py`](param.py) 中路径对齐）。

2. **生成 `.npy` 与索引**

   ```bash
   python data_prepare.py
   ```

   主要产物：`train_patches.npy`（HSI+LiDAR）、`train_rgb_patches.npy`、`rgb_hr.npy`、`rgb_hr.meta.json`、`train_labels.npy` / `test_labels.npy`、`label_shift.npy`（详见 `data_prepare.py` 文件头）。

3. **训练**

   ```bash
   python main.py
   ```

   常用可选参数：`--verify-projection-grad`、`--no-artifacts`、`--no-conf-detail`（见 [`main.py`](main.py)）。

---

## Shell 消融脚本（Bash）

| 脚本 | 作用 |
|------|------|
| [`run.sh`](run.sh) | 7 组 `MMDIFF_MODALITY_COMBO` 串行消融；任一步失败即退出；末尾 **通知 webhook + 关机**（仅适合无人值守主机，使用前请按需裁剪）。 |
| [`run_cls_token_dim_ablation.sh`](run_cls_token_dim_ablation.sh) | 文件名历史遗留：**实际消融**中心距离 **`MMDIFF_CENTER_DISTANCE_BIAS_ALPHA`**（见脚本内注释与 `multimodal.py`）；串行两组 α 调用 `python main.py`。 |

脚本内普遍设置：`OMP_NUM_THREADS`、`MMDIFF_RANDOM_SEED`、`MMDIFF_RUN_TIMESTAMP`，并与 `MMDIFF_EXPERIMENT_TAG` / `MMDIFF_EXPERIMENT_NUM` 配合，避免多组实验写入同一 TensorBoard 目录。

---

## 目录结构（概要）

```
MSDQ/
├── main.py              # 训练 CLI 入口
├── param.py             # 默认配置 + MMDIFF_* 环境变量覆盖
├── data_prepare.py      # .mat → 整幅 numpy + 标签索引
├── eval_ckps.py         # 批量评估 checkpoint-* 下的 classifier.pt
├── pipeline/            # runner、data、dataloader、loop、日志、checkpoint
├── model/               # multimodal.py、rgb_student.py、spatial_fusion_decoder.py
├── utils/               # logger、校验与辅助脚本；Houston .mat 约定文档
├── knowledge_distill/   # RGB 蒸馏相关脚本
├── model/compare_model/ # 对比方法注册（配合 MMDIFF_COMPARE_RUN 等）
├── paper_drafts/        # 论文草稿占位（大部被 .gitignore，见目录内 README）
├── contrib/agent-collab-bootstrap/  # 协作模板（可入库）；复制到 agent-collab/
└── agent-collab/        # 【本地协作】.gitignore；见下方「Agent / Colab」
```

---

## 配置与扩展点

### `param.py`

- **`MMDIFF_*`**：在 `build_opt()` 之后在 `_apply_mmdiff_env_overrides()` 中覆盖训练/模型字段；完整枚举见函数文档字符串（如学习率、`MMDIFF_MODALITY_COMBO`、`MMDIFF_CENTER_DISTANCE_BIAS_ALPHA`、`MMDIFF_RESUME_CHECKPOINT`、`MMDIFF_*` scheduler 等）。
- **`ENABLED_MODALITIES`**：由 `MMDIFF_MODALITY_COMBO` 解析，`hsi+rgb+lidar` 等为常见写法。
- **路径常量**：默认 `DATA_DIR`、`CKPS_DIR`（分类权重）、`TB_LOG_ROOT` 等指向 **`../../autodl-fs`** / **`../../autodl-tmp`** / **`../../tf-logs`**；换环境时请统一改 `param.py`（及 `data_prepare.py` 中一致的路径）。

### 训练流程

- **入口**：`pipeline/runner.run_training` → DataLoader、`pipeline/loop`、`pipeline/logging_utils`。  
- **TensorBoard run 命名**：`prepare_tb_run_dir()`（时间戳、e 编号、`MMDIFF_EXPERIMENT_TAG` 等）。  
- **可选损失**：全局 + center 双 CE（`USE_CENTER_LOSS`），`pipeline/loop.compute_classification_loss`。

### RGB 严格视野与高分辨率块

启用 RGB 时，`pipeline/data.py` 可走 HR 对齐路径（依赖 `rgb_hr.npy` + `rgb_hr.meta.json`，由 `data_prepare.py` 写出）。

---

## 评估与对比实验

- **批量评测 checkpoint**：[`eval_ckps.py`](eval_ckps.py)。  
- **对比模型**：参见 `utils/main_compare.py`、`MMDIFF_COMPARE_RUN`、`MMDIFF_COMPARE_MODEL`（与 `logging_utils._is_compare_run` 等配合）。

---

## 依赖与环境

仓库未附带 `requirements.txt`；典型依赖包括：**PyTorch**、NumPy、SciPy、tqdm、scikit-learn、`tensorboard`。具体以脚本 `import` 为准。

建议在训练前导出合理 **`OMP_NUM_THREADS`**（[`main.py`](main.py) 在导入 torch 时空串会回填为 `4`，避免 libgomp 报错）。

---

## 安全与隐私

- **`run.sh`（及类似脚本）中含第三方通知 URL**。若其中有 token，请勿提交到公有仓库截图或日志副本；建议使用环境变量或本地覆盖脚本后再跑。  
- **数据集与原图**一般不提交 Git；沿用 `.gitignore` 中的 `*.pt`、`tf-logs/`、大文件规则。

---

## Agent / Colab 协作与工作日志

- **本地资料库**：根目录 **`agent-collab/`**（已列入 `.gitignore`），用于共享上下文与 **工作日志**（详见 `agent-collab/README.md`、`agent-collab/worklog/README.md`）。  
- **新 clone**：若暂无 `agent-collab/`，从 **[`contrib/agent-collab-bootstrap/`](contrib/agent-collab-bootstrap/README.md)** 复制整目录到仓库根并重命名为 `agent-collab`。  
- **Cursor 规则**：详见 [`.cursor/rules/agent-collab.mdc`](.cursor/rules/agent-collab.mdc)，要求会话开始优先阅读协作说明并更新 `worklog/`。
