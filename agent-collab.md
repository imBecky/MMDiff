# Agent collab — 数据集划分与本地标签核查记录

供论文 **Experimental setup / Data protocol**、协作 Agent 与后续实验复现共用。本文区分 **Houston2018（官方稀疏标注）**、**SZUTree（脚本按类比例抽样）**，并记录 **两次**对用户本地 `E:\MMDiff\train_labels.npy` / `test_labels.npy` 的读取结果（两次统计不同，说明其间更换过数据或重新生成过 `prepared`）。

---

## A. 各数据集「训练/测试」是怎么来的

### 1) Houston2018（`rs-fusion-datasets`）

- **来源**：`from rs_fusion_datasets import fetch_houston2018_ouc`，`train_label` / `test_label` 为数据包（OUC 镜像 / IEEE GRSS Data Fusion 2018 协议）**已给定的稀疏矩阵**。
- **本仓库**：`utils/extrac_dataset.py` **不再随机划分**，仅将 `a[2]`、`a[3]` 写入 `houston2018.mat` 的 `train` / `test`。
- **含义**：训练像素 = **官方固定协议下的有标注训练位置**；**不是**本仓库里「每类总像素 × 固定百分比」那种在导出脚本中显式计算的比例抽样。各类训练样本数常呈现 **近似固定档位**（例如多数类 1000、少数类更少），与 OUC 公开设定一致。
- **论文表述建议**：采用公开基准提供的 Houston2018 **官方训练/测试稀疏标签**（经 [rs-fusion-datasets](https://github.com/songyz2019/rs-fusion-datasets) 加载）。

### 2) SZUTree（`utils/extract_szutree_dataset.py`）

- **LR 标签**：5 cm `label.mat` → **2×2 众数**下采样至与 HSI/LiDAR 一致的 LR 网格；必要时对标签/RGB 做与 HSI 空间维对齐的转置处理（见脚本说明）。
- **划分**：对 LR 标签图上 **每个前景类 `c ∈ {1,…,20}`** 独立随机打乱，取约 **`train_percent`%** 的该类标注像素进 **train**，其余进 **test**（`--train-percent-per-class`，默认 `1.0` 即 **每类 1%**）；非空类至少保留 1 个训练像素（若实现如此）。**train/test 像素不重叠**。
- **类极不平衡时**：按类 1% 属于 **per-class stratified** 协议，保证每类都进入训练集；测试集上类别仍可能极不均衡，报告指标时建议除 OA 外写明 **AA / 每类精度 / κ** 等。
- **论文表述建议**：写明 **per-class random sampling**、百分比 **p**、随机种子、与 Houston **官方固定标注** 非同一协议（若同一篇论文对比两者）。

### 3) 生成 `prepared` 之后（两种数据集共用）

- **`utils/data_prepare.py`**：读取 `.mat` 中 `train` / `test` 矩阵，对 **`train > 0` / `test > 0`** 的像素建索引，得到 `train_labels.npy` / `test_labels.npy`，形状 **`(N, 3)` int32**：`[原始类别 id, row, col]`；**不在此步再次随机抽样**。
- **`label_shift.npy`**：train 与 test 中出现的最小原始标签，训练时平移到 `0 … NUM_CLASSES-1`。
- **实际训练循环**（`pipeline/data.py`）：从 **`train_labels.npy`** 再按 **`VAL_RATIO`**（默认 0.1）做 **分层** train/val 划分（`stratify=labels`）。**`test_labels.npy` 对应独立测试集**，一般不参与该划分。

---

## B. 本地标签文件两次读取结果（`E:\MMDiff\`）

以下均为在协作对话中通过 Python `np.load` 读取 **`E:\MMDiff\train_labels.npy`** 与 **`E:\MMDiff\test_labels.npy`** 的统计。**两次结果不一致**，表明用户曾在两次读取之间 **更换过数据集或重新生成了标签文件**。

### 第一次读取（与 SZUTree「每类约 1%」协议一致）

| 文件 | shape | 第 0 列 (原始类 id) | 行列范围 (LR) | 备注 |
|------|--------|---------------------|---------------|------|
| `train_labels.npy` | `(15625, 3)` | 1–20 | row ≈ 39–3045, col ≈ 0–2404 | 20 类均有样本；**每类约占该类 train+test 总和约 1%** |
| `test_labels.npy` | `(1546836, 3)` | 1–20 | row ≈ 32–3050, col 0–2404 | 与 train 规模比约 **99% : 1%** |

- **`(row,col)` train ∩ test**：**0**（无泄漏）。
- **全局**：train 约占全部标注像素 **≈ 1.000%**（15625 / 1562461）。

### 第二次读取（与 Houston2018 OUC 常见「每类固定训练样本数」形态一致）

| 文件 | shape | 第 0 列 (原始类 id) | 行列范围 (LR) | 备注 |
|------|--------|---------------------|---------------|------|
| `train_labels.npy` | `(18750, 3)` | 1–20 | row 0–1201, col 0–4767 | **多数类 1000**；**id 7 → 500**，**id 17 → 250**；其余 18 类各 **1000** |
| `test_labels.npy` | `(2000160, 3)` | 1–20 | row 0–1201, col 0–4767 | 各类像素数**极不均衡**（如某类近 90 万、某类仅数百至数千） |

- **`(row,col)` train ∩ test**：**0**。
- **全局**：train 约占 train+test 总和 **≈ 0.9287%**（18750 / 2018910）。

**解读**：第二次统计更接近 **Houston 官方训练集「每类有上限的固定标注」**；第一次更接近 **SZUTree 脚本默认每类 1%**。写作与跑实验时请 **以当前磁盘上的 `train_labels.npy` 为准** 并核对 `DATA_DIR` 与数据来源。

---

## C. English (paper-ready, both protocols)

**Houston2018.** We use the public Houston2018 benchmark with **official sparse training and testing annotations** provided by the dataset release and loaded via `rs-fusion-datasets` (`fetch_houston2018_ouc`). Our codebase does not re-sample pixels for this split; `data_prepare.py` exports all training-labeled pixels into `train_labels.npy` and all test-labeled pixels into `test_labels.npy`.

**SZUTree.** We build an LR label map by **2×2 majority downsampling** of the 5 cm reference labels to match the HSI/LiDAR grid. For each of the 20 foreground classes, we randomly select **p%** (default **p = 1**) of that class’s labeled LR pixels for **training** and assign the rest to **testing**, with a fixed seed and **no pixel overlap** between splits. After exporting a unified `.mat`, `data_prepare.py` converts the `train`/`test` masks into index files. **This per-class percentage protocol differs from Houston2018’s official fixed training mask.**

**Training vs. validation (both datasets).** From `train_labels.npy`, we further hold out a **stratified** validation subset using `VAL_RATIO` (default 10%) before optimization.

---

## D. 命令备忘

```text
# SZUTree → 兼容 .mat
python utils/extract_szutree_dataset.py --export --train-percent-per-class 1.0 --seed 42
# 预处理前需 MMDIFF_HSI_CHANNELS=98
python utils/data_prepare.py

# Houston：通常由 extrac_dataset.py + fetch_houston2018_ouc 生成 .mat 后再 data_prepare
```
