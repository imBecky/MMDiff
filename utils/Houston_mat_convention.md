# Houston2018 兼容 .mat 约定（与 `utils/extrac_dataset.py` 一致）

`fetch_houston2018_ouc()` 等数据源提供的 Tensor 在脚本中组成为字典后 `savemat` 写入，键为：

| 键    | 约定形状 | 说明 |
|-------|-----------|------|
| `hsi` | 依数据源；后续常用 PCA 后 `(C,H,W)` 或本仓库 prepared 中 `(H,W,C)` 经 `data_prepare` 处理 | 多光谱立方体 |
| `lidar` | `(1,H,W)` 或经加载后与 HSI 同 (H,W) | CHW 或经转换 |
| `rgb`  | **`(3,H,W)` 或已与 HSI 同 `(H_lr,W_lr)`** | CHW（脚本内转 `(H,W,3)`）；可与 LR 一样大或与 LR 呈整倍数关系（大则裁切后再池化对齐） |
| `train` | 稀疏/稠密，非零 = 类 id | 与 **LR 标签栅格** 同 (H,W) |
| `test`  | 同上 | 与 train 不重叠的采样划分 |

`train`/`test` 的 (row,col) 即训练时裁 patch 的坐标；Houston 流程**不对标签做重采样**。

本仓库的 `data_prepare.py` 会读取上述结构（HSI+LiDAR 拼为 `(H,W,…)`、RGB 转 `(H,W,3)`），并生成 `train_labels.npy` 等索引表。

**SZUTree** 导出 `szutree_r1.mat` 时沿用同一键名，但空间对齐在 `extract_szutree_dataset.py` 中一次完成，使各模态与 `label.mat` 同栅格；详见 `extract_szutree_dataset.py` 头部说明与同目录 `*.meta.json`。
