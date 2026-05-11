# 工作日志（Agent / 人类共用）

## 写什么

每条记录建议包含：

- **日期**（会话当天）
- **目标**：本次要完成什么
- **已做**：改动的路径、跑的命令、`MMDIFF_*` 关键取值
- **结果**： metrics / OA、TensorBoard run 目录名、`checkpoint-*` 路径（若已知）
- **阻塞**：报错摘要、缺失数据文件等
- **下一步**：交接给下一轮 agent / 自己的事

## 怎么组织文件

任选一种，团队内保持一致即可：

1. **按日**：`YYYY-MM-DD.md`（同一天多次会话用二级标题 `###` 分段）。
2. **按专题**：例如 `modal_ablation.md`、`center_bias_alpha.md`。

## 模板片段（复制到新条目）

```markdown
### YYYY-MM-DD — <一句话标题>

- 目标：
- 已做：
- 环境与变量：（例 `MMDIFF_MODALITY_COMBO=...`、`MMDIFF_CENTER_DISTANCE_BIAS_ALPHA=...`）
- 产出路径：`tf-logs/...`、`classifier/<run>/...`
- 结果：（指标 /「进行中」）
- 下一步：
```
