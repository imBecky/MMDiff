# Agent 协作资料库（Bootstrap）

本目录是可提交进 Git 的**模板**。实际协作目录 **`agent-collab/`** 在仓库根 [.gitignore](.gitignore) 中被忽略，用于本机上下文与工作日志。

## 初次使用（新 clone）

在仓库根执行（任选其一）：

- Bash: `cp -r contrib/agent-collab-bootstrap agent-collab`
- PowerShell: `Copy-Item -Recurse contrib/agent-collab-bootstrap agent-collab`

之后仅在 `agent-collab/` 内维护日志与笔记；不要将密钥写进任意可提交路径。

## 必读（给 Agent）

1. **先读仓库根目录 [README.md](../../README.md)**：项目入口、`MMDIFF_*` 约定、目录结构与安全提示。
2. **再读本目录**[`worklog/README.md`](worklog/README.md)：如何追加工作日志。
3. **实验可追溯**：脚本里建议固定或显式导出 `MMDIFF_RUN_TIMESTAMP`，串行消融时配合 `MMDIFF_EXPERIMENT_TAG` / `MMDIFF_EXPERIMENT_NUM`，避免 TensorBoard run 目录互相覆盖（见 `pipeline/logging_utils.prepare_tb_run_dir`）。
4. **数据路径**：默认数据与产物路径指向 `../../autodl-fs/...`、`../../autodl-tmp/...`、`../../tf-logs`（见 `param.py`），按本机挂载调整。

## 可选子目录约定

可在 `agent-collab/` 下自建（按需）：

| 子目录       | 用途                     |
|-------------|--------------------------|
| `worklog/`  | 按日或专题工作日志（必用） |
| `decisions/`| ADR：架构/权衡结论       |
| `scratch/`  | 临时草稿，可随手删       |
