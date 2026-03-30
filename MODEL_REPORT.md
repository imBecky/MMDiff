# 多模态分类器结构报告（自动生成）
- 生成时间：2026-03-30T10:56:04
- 工作目录：`E:\MMDiff`

## 1. 配置摘要（来自 `param.py` / 环境变量覆盖后）
当前使用conda activate hbq环境，但是最终代码是会部署到Linux服务器上使用的，不需要在windows环境里面跑。
所有bash文件背后都还要写一样的关机和提示代码，以防跑完代码没关机浪费租金。
### 1.1 数据集与模态
| 项 | 值 |
| --- | --- |
| 类别数 `n_cls` | 20 |
| HSI 通道 `hsi_channels` | 50 |
| LiDAR 通道 `lidar_channel` | 1 |
| 模态 `modalities` | `['hsi', 'lidar']` |

### 1.2 分类头 `model_cls`
| 项 | 值 |
| --- | --- |
| 扩散时间步 `t` | `[50]` |
| 特征层 `feat_scales` | `['down_blocks.1', 'mid_block', 'up_blocks.1']` |
| `token_dim` | 256 |
| `transformer_heads` | 4 |
| `transformer_layers` | 2 |
| `transformer_ff_dim` | 512 |
| `transformer_dropout` | 0.1 |
| `head_hidden` | 128 |
| `use_supcon` | False |
| `supcon_proj_dim` | 128 |

### 1.3 投影与 HSI/LiDAR `module_cast3`
| 项 | 值 |
| --- | --- |
| `lidar_hidden` | 16 |
| `lidar_extra_blocks` | 0 |
| `hsi_residual_blocks` | 3 |
| `hsi_conv_hidden` | 96 |
| `hsi_se_ratio` | 8 |
| `hsi_agg_mode` | `attn_pool` |

### 1.4 学生扩散（冻结）
| 项 | 值 |
| --- | --- |
| `STUDENT_CHECKPOINT` | `..\..\autodl-fs\student32\final` |
| `STUDENT_SIZE` | 32 |
| `STUDENT_CHANNELS` | `(128, 256, 512, 512)` |
| `FEAT_SCALES`（与 `feat_scales` 对齐） | `['down_blocks.1', 'mid_block', 'up_blocks.1']` |

> **注意**：本地未找到学生扩散目录 `..\..\autodl-fs\student32\final`，无法实例化 UNet 探测各层通道数；在服务器上重新运行本脚本可得到完整报告。
