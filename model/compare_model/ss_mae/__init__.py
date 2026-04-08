"""SS-MAE（TGRS 2023）官方代码迁入本仓库，与 summitgao/SS-MAE 对齐。

- 结构与权重：见 `mae.py` / `vit.py`（来源 https://github.com/summitgao/SS-MAE ）
- 对比实验入口：`class SSMAEClassifier` in `model/compare_model/architectures.py`
"""
from .mae import MAEVisionTransformers, VisionTransfromers

__all__ = ['MAEVisionTransformers', 'VisionTransfromers']
