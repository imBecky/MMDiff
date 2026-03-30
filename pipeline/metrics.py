"""兼容旧路径：`from pipeline.metrics import accuracies`。实现见 classification_metrics。"""
from .classification_metrics import accuracies

__all__ = ['accuracies']
