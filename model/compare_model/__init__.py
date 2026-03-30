"""对比实验模型包：与 pipeline.run_training(create_classifier) 兼容。"""
from .registry import REGISTRY, create_compare_classifier, resolve_compare_model_name

__all__ = ['REGISTRY', 'create_compare_classifier', 'resolve_compare_model_name']
