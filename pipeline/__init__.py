"""
`pipeline/`：数据读取与 DataLoader、epoch 循环、验证/测试、checkpoint、学生扩散封装。

模型通过 `runner.run_training(create_classifier)` 注入，`create_classifier(opt, diffusion)` 返回
具备 `loss_func`、`optimizer`、可选 `exp_lr_scheduler` 的 `nn.Module`（与当前 `MultimodalClassifier` 契约一致）。
"""

from .checkpoint import (
    ensure_model_path_parent,
    get_latest_checkpoint_path,
    save_classifier_checkpoint,
    save_classifier_training_state,
)
from .data import (
    IndexedTensorDataset,
    batch_to_dict,
    build_dataloaders,
    load_data,
    split_train_val,
    subset_train_balanced_per_class,
)
from .loop import compute_classification_loss, evaluate, train_one_epoch
from .metrics import accuracies
from .runner import run_training, verify_projection_gradients
from .student_diffusion import StudentDiffusionWrapper, normalize_student_checkpoint_dir

__all__ = [
    'IndexedTensorDataset',
    'StudentDiffusionWrapper',
    'accuracies',
    'batch_to_dict',
    'build_dataloaders',
    'compute_classification_loss',
    'ensure_model_path_parent',
    'evaluate',
    'get_latest_checkpoint_path',
    'load_data',
    'normalize_student_checkpoint_dir',
    'run_training',
    'save_classifier_checkpoint',
    'save_classifier_training_state',
    'split_train_val',
    'subset_train_balanced_per_class',
    'train_one_epoch',
    'verify_projection_gradients',
]
