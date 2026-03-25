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
    PatchDataset,
    batch_to_dict,
    build_dataloaders,
    build_test_loader,
    load_test_indices_shifted,
    load_train_bundle,
    split_train_val_indices,
    subset_train_indices_balanced,
)
from .loop import compute_classification_loss, evaluate, train_one_epoch
from .metrics import accuracies
from .runner import TrainingRunOptions, run_training, verify_projection_gradients
from .student_diffusion import StudentDiffusionWrapper, normalize_student_checkpoint_dir

__all__ = [
    'PatchDataset',
    'StudentDiffusionWrapper',
    'accuracies',
    'batch_to_dict',
    'compute_classification_loss',
    'ensure_model_path_parent',
    'evaluate',
    'get_latest_checkpoint_path',
    'build_dataloaders',
    'build_test_loader',
    'load_test_indices_shifted',
    'load_train_bundle',
    'normalize_student_checkpoint_dir',
    'TrainingRunOptions',
    'run_training',
    'save_classifier_checkpoint',
    'save_classifier_training_state',
    'split_train_val_indices',
    'subset_train_indices_balanced',
    'train_one_epoch',
    'verify_projection_gradients',
]
