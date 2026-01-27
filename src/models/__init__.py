"""
掌纹识别项目 - 主包
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# 导入常用模块 - 简化版本，避免导入问题
from .config import Config
# 注释掉有问题的导入
# from .data_loader import PLSUDataset, create_dataloaders
from .preprocessor import ImagePreprocessor
from .trainer import Trainer, DiceLoss, FocalLoss, CombinedLoss
from .evaluator import PalmLineEvaluator
from .predictor import PalmLinePredictor, GradioApp
from .utils import (
    set_seed,
    create_directory,
    save_json,
    load_json,
    timeit,
    visualize_sample,
    calculate_iou,
    calculate_metrics,
    plot_training_history,
    extract_lines_from_mask,
    smooth_contour,
    get_current_time,
    print_config,
    check_gpu
)

# 导入模型
from .models import UNet, ResUNet

# 定义__all__以便于from src import *
__all__ = [
    'Config',
    # 'PLSUDataset',  # 注释掉
    # 'create_dataloaders',  # 注释掉
    'ImagePreprocessor',
    'Trainer',
    'DiceLoss',
    'FocalLoss',
    'CombinedLoss',
    'PalmLineEvaluator',
    'PalmLinePredictor',
    'GradioApp',
    'UNet',
    'ResUNet',
    'set_seed',
    'create_directory',
    'save_json',
    'load_json',
    'timeit',
    'visualize_sample',
    'calculate_iou',
    'calculate_metrics',
    'plot_training_history',
    'extract_lines_from_mask',
    'smooth_contour',
    'get_current_time',
    'print_config',
    'check_gpu'
]