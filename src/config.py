import os
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch

@dataclass
class Config:
    """项目配置类"""
    
    # 路径配置
    data_root: str = "PLSU"
    image_dir: str = "img"
    mask_dir: str = "Mask"
    output_dir: str = f"ans-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 数据集划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 图像预处理
    image_size: Tuple[int, int] = (512, 512)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 训练配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15
    
    # 模型配置
    model_name: str = "resunet"  # "unet" or "resunet"
    in_channels: int = 3
    out_channels: int = 4  # background + 3 lines
    encoder_depth: int = 5
    decoder_channels: List[int] = (256, 128, 64, 32, 16)
    
    # 损失函数
    use_dice_loss: bool = True
    use_focal_loss: bool = True
    
    # 评估指标
    iou_threshold: float = 0.5
    coverage_threshold: float = 0.8  # 覆盖标注点不低于80%
    false_positive_threshold: float = 0.2  # 误检点不超过20%
    
    # 后处理
    min_line_length: int = 50
    max_gap_length: int = 10
    line_thickness: int = 3
    colors: List[Tuple[int, int, int]] = (
        (255, 0, 0),    # Red for Heart line
        (0, 255, 0),    # Green for Head line
        (0, 0, 255),    # Blue for Life line
    )
    
    # 保存配置
    save_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tensorboard_dir: str = "runs"
    
    def __post_init__(self):
        """初始化后创建必要目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)