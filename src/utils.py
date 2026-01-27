"""
通用工具函数
"""

import os
import json
import time
import random
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed: int = 42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directory(dir_path: str):
    """创建目录，如果不存在"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_json(data: Dict, file_path: str):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def timeit(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper


def visualize_sample(image: np.ndarray, mask: np.ndarray, 
                    predictions: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
    """
    可视化样本、标注和预测
    
    Args:
        image: 原始图像 (H, W, 3)
        mask: 标注掩码 (H, W) 或 (3, H, W)
        predictions: 预测掩码，与mask同格式
        save_path: 保存路径，如果为None则显示
    """
    fig, axes = plt.subplots(1, 3 if predictions is None else 4, 
                            figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示标注
    if len(mask.shape) == 3 and mask.shape[0] == 3:
        # 如果是三条线的掩码，合并显示
        mask_combined = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(3):
            mask_combined[mask[i] > 0] = colors[i]
        axes[1].imshow(mask_combined)
        axes[1].set_title('标注（红:心线, 绿:头线, 蓝:生命线）')
    else:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('标注')
    axes[1].axis('off')
    
    if predictions is not None:
        # 显示预测
        if len(predictions.shape) == 3 and predictions.shape[0] == 3:
            pred_combined = np.zeros((predictions.shape[1], predictions.shape[2], 3), dtype=np.uint8)
            for i in range(3):
                pred_combined[predictions[i] > 0] = colors[i]
            axes[2].imshow(pred_combined)
            axes[2].set_title('预测（红:心线, 绿:头线, 蓝:生命线）')
        else:
            axes[2].imshow(predictions, cmap='gray')
            axes[2].set_title('预测')
        axes[2].axis('off')
        
        # 显示叠加结果
        overlay = image.copy()
        if len(predictions.shape) == 3:
            for i in range(3):
                if predictions[i].sum() > 0:
                    # 找到轮廓
                    contours, _ = cv2.findContours(
                        predictions[i].astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, colors[i], 2)
        else:
            contours, _ = cv2.findContours(
                predictions.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        axes[3].imshow(overlay)
        axes[3].set_title('叠加结果')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """计算多种评估指标"""
    # 确保是二值图像
    pred_bin = pred > 0
    target_bin = target > 0
    
    # 计算基本指标
    tp = np.logical_and(pred_bin, target_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(target_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), target_bin).sum()
    tn = np.logical_and(np.logical_not(pred_bin), np.logical_not(target_bin)).sum()
    
    # 计算指标
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = calculate_iou(pred_bin, target_bin)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'true_positive': int(tp),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_negative': int(tn)
    }


def plot_training_history(history: Dict[str, List], save_path: str = None):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_accuracy'], label='训练准确率')
    axes[0, 1].plot(history['val_accuracy'], label='验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('训练和验证准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线（如果有）
    if 'learning_rate' in history:
        axes[1, 0].plot(history['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 处理时间
    if 'train_time' in history:
        axes[1, 1].plot(history['train_time'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('时间（秒）')
        axes[1, 1].set_title('每轮训练时间')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def extract_lines_from_mask(mask: np.ndarray, min_length: int = 50) -> List[np.ndarray]:
    """
    从掩码中提取线条，确保每条线是连续的
    
    Args:
        mask: 二值掩码
        min_length: 最小线条长度
        
    Returns:
        线条列表，每个元素是一个二值掩码
    """
    # 找到所有连通分量
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    lines = []
    for label in range(1, num_labels):
        # 获取当前连通分量的掩码
        line_mask = (labels == label).astype(np.uint8)
        
        # 检查长度
        if line_mask.sum() >= min_length:
            lines.append(line_mask)
    
    # 按面积排序（从大到小）
    lines.sort(key=lambda x: x.sum(), reverse=True)
    
    return lines


def smooth_contour(contour: np.ndarray, epsilon_factor: float = 0.001) -> np.ndarray:
    """平滑轮廓"""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    smoothed = cv2.approxPolyDP(contour, epsilon, True)
    return smoothed


def get_current_time() -> str:
    """获取当前时间字符串"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_config(config: Any):
    """打印配置信息"""
    print("=" * 60)
    print("配置信息:")
    print("=" * 60)
    
    for key, value in vars(config).items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    print("=" * 60)


def check_gpu():
    """检查GPU可用性"""
    print("检查GPU可用性...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("警告: 没有可用的GPU，将使用CPU训练")
    
    print("=" * 60)