import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Tuple, List, Optional
import time
from tqdm import tqdm
import os
from datetime import datetime

from .config import Config
from .models.unet import UNet
from .models.resunet import ResUNet


class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        
        # 展平
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal损失函数，处理类别不平衡"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(predictions, targets)
        
        # Focal loss
        p_t = torch.exp(-bce)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失函数：Dice + Focal"""
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


class Trainer:
    """训练器类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化模型
        if config.model_name == "unet":
            self.model = UNet(
                in_channels=config.in_channels,
                out_channels=config.out_channels
            )
        elif config.model_name == "resunet":
            self.model = ResUNet(
                in_channels=config.in_channels,
                out_channels=config.out_channels
            )
        else:
            raise ValueError(f"未知模型: {config.model_name}")
        
        self.model.to(self.device)
        
        # 损失函数
        if config.use_dice_loss and config.use_focal_loss:
            self.criterion = CombinedLoss()
        elif config.use_dice_loss:
            self.criterion = DiceLoss()
        elif config.use_focal_loss:
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # 早停
        self.early_stopping_patience = config.patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.tensorboard_dir)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_time': []
        }
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        batch_times = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 后向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                targets = masks > 0.5
                
                # 计算符合要求的准确率
                batch_correct, batch_pixels = self._calculate_accuracy(preds, targets)
                total_correct += batch_correct
                total_pixels += batch_pixels
                batch_accuracy = batch_correct / batch_pixels if batch_pixels > 0 else 0
            
            total_loss += loss.item()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': f'{batch_accuracy:.2%}',
                'time': f'{batch_time:.2f}s'
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            self.writer.add_scalar('Train/batch_accuracy', batch_accuracy, global_step)
            self.writer.add_scalar('Train/batch_time', batch_time, global_step)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
        avg_batch_time = np.mean(batch_times)
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_accuracy'].append(epoch_accuracy)
        self.history['train_time'].append(avg_batch_time)
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 计算准确率
                preds = torch.sigmoid(outputs) > 0.5
                targets = masks > 0.5
                
                batch_correct, batch_pixels = self._calculate_accuracy(preds, targets)
                total_correct += batch_correct
                total_pixels += batch_pixels
                
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        val_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
        
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_accuracy)
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/loss', val_loss, epoch)
        self.writer.add_scalar('Val/accuracy', val_accuracy, epoch)
        
        return val_loss, val_accuracy
    
    def _calculate_accuracy(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
        """
        计算符合要求的准确率
        
        标准：
        1. 处理结果覆盖对应标注答案的点不低于80%
        2. 处理结果落在标注答案外的点不多于处理结果的20%
        """
        batch_size = preds.shape[0]
        total_correct = 0
        
        for i in range(batch_size):
            for line_idx in range(self.config.out_channels - 1):  # 忽略背景
                pred_line = preds[i, line_idx + 1]
                target_line = targets[i, line_idx + 1]
                
                if pred_line.sum() == 0 and target_line.sum() == 0:
                    total_correct += 1
                    continue
                
                # 计算覆盖度
                intersection = (pred_line & target_line).sum().item()
                union = (pred_line | target_line).sum().item()
                
                if target_line.sum() > 0:
                    coverage = intersection / target_line.sum().item()
                else:
                    coverage = 1.0 if pred_line.sum() == 0 else 0.0
                
                # 计算误检率
                false_positive = (pred_line & ~target_line).sum().item()
                if pred_line.sum() > 0:
                    fp_rate = false_positive / pred_line.sum().item()
                else:
                    fp_rate = 0.0
                
                # 检查是否满足要求
                if (coverage >= self.config.coverage_threshold and 
                    fp_rate <= self.config.false_positive_threshold):
                    total_correct += 1
        
        total_pixels = batch_size * (self.config.out_channels - 1)
        
        return total_correct, total_pixels
    
    def save_checkpoint(self, epoch: int, val_loss: float, filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pth'
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config
        }, checkpoint_path)
        
        print(f'检查点保存到: {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练循环"""
        print(f"开始训练，使用设备: {self.device}")
        print(f"模型: {self.config.model_name}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, epoch)
            print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, 'best_model.pth')
                self.patience_counter = 0
                print(f"最佳模型更新，验证损失: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"早停计数: {self.patience_counter}/{self.early_stopping_patience}")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                print(f"早停触发，在epoch {epoch}")
                break
        
        # 保存最终模型
        self.save_checkpoint(epoch, val_loss, 'final_model.pth')
        
        # 关闭TensorBoard
        self.writer.close()
        
        return self.history