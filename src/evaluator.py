import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import Config
from .preprocessor import ImagePreprocessor


class PalmLineEvaluator:
    """掌纹识别评估器"""
    
    def __init__(self, config: Config, model: torch.nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = ImagePreprocessor(config.image_size)
        
        # 结果存储
        self.results = {
            'total': 0,
            'correct': 0,
            'failed': 0,
            'line_stats': {
                'heart': {'total': 0, 'correct': 0},
                'head': {'total': 0, 'correct': 0},
                'life': {'total': 0, 'correct': 0}
            },
            'metrics': {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'iou': []
            },
            'failed_cases': []
        }
    
    def evaluate(self, data_loader) -> Dict:
        """评估模型"""
        print("开始评估...")
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='评估'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 预测
                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5
                
                # 计算指标
                batch_results = self._evaluate_batch(preds, masks)
                self._update_results(batch_results)
                
                # 收集预测和标签用于总体指标
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        # 计算总体指标
        self._calculate_overall_metrics(all_preds, all_targets)
        
        return self.results
    
    def _evaluate_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict:
        """评估单个批次"""
        batch_size = preds.shape[0]
        batch_results = {
            'correct': 0,
            'failed': 0,
            'line_stats': {'heart': 0, 'head': 0, 'life': 0}
        }
        
        for i in range(batch_size):
            image_correct = True
            
            for line_idx, line_name in enumerate(['heart', 'head', 'life']):
                pred_line = preds[i, line_idx + 1]  # 跳过背景
                target_line = targets[i, line_idx + 1]
                
                # 计算指标
                metrics = self._calculate_line_metrics(pred_line, target_line)
                
                # 检查是否满足要求
                if (metrics['coverage'] >= self.config.coverage_threshold and 
                    metrics['false_positive_rate'] <= self.config.false_positive_threshold):
                    batch_results['line_stats'][line_name] += 1
                else:
                    image_correct = False
            
            if image_correct:
                batch_results['correct'] += 1
            else:
                batch_results['failed'] += 1
        
        return batch_results
    
    def _calculate_line_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict:
        """计算单条线的指标"""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # 计算交集和并集
        intersection = np.logical_and(pred_np, target_np).sum()
        union = np.logical_or(pred_np, target_np).sum()
        
        # 覆盖度（召回率）
        if target_np.sum() > 0:
            coverage = intersection / target_np.sum()
        else:
            coverage = 1.0 if pred_np.sum() == 0 else 0.0
        
        # 误检率
        if pred_np.sum() > 0:
            false_positive = np.logical_and(pred_np, np.logical_not(target_np)).sum()
            false_positive_rate = false_positive / pred_np.sum()
        else:
            false_positive_rate = 0.0
        
        # IoU
        iou = intersection / union if union > 0 else 0.0
        
        # 精度
        precision = intersection / pred_np.sum() if pred_np.sum() > 0 else 0.0
        
        return {
            'coverage': coverage,
            'false_positive_rate': false_positive_rate,
            'iou': iou,
            'precision': precision
        }
    
    def _update_results(self, batch_results: Dict):
        """更新总体结果"""
        self.results['total'] += batch_results['correct'] + batch_results['failed']
        self.results['correct'] += batch_results['correct']
        self.results['failed'] += batch_results['failed']
        
        for line_name in ['heart', 'head', 'life']:
            self.results['line_stats'][line_name]['total'] += 1
            self.results['line_stats'][line_name]['correct'] += batch_results['line_stats'][line_name]
    
    def _calculate_overall_metrics(self, all_preds: List, all_targets: List):
        """计算总体指标"""
        # 内存高效的方式计算指标 - 逐个处理批次
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        total_intersection = 0
        total_union = 0
        
        for batch_preds, batch_targets in zip(all_preds, all_targets):
            # 只考虑三条主线
            batch_preds_lines = batch_preds[:, 1:].reshape(-1)  # 跳过背景
            batch_targets_lines = batch_targets[:, 1:].reshape(-1)
            
            # 计算混淆矩阵元素
            true_positives = np.logical_and(batch_preds_lines, batch_targets_lines).sum()
            false_positives = np.logical_and(batch_preds_lines, np.logical_not(batch_targets_lines)).sum()
            false_negatives = np.logical_and(np.logical_not(batch_preds_lines), batch_targets_lines).sum()
            
            # 计算IoU相关值
            intersection = true_positives
            union = true_positives + false_positives + false_negatives
            
            # 累加
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives
            total_intersection += intersection
            total_union += union
        
        # 计算总体指标
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = total_intersection / total_union if total_union > 0 else 0.0
        
        self.results['metrics']['precision'].append(precision)
        self.results['metrics']['recall'].append(recall)
        self.results['metrics']['f1_score'].append(f1)
        self.results['metrics']['iou'].append(iou)
    
    def generate_report(self) -> str:
        """生成评估报告"""
        total = self.results['total']
        correct = self.results['correct']
        accuracy = correct / total if total > 0 else 0
        
        report = f"""
{'='*60}
掌纹识别评估报告
{'='*60}
评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总样本数: {total}
正确识别: {correct}
识别准确率: {accuracy:.2%}

各主线识别率:
  感情线 (Heart): {self.results['line_stats']['heart']['correct']}/{self.results['line_stats']['heart']['total']} = {self.results['line_stats']['heart']['correct']/self.results['line_stats']['heart']['total']:.2%}
  智慧线 (Head): {self.results['line_stats']['head']['correct']}/{self.results['line_stats']['head']['total']} = {self.results['line_stats']['head']['correct']/self.results['line_stats']['head']['total']:.2%}
  生命线 (Life): {self.results['line_stats']['life']['correct']}/{self.results['line_stats']['life']['total']} = {self.results['line_stats']['life']['correct']/self.results['line_stats']['life']['total']:.2%}

评估指标:
  平均精度 (Precision): {np.mean(self.results['metrics']['precision']):.4f}
  平均召回率 (Recall): {np.mean(self.results['metrics']['recall']):.4f}
  平均F1分数: {np.mean(self.results['metrics']['f1_score']):.4f}
  平均IoU: {np.mean(self.results['metrics']['iou']):.4f}

失败案例数: {len(self.results['failed_cases'])}
{'='*60}
"""
        return report
    
    def save_visualization(self, image: np.ndarray, predictions: np.ndarray, 
                          image_path: str, output_dir: str):
        """保存可视化结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取原始图像文件名
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # 绘制预测的线条
        overlay = image.copy()
        
        # 为每条线绘制不同颜色
        colors = self.config.colors
        line_names = ['Heart', 'Head', 'Life']
        
        for i in range(3):
            line_mask = predictions[i]
            if line_mask.sum() > 0:
                # 找到轮廓
                contours, _ = cv2.findContours(
                    line_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 绘制轮廓
                cv2.drawContours(overlay, contours, -1, colors[i], 
                               self.config.line_thickness)
                
                # 添加标签
                if len(contours) > 0:
                    M = cv2.moments(contours[0])
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(overlay, line_names[i], (cx, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{name}_result{ext}")
        cv2.imwrite(output_path, overlay)
        
        # 保存单独的掩码
        for i, line_name in enumerate(line_names):
            mask_output = os.path.join(output_dir, f"{name}_{line_name.lower()}_mask{ext}")
            cv2.imwrite(mask_output, predictions[i] * 255)
        
        return output_path