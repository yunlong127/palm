#!/usr/bin/env python3
"""
图像处理器 - 封装深度学习模型的预测功能
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from scripts.predict_final import PalmLinePredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("警告: 无法导入预测模块，请确保模型已训练")


@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    overlay_image: Optional[np.ndarray] = None
    confidences: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    suggestions: List[str] = None
    image_size: Optional[Tuple[int, int]] = None
    lines_data: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predictor = None
        self.model_loaded = False
        
        # 默认模型路径
        self.model_paths = {
            'unet': 'checkpoints/best_three_lines_model.pth',
            'resunet': 'checkpoints/best_resunet_model.pth'
        }
    
    def load_model(self) -> bool:
        """加载模型"""
        if not MODEL_AVAILABLE:
            return False
        
        try:
            model_type = self.config.get('model_type', 'unet')
            model_path = self.model_paths.get(model_type)
            
            if not os.path.exists(model_path):
                # 尝试其他可能的位置
                alt_path = Path("checkpoints") / "best_model.pth"
                if alt_path.exists():
                    model_path = str(alt_path)
                else:
                    return False
            
            device = 'cuda' if self.config.get('use_gpu', True) else 'cpu'
            self.predictor = PalmLinePredictor(model_path, device)
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def process_image(self, image_path: str) -> ProcessResult:
        """处理单张图像"""
        start_time = time.time()
        
        try:
            # 检查文件
            if not os.path.exists(image_path):
                error_msg = f"图片文件不存在: {image_path}"
                print(error_msg)
                return ProcessResult(
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message=error_msg,
                    suggestions=["请检查文件路径是否正确", "确保文件没有被移动或删除"]
                )
            
            # 加载模型（如果尚未加载）
            if not self.model_loaded:
                if not self.load_model():
                    return ProcessResult(
                        success=False,
                        processing_time=time.time() - start_time,
                        error_message="模型加载失败",
                        suggestions=["请确保模型文件存在", "运行训练脚本生成模型"]
                    )
            
            # 读取图像
            print(f"尝试读取图片: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                # 尝试使用numpy读取
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    print("使用PIL成功读取图片")
                except Exception as e:
                    error_msg = f"无法读取图片: {image_path}, 错误: {e}"
                    print(error_msg)
                    return ProcessResult(
                        success=False,
                        processing_time=time.time() - start_time,
                        error_message=error_msg,
                        suggestions=["请检查图片格式是否支持", "尝试使用其他图片", "确保文件路径正确"]
                    )
            
            original_size = image.shape[:2]
            
            # 预处理图像
            processed_image = self.preprocess_image(image)
            
            # 查找对应的标注答案（用于评估）
            mask_path = self.find_mask_path(image_path)
            
            # 使用预测器进行预测
            print(f"开始预测: {image_path}")
            results = self.predictor.predict_single_image(image_path, mask_path)
            print(f"预测完成，结果键: {results.keys()}")
            
            # 计算处理时间
            processing_time = time.time() - start_time
            print(f"处理时间: {processing_time:.2f}秒")
            
            # 准备结果
            confidences = self.calculate_confidences(results)
            print(f"置信度: {confidences}")
            
            suggestions = self.generate_suggestions(results, confidences)
            
            # 提取线条数据
            lines_data = self.extract_lines_data(results)
            
            return ProcessResult(
                success=True,
                overlay_image=results.get('overlay'),
                confidences=confidences,
                processing_time=processing_time,
                suggestions=suggestions,
                image_size=original_size,
                lines_data=lines_data
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessResult(
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                suggestions=self.get_error_suggestions(e)
            )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 处理EXIF方向
        image = self.fix_image_orientation(image)
        
        # 如果启用增强
        if self.config.get('enhance_image', True):
            image = self.enhance_image(image)
        
        return image
    
    def fix_image_orientation(self, image: np.ndarray) -> np.ndarray:
        """修正图像方向（简化的EXIF处理）"""
        # 这里可以添加更完善的EXIF处理
        # 目前只处理简单的旋转
        height, width = image.shape[:2]
        if width > height:
            # 可能是横向图片，旋转为纵向
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        return image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """增强图像对比度"""
        # 转换为YUV颜色空间
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # 对亮度通道进行CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 轻微锐化
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def find_mask_path(self, image_path: str) -> Optional[str]:
        """查找对应的标注答案路径"""
        image_path = Path(image_path)
        
        # 尝试在常见的标注目录中查找
        possible_dirs = [
            Path("PLSU") / "Mask",
            Path("mask"),
            Path("masks"),
            image_path.parent / "Mask"
        ]
        
        possible_names = [
            image_path.stem + '.png',
            image_path.stem + '.jpg',
            image_path.name.replace('.jpg', '.png').replace('.jpeg', '.png'),
        ]
        
        for mask_dir in possible_dirs:
            if mask_dir.exists():
                for mask_name in possible_names:
                    mask_path = mask_dir / mask_name
                    if mask_path.exists():
                        return str(mask_path)
        
        return None
    
    def calculate_confidences(self, results: Dict) -> Dict[str, float]:
        """计算置信度"""
        confidences = {}
        
        # 基于掌纹面积与手掌面积的比值计算置信度
        total_confidence = self.calculate_area_ratio_confidence(results)
        
        # 只输出总置信度
        total_confidence = min(max(total_confidence, 0), 1)  # 限制在0-1之间
        confidences['total'] = round(total_confidence, 2)
        
        return confidences
    
    def calculate_continuity(self, mask: np.ndarray) -> float:
        """计算掩码的连续性"""
        if mask is None or mask.sum() == 0:
            return 0.0
        
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 计算最大轮廓的连续性
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # 面积与周长的比值可以反映连续性
        area = cv2.contourArea(max_contour)
        continuity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return min(max(continuity, 0), 1)
    
    def calculate_area_ratio_confidence(self, results: Dict) -> float:
        """根据掌纹面积与手掌面积的比值计算置信度"""
        # 获取预测掩码
        pred_mask = results.get('full_prediction')
        if pred_mask is None:
            return 0.3  # 默认较低置信度
        
        # 二值化掩码
        _, binary_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 计算掌纹面积（掩码中白色像素的数量）
        palm_line_area = cv2.countNonZero(binary_mask)
        
        # 计算手掌面积（通过寻找手掌轮廓）
        hand_area = self.calculate_hand_area(binary_mask)
        
        if hand_area == 0:
            return 0.2  # 如果无法计算手掌面积，返回较低置信度
        
        # 计算比值
        area_ratio = palm_line_area / hand_area
        
        # 基于比值计算置信度
        # 理想的比值范围是0.1-0.3，在此范围内置信度较高
        if 0.1 <= area_ratio <= 0.3:
            # 在理想范围内，置信度随比值增加而增加
            confidence = 0.7 + (area_ratio - 0.1) * 1.5
        elif area_ratio < 0.1:
            # 比值过小，置信度较低
            confidence = area_ratio * 7
        else:
            # 比值过大，可能是噪声
            confidence = 1.0 - (area_ratio - 0.3) * 1.5
        
        return confidence
    
    def calculate_hand_area(self, binary_mask: np.ndarray) -> int:
        """计算手掌面积"""
        # 对掩码进行形态学操作，填充小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # 找到最大的轮廓作为手掌
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的边界矩形面积作为手掌面积
        x, y, w, h = cv2.boundingRect(max_contour)
        hand_area = w * h
        
        return hand_area
    
    def generate_suggestions(self, results: Dict, confidences: Dict) -> List[str]:
        """生成建议"""
        suggestions = []
        confidence_threshold = self.config.get('confidence_threshold', 0.3)
        
        # 检查总置信度
        total_confidence = confidences.get('total', 0)
        if total_confidence < confidence_threshold:
            suggestions.append("掌纹识别置信度较低，建议重新拍摄")
        else:
            suggestions.append("识别成功，掌纹清晰")
        
        # 通用建议
        suggestions.append("保持手掌稳定，避免模糊")
        suggestions.append("确保光照均匀，避免阴影")
        
        return suggestions
    
    def extract_lines_data(self, results: Dict) -> List[Dict]:
        """提取线条数据"""
        lines_data = []
        
        # 从预测结果中提取线条轮廓
        pred_mask = results.get('full_prediction')
        if pred_mask is None:
            return lines_data
        
        # 二值化
        _, binary = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 找到所有轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return lines_data
        
        # 按面积排序，取最大的三个作为三条主线
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        
        line_names = ['heart', 'head', 'life']
        for i, (contour, line_name) in enumerate(zip(contours, line_names)):
            # 简化轮廓
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 提取点集
            points = approx.squeeze().tolist()
            if isinstance(points[0], float):
                points = [points]  # 单个点的情况
            
            lines_data.append({
                'name': line_name,
                'points': points,
                'length': len(points),
                'confidence': 0.8 - i * 0.1  # 简单的置信度分配
            })
        
        return lines_data
    
    def get_error_suggestions(self, error: Exception) -> List[str]:
        """根据错误类型生成建议"""
        error_msg = str(error).lower()
        
        if 'cuda' in error_msg or 'gpu' in error_msg:
            return [
                "GPU内存不足，尝试使用CPU模式",
                "关闭其他占用GPU的程序",
                "尝试处理更小的图片"
            ]
        elif 'memory' in error_msg:
            return [
                "内存不足，尝试关闭其他程序",
                "尝试处理更小的图片",
                "重启应用程序"
            ]
        elif 'file' in error_msg or 'path' in error_msg:
            return [
                "检查文件路径是否正确",
                "确保文件没有被移动或删除",
                "尝试使用绝对路径"
            ]
        else:
            return [
                "请重新拍摄清晰的手掌照片",
                "确保手掌完全在画面中",
                "调整光照条件，避免过暗或过亮",
                "联系技术支持"
            ]