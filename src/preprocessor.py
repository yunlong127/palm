import cv2
import numpy as np
from typing import Tuple, List, Optional
import torch
from PIL import Image, ImageOps
import exifread


class ImagePreprocessor:
    """图像预处理类，处理EXIF、方向、ROI等"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        
    def process_image(self, image_path: str) -> np.ndarray:
        """
        处理输入图像：读取、处理EXIF、调整方向
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理后的图像
        """
        # 读取图像并处理EXIF方向
        image = self._read_image_with_exif(image_path)
        
        # 检测手掌区域
        roi = self._detect_palm_roi(image)
        
        # 裁剪并调整大小
        if roi is not None:
            x, y, w, h = roi
            cropped = image[y:y+h, x:x+w]
            processed = cv2.resize(cropped, self.target_size)
        else:
            # 如果没有检测到手掌，使用整个图像
            processed = cv2.resize(image, self.target_size)
        
        # 增强对比度
        processed = self._enhance_contrast(processed)
        
        return processed
    
    def _read_image_with_exif(self, image_path: str) -> np.ndarray:
        """读取图像并处理EXIF方向"""
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            # 使用PIL作为备用
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 根据EXIF标签旋转图像
        if 'Image Orientation' in tags:
            orientation = tags['Image Orientation'].values[0]
            
            if orientation == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif orientation == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return image
    
    def _detect_palm_roi(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测手掌区域
        
        Args:
            image: 输入图像
            
        Returns:
            (x, y, width, height) 或 None
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 肤色检测范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的轮廓（假设是手掌）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 扩展边界框
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        return (x, y, w, h)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强图像对比度，突出掌纹"""
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE（对比度受限的自适应直方图均衡）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 边缘增强
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # 合并回BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def extract_palm_lines(self, image: np.ndarray) -> np.ndarray:
        """
        提取掌纹线条
        
        Args:
            image: 输入图像
            
        Returns:
            掌纹线条掩码
        """
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Frangi滤波器增强线条
        lines = self._frangi_filter(gray)
        
        # 二值化
        _, binary = cv2.threshold(lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 细化
        skeleton = cv2.ximgproc.thinning(binary)
        
        # 去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _frangi_filter(self, image: np.ndarray) -> np.ndarray:
        """Frangi滤波器，用于增强线条结构"""
        # 实现简化的Frangi滤波器
        # 计算Hessian矩阵的特征值
        scale = 1.0
        beta = 0.5
        c = 0.5
        
        # 计算梯度
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算二阶导数
        dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
        dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算特征值
        lambda1 = 0.5 * (dxx + dyy + np.sqrt((dxx - dyy)**2 + 4 * dxy**2))
        lambda2 = 0.5 * (dxx + dyy - np.sqrt((dxx - dyy)**2 + 4 * dxy**2))
        
        # 计算Frangi响应
        Rb = (lambda2 / (lambda1 + 1e-6))**2
        S = np.sqrt(lambda1**2 + lambda2**2)
        
        # 抑制非线条区域
        response = np.exp(-Rb / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * c**2)))
        response[lambda1 < 0] = 0
        
        # 归一化
        response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
        
        return response.astype(np.uint8)