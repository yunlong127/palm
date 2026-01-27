import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json
import os
from PIL import Image
import gradio as gr

from .config import Config
from .preprocessor import ImagePreprocessor
from .models.unet import UNet
from .models.resunet import ResUNet


class PalmLinePredictor:
    """掌纹预测器"""
    
    def __init__(self, config: Config, model_path: str = None):
        self.config = config
        self.device = torch.device(config.device)
        self.preprocessor = ImagePreprocessor(config.image_size)
        
        # 加载模型
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
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"警告: 模型路径不存在: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载: {model_path}")
    
    def predict(self, image_path: str) -> Dict:
        """
        预测掌纹
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            包含预测结果和置信度的字典
        """
        start_time = time.time()
        
        try:
            # 预处理图像
            processed_image = self.preprocessor.process_image(image_path)
            
            # 转换为tensor
            image_tensor = self._preprocess_image(processed_image)
            
            # 预测
            with torch.no_grad():
                output = self.model(image_tensor)
                pred = torch.sigmoid(output) > 0.5
            
            # 后处理
            predictions = pred[0].cpu().numpy()
            
            # 提取三条主线
            lines = self._extract_main_lines(predictions[1:])  # 跳过背景
            
            # 计算置信度
            confidences = self._calculate_confidence(lines)
            
            # 生成结果
            result = {
                'success': True,
                'lines': lines,
                'confidences': confidences,
                'processing_time': time.time() - start_time,
                'error_message': None,
                'suggestions': []
            }
            
            # 检查是否所有线都检测到
            for line_name, confidence in confidences.items():
                if confidence < 0.3:
                    result['suggestions'].append(
                        f"{line_name}线置信度较低，建议重新拍摄"
                    )
            
        except Exception as e:
            result = {
                'success': False,
                'lines': None,
                'confidences': None,
                'processing_time': time.time() - start_time,
                'error_message': str(e),
                'suggestions': [
                    "请确保拍摄清晰的手掌照片",
                    "确保手掌完全在画面中",
                    "调整光照条件，避免过暗或过亮"
                ]
            }
        
        return result
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像为模型输入"""
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array(self.config.mean)
        std = np.array(self.config.std)
        image = (image - mean) / std
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _extract_main_lines(self, predictions: np.ndarray) -> List[np.ndarray]:
        """从预测中提取三条主线"""
        lines = []
        line_names = ['heart', 'head', 'life']
        
        for i in range(3):
            line_mask = predictions[i]
            
            # 骨架化
            skeleton = cv2.ximgproc.thinning(line_mask.astype(np.uint8))
            
            # 去除小连通分量
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                skeleton, connectivity=8
            )
            
            if num_labels > 1:
                # 找到最大的连通分量
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_idx = np.argmax(areas) + 1
                main_line = (labels == max_idx).astype(np.uint8)
            else:
                main_line = np.zeros_like(line_mask)
            
            lines.append(main_line)
        
        return lines
    
    def _calculate_confidence(self, lines: List[np.ndarray]) -> Dict[str, float]:
        """计算每条线的置信度"""
        confidences = {}
        line_names = ['heart', 'head', 'life']
        
        for i, line_name in enumerate(line_names):
            line_mask = lines[i]
            
            # 基于线长和连续性计算置信度
            if line_mask.sum() > 0:
                # 计算线的长度
                skeleton = cv2.ximgproc.thinning(line_mask.astype(np.uint8))
                length = skeleton.sum()
                
                # 计算连续性（通过端点数量）
                endpoints = self._count_endpoints(skeleton)
                continuity = 1.0 / (endpoints + 1) if endpoints > 0 else 1.0
                
                # 综合置信度
                confidence = min(continuity * 0.7 + (length / 1000) * 0.3, 1.0)
            else:
                confidence = 0.0
            
            confidences[line_name] = round(confidence, 2)
        
        return confidences
    
    def _count_endpoints(self, skeleton: np.ndarray) -> int:
        """计算骨架的端点数量"""
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # 卷积计算邻居数量
        neighbors = cv2.filter2D(skeleton, -1, kernel)
        
        # 端点是只有一个邻居的点
        endpoints = np.sum((skeleton > 0) & (neighbors == 11))
        
        return endpoints
    
    def visualize_result(self, image_path: str, result: Dict, 
                        output_dir: str = None) -> Tuple[np.ndarray, str]:
        """可视化预测结果"""
        if not result['success']:
            # 读取原始图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 添加错误信息
            error_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.putText(error_image, f"错误: {result['error_message']}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            return error_image, None
        
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 创建叠加图像
        overlay = image_rgb.copy()
        
        # 绘制线条
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红绿蓝
        line_names = ['Heart', 'Head', 'Life']
        
        for i, (line_mask, color, line_name) in enumerate(zip(
            result['lines'], colors, line_names)):
            
            if line_mask.sum() > 0:
                # 找到轮廓
                contours, _ = cv2.findContours(
                    line_mask.astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 绘制轮廓
                for contour in contours:
                    # 简化轮廓
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 绘制线条
                    if len(approx) > 1:
                        cv2.polylines(overlay, [approx], False, color, 3)
                
                # 添加置信度标签
                if len(contours) > 0:
                    M = cv2.moments(contours[0])
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        label = f"{line_name}: {result['confidences'][line_name.lower()]:.0%}"
                        cv2.putText(overlay, label, (cx, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 添加处理时间
        cv2.putText(overlay, f"处理时间: {result['processing_time']:.2f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_result{ext}")
            
            # 保存为BGR格式
            output_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_bgr)
            
            # 保存JSON结果
            json_path = os.path.join(output_dir, f"{name}_result.json")
            self._save_json_result(result, json_path)
            
            return overlay, output_path
        
        return overlay, None
    
    def _save_json_result(self, result: Dict, json_path: str):
        """保存JSON格式的结果"""
        json_result = {
            'success': result['success'],
            'confidences': result['confidences'],
            'processing_time': result['processing_time'],
            'error_message': result['error_message'],
            'suggestions': result['suggestions']
        }
        
        # 如果有线条数据，转换为点集
        if result['lines']:
            lines_data = []
            line_names = ['heart', 'head', 'life']
            
            for i, line_name in enumerate(line_names):
                line_mask = result['lines'][i]
                if line_mask.sum() > 0:
                    # 找到轮廓并提取点
                    contours, _ = cv2.findContours(
                        line_mask.astype(np.uint8) * 255,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        # 取最大的轮廓
                        main_contour = max(contours, key=cv2.contourArea)
                        points = main_contour.squeeze().tolist()
                        
                        lines_data.append({
                            'name': line_name,
                            'points': points,
                            'confidence': result['confidences'][line_name]
                        })
            
            json_result['lines'] = lines_data
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)


class GradioApp:
    """Gradio Web界面"""
    
    def __init__(self, predictor: PalmLinePredictor):
        self.predictor = predictor
        
    def create_interface(self):
        """创建Gradio界面"""
        
        def process_image(input_image, use_advanced_preprocessing):
            # 临时保存图像
            temp_path = "temp_input.jpg"
            input_image.save(temp_path)
            
            # 预测
            result = self.predictor.predict(temp_path)
            
            # 可视化
            overlay, output_path = self.predictor.visualize_result(
                temp_path, result, self.predictor.config.output_dir
            )
            
            # 生成结果文本
            if result['success']:
                result_text = f"✅ 识别成功！\n\n"
                result_text += f"📊 置信度:\n"
                for line_name, confidence in result['confidences'].items():
                    result_text += f"  • {line_name.capitalize()}线: {confidence:.0%}\n"
                result_text += f"\n⏱️ 处理时间: {result['processing_time']:.2f}秒"
                
                if result['suggestions']:
                    result_text += f"\n\n💡 建议:\n"
                    for suggestion in result['suggestions']:
                        result_text += f"  • {suggestion}\n"
            else:
                result_text = f"❌ 识别失败！\n\n"
                result_text += f"错误信息: {result['error_message']}\n\n"
                result_text += f"💡 建议:\n"
                for suggestion in result['suggestions']:
                    result_text += f"  • {suggestion}\n"
            
            return overlay, result_text
        
        # 创建界面
        with gr.Blocks(title="掌纹识别系统", theme=gr.themes.Soft()) as app:
            gr.Markdown("# ✋ 手掌掌纹识别系统")
            gr.Markdown("上传手掌照片，自动识别三条主线：感情线、智慧线、生命线")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="上传手掌照片",
                        type="pil",
                        height=400
                    )
                    
                    advanced_checkbox = gr.Checkbox(
                        label="使用高级预处理",
                        value=True
                    )
                    
                    process_btn = gr.Button("识别掌纹", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(
                        label="识别结果",
                        height=400
                    )
                    
                    output_text = gr.Textbox(
                        label="识别结果详情",
                        lines=10
                    )
            
            # 示例
            gr.Markdown("### 📸 拍摄建议")
            gr.Markdown("""
            1. 确保手掌完全在画面中
            2. 手心朝上，手指自然分开
            3. 光线充足但避免反光
            4. 背景尽量简单
            5. 图像清晰，不模糊
            """)
            
            # 绑定事件
            process_btn.click(
                fn=process_image,
                inputs=[input_image, advanced_checkbox],
                outputs=[output_image, output_text]
            )
        
        return app