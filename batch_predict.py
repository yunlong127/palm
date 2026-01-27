#!/usr/bin/env python3
"""
批量掌纹识别测试脚本
读取test文件夹中的所有图片，使用最佳模型进行预测，
并将结果保存到ans文件夹中。
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入U-Net模型定义
class DoubleConv(torch.nn.Module):
    """双卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(torch.nn.Module):
    """U-Net模型"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                 features: list = [64, 128, 256, 512]):
        super().__init__()
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # 解码器
        for feature in reversed(features):
            self.decoder.append(
                torch.nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # 最终卷积
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # 编码器
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # 调整尺寸以匹配
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], 
                    mode='bilinear', align_corners=True
                )
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class BatchPalmLinePredictor:
    """批量掌纹预测器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        print(f"预测器初始化完成，设备: {self.device}")
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        
        # 创建模型
        model = UNet(
            in_channels=3,
            out_channels=1,
            features=[32, 64, 128, 256, 512]
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"模型加载完成")
        
        # 保存图像尺寸信息
        if 'image_size' in checkpoint:
            self.image_size = checkpoint['image_size']
        else:
            self.image_size = (512, 512)
        
        return model
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 调整大小
        image_resized = cv2.resize(image, self.image_size)
        
        # 转换为float32
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        
        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return image_tensor, image_resized.shape[:2]
    
    def predict_image(self, image):
        """预测单张图像"""
        # 预处理
        image_tensor, _ = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        return pred_mask
    
    def create_overlay(self, image, pred_mask):
        """创建红线叠加图像"""
        # 调整掩码尺寸
        pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
        
        # 二值化
        binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
        
        # 创建叠加图像
        overlay = image.copy()
        
        if binary_mask is not None and np.sum(binary_mask) > 0:
            # 找到轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 用红色绘制所有轮廓
            for contour in contours:
                if len(contour) >= 2:
                    # 简化轮廓
                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 绘制线条
                    for i in range(len(approx) - 1):
                        pt1 = tuple(approx[i][0])
                        pt2 = tuple(approx[i+1][0])
                        cv2.line(overlay, pt1, pt2, (0, 0, 255), 2)  # 红色
        
        return overlay


def main():
    parser = argparse.ArgumentParser(description='批量掌纹识别测试脚本')
    parser.add_argument('--test-dir', type=str, default='test',
                       help='测试图片目录')
    parser.add_argument('--output-dir', type=str, default='ans',
                       help='输出结果目录')
    parser.add_argument('--model', type=str, default='checkpoints/best_three_lines_model.pth',
                       help='模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先运行训练脚本: python scripts/train_final.py")
        return
    
    # 检查测试目录
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"错误: 测试目录不存在: {args.test_dir}")
        print("请创建test目录并放入测试图片")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化预测器
    try:
        predictor = BatchPalmLinePredictor(args.model, args.device)
    except Exception as e:
        print(f"初始化预测器时出错: {e}")
        return
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(test_dir.glob(f'*{ext}')))
        image_files.extend(list(test_dir.glob(f'*{ext.upper()}')))
    
    if len(image_files) == 0:
        print(f"错误: 在 {args.test_dir} 中找不到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张测试图片")
    
    # 处理每张图片
    successful = 0
    failed = 0
    processed_files = []
    failed_files = []
    
    for img_path in tqdm(image_files, desc="处理图片"):
        try:
            # 检查是否已存在输出文件
            output_path = output_dir / f"{img_path.stem}_result.jpg"
            if args.skip_existing and output_path.exists():
                print(f"跳过已存在的文件: {output_path.name}")
                processed_files.append(img_path)
                successful += 1
                continue
            
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 无法读取图像 {img_path.name}，跳过")
                failed_files.append((img_path, "无法读取图像"))
                failed += 1
                continue
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 预测
            pred_mask = predictor.predict_image(image_rgb)
            
            # 创建叠加图像
            overlay = predictor.create_overlay(image, pred_mask)
            
            # 保存结果
            cv2.imwrite(str(output_path), overlay)
            
            # 同时保存原始预测掩码（可选）
            mask_output_path = output_dir / f"{img_path.stem}_mask.jpg"
            pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
            binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(mask_output_path), binary_mask)
            
            processed_files.append(img_path)
            successful += 1
            
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {e}")
            failed_files.append((img_path, str(e)))
            failed += 1
    
    # 生成统计报告
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"总图片数: {len(image_files)}")
    print(f"成功处理: {successful}")
    print(f"处理失败: {failed}")
    print(f"输出目录: {output_dir}")
    
    # 创建汇总文件
    summary_path = output_dir / "batch_summary.txt"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("批量掌纹识别结果汇总\n")
            f.write("=" * 50 + "\n")
            import time
            from datetime import datetime
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试目录: {test_dir}\n")
            f.write(f"输出目录: {output_dir}\n")
            f.write(f"模型路径: {args.model}\n")
            f.write(f"设备: {args.device}\n")
            f.write(f"总图片数: {len(image_files)}\n")
            f.write(f"成功处理: {successful}\n")
            f.write(f"处理失败: {failed}\n")
            f.write("\n处理的文件:\n")
            
            for img_path in processed_files:
                output_path = output_dir / f"{img_path.stem}_result.jpg"
                if output_path.exists():
                    f.write(f"[OK] {img_path.name} -> {output_path.name}\n")
            
            if failed_files:
                f.write("\n失败的文件:\n")
                for img_path, error_msg in failed_files:
                    f.write(f"[FAIL] {img_path.name} (错误: {error_msg[:50]}...)\n")
        
        print(f"\n汇总文件已保存: {summary_path}")
    except Exception as e:
        print(f"保存汇总文件时出错: {e}")
    
    # 创建预览图（选择前4张）
    if successful > 0:
        try:
            preview_images = []
            for img_path in image_files[:min(4, len(image_files))]:
                result_path = output_dir / f"{img_path.stem}_result.jpg"
                if result_path.exists():
                    preview_images.append(result_path)
            
            if len(preview_images) > 0:
                fig, axes = plt.subplots(1, len(preview_images), figsize=(15, 5))
                if len(preview_images) == 1:
                    axes = [axes]
                
                for i, img_path in enumerate(preview_images):
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(img_path.stem)
                    axes[i].axis('off')
                
                plt.tight_layout()
                preview_path = output_dir / "preview.jpg"
                plt.savefig(str(preview_path), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"预览图已保存: {preview_path}")
        except Exception as e:
            print(f"创建预览图时出错: {e}")
    
    # 显示结果文件列表
    print(f"\n生成的结果文件:")
    result_files = list(output_dir.glob("*_result.jpg"))
    for result_file in result_files:
        print(f"  {result_file.name}")


if __name__ == '__main__':
    main()