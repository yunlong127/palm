#!/usr/bin/env python3
"""
掌纹识别预测脚本 - 显示原始图像、标注答案和预测结果
"""

import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# 1. U-Net 模型定义
# ============================================================================
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


# ============================================================================
# 2. 预测器类
# ============================================================================
class PalmLinePredictor:
    """掌纹预测器"""
    
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
        if 'epoch' in checkpoint:
            print(f"训练轮次: {checkpoint['epoch']}")
        if 'iou' in checkpoint:
            print(f"验证IoU: {checkpoint['iou']:.4f}")
        
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
    
    def predict_single_image(self, image_path, mask_path=None):
        """预测单张图像"""
        # 读取图像 - 使用PIL来处理中文路径
        try:
            from PIL import Image
            pil_image = Image.open(str(image_path))
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 如果PIL失败，尝试cv2
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}, 错误: {e}")
        
        original_shape = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        image_tensor, _ = self.preprocess_image(image_rgb)
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # 调整回原始尺寸
        pred_mask_resized = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
        
        # 二值化
        binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255
        
        # 创建叠加图像（只用红色显示完整预测）
        overlay = self.create_simple_overlay(image, binary_mask)
        
        # 读取标注答案（如果提供）
        ground_truth = None
        if mask_path and os.path.exists(mask_path):
            # 使用PIL读取标注答案
            try:
                from PIL import Image
                pil_mask = Image.open(str(mask_path)).convert('L')
                ground_truth = np.array(pil_mask)
                # 调整到原始图像大小
                ground_truth = cv2.resize(ground_truth, (original_shape[1], original_shape[0]))
            except Exception as e:
                print(f"无法读取标注答案: {mask_path}, 错误: {e}")
        
        # 计算IoU（如果有标注答案）
        iou = None
        if ground_truth is not None and binary_mask is not None:
            # 二值化标注答案
            gt_binary = (ground_truth > 127).astype(np.uint8)
            pred_binary = (binary_mask > 127).astype(np.uint8)
            
            # 计算IoU
            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()
            
            if union > 0:
                iou = intersection / union
        
        return {
            'image': image,
            'overlay': overlay,
            'full_prediction': binary_mask,
            'ground_truth': ground_truth,
            'iou': iou,
            'prediction_prob': pred_mask_resized
        }
    
    def create_simple_overlay(self, image, binary_mask):
        """创建简单叠加图像 - 只用红色显示完整预测"""
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
    
    def visualize_comparison(self, results, save_path=None):
        """可视化比较：原始图像、标注答案、预测叠加结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 标注答案（如果有）
        if results['ground_truth'] is not None:
            axes[1].imshow(results['ground_truth'], cmap='gray')
            axes[1].set_title('标注答案')
        else:
            axes[1].text(0.5, 0.5, '无标注答案', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1].transAxes,
                        fontsize=12)
            axes[1].set_title('标注答案（未找到）')
        axes[1].axis('off')
        
        # 预测叠加结果
        axes[2].imshow(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB))
        
        # 添加IoU信息
        title = '预测结果叠加'
        if results['iou'] is not None:
            title += f'\nIoU: {results["iou"]:.3f}'
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================================
# 3. 主程序
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='掌纹识别预测 - 显示原始图像、标注答案和预测结果')
    parser.add_argument('--model', type=str, default='checkpoints/best_three_lines_model.pth',
                       help='模型路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--mask-dir', type=str, default='PLSU/Mask',
                       help='标注答案目录')
    parser.add_argument('--output', type=str, default='comparison_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='显示可视化结果')
    
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
    
    # 初始化预测器
    try:
        predictor = PalmLinePredictor(args.model, args.device)
    except Exception as e:
        print(f"初始化预测器时出错: {e}")
        return
    
    # 处理输入
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理单张图像
    if input_path.is_file():
        try:
            print(f"处理图像: {input_path.name}")
            
            # 查找对应的标注答案
            mask_path = None
            if os.path.exists(args.mask_dir):
                # 尝试不同的文件名模式
                possible_names = [
                    input_path.stem + '.png',
                    input_path.stem + '.jpg',
                    input_path.name.replace('.jpg', '.png').replace('.jpeg', '.png'),
                ]
                
                for mask_name in possible_names:
                    potential_path = Path(args.mask_dir) / mask_name
                    if potential_path.exists():
                        mask_path = str(potential_path)
                        print(f"找到标注答案: {mask_path}")
                        break
            
            # 预测
            results = predictor.predict_single_image(str(input_path), mask_path)
            
            # 保存结果
            filename = input_path.stem
            
            # 1. 保存原始图像
            original_path = output_dir / f"{filename}_original.jpg"
            cv2.imwrite(str(original_path), results['image'])
            
            # 2. 保存预测叠加图
            overlay_path = output_dir / f"{filename}_prediction_overlay.jpg"
            cv2.imwrite(str(overlay_path), results['overlay'])
            
            # 3. 保存完整预测（二值掩码）
            prediction_path = output_dir / f"{filename}_prediction_binary.jpg"
            cv2.imwrite(str(prediction_path), results['full_prediction'])
            
            # 4. 保存标注答案（如果有）
            if results['ground_truth'] is not None:
                gt_path = output_dir / f"{filename}_ground_truth.jpg"
                cv2.imwrite(str(gt_path), results['ground_truth'])
            
            # 5. 保存比较图
            comparison_path = output_dir / f"{filename}_comparison.jpg"
            predictor.visualize_comparison(results, str(comparison_path))
            
            # 打印结果
            print(f"\n✅ 预测完成!")
            print(f"  原始图像: {original_path}")
            print(f"  预测叠加图: {overlay_path}")
            print(f"  完整预测: {prediction_path}")
            if results['ground_truth'] is not None:
                print(f"  标注答案: {gt_path}")
            print(f"  比较图: {comparison_path}")
            
            if results['iou'] is not None:
                print(f"\nIoU（交并比）: {results['iou']:.3f}")
            
            # 显示结果
            if args.visualize:
                predictor.visualize_comparison(results)
                cv2.imshow('Prediction Overlay', results['overlay'])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"预测时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理整个目录
    elif input_path.is_dir():
        print("批量处理模式 - 仅处理前10张图像作为示例")
        
        # 获取图像文件
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
        
        
        if len(image_files) == 0:
            print(f"错误: 在 {args.input} 中找不到图像文件")
            return
        
        ious = []
        
        for img_path in image_files:
            try:
                print(f"处理: {img_path.name}")
                
                # 查找对应的标注答案
                mask_path = None
                if os.path.exists(args.mask_dir):
                    mask_name = img_path.stem + '.png'
                    potential_path = Path(args.mask_dir) / mask_name
                    if potential_path.exists():
                        mask_path = str(potential_path)
                
                # 预测
                results = predictor.predict_single_image(str(img_path), mask_path)
                
                # 保存比较图
                filename = img_path.stem
                comparison_path = output_dir / f"{filename}_comparison.jpg"
                predictor.visualize_comparison(results, str(comparison_path))
                
                # 记录IoU
                if results['iou'] is not None:
                    ious.append(results['iou'])
                
            except Exception as e:
                print(f"处理 {img_path.name} 时出错: {e}")
        
        # 汇总统计
        print(f"\n处理完成! 总共处理 {len(image_files)} 张图像")
        print(f"结果保存在: {args.output}")
        
        if ious:
            avg_iou = np.mean(ious)
            std_iou = np.std(ious)
            print(f"\nIoU统计（仅限有标注答案的图像）:")
            print(f"  平均IoU: {avg_iou:.3f}")
            print(f"  IoU标准差: {std_iou:.3f}")
            print(f"  最小IoU: {np.min(ious):.3f}")
            print(f"  最大IoU: {np.max(ious):.3f}")
        
        # 创建汇总报告
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write(f"掌纹识别结果汇总\n")
            f.write(f"================\n")
            f.write(f"处理时间: {Path(__file__).stat().st_mtime}\n")
            f.write(f"总图像数: {len(image_files)}\n")
            f.write(f"模型路径: {args.model}\n\n")
            
            if ious:
                f.write(f"IoU统计:\n")
                f.write(f"  平均IoU: {np.mean(ious):.3f}\n")
                f.write(f"  标准差: {np.std(ious):.3f}\n")
                f.write(f"  范围: [{np.min(ious):.3f}, {np.max(ious):.3f}]\n")
    
    else:
        print(f"错误: 输入路径不存在: {args.input}")


if __name__ == '__main__':
    main()