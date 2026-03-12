#!/usr/bin/env python3
"""
测试叠加图像生成 - 使用真正的手掌图片
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scripts.predict_final import PalmLinePredictor

def test_overlay():
    """测试叠加图像"""
    # 查找模型文件
    model_files = list(Path("checkpoints").glob("*.pth"))
    if not model_files:
        print("错误: 未找到模型文件")
        return
    
    model_path = str(model_files[0])
    print(f"使用模型: {model_path}")
    
    # 初始化预测器
    try:
        predictor = PalmLinePredictor(model_path, 'cpu')
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 使用PLSU数据集中的手掌图片
    palm_image = "PLSU/img/image1005.jpg"
    
    if not Path(palm_image).exists():
        print(f"错误: 未找到手掌图片 {palm_image}")
        return
    
    print(f"测试图片: {palm_image}")
    
    # 先读取图片看看
    img = cv2.imread(palm_image)
    if img is not None:
        print(f"图片尺寸: {img.shape}")
    
    # 预测
    try:
        print("\n开始预测...")
        results = predictor.predict_single_image(palm_image)
        print(f"预测完成")
        
        # 检查 prediction_prob
        pred_prob = results.get('prediction_prob')
        if pred_prob is not None:
            print(f"\n预测概率图:")
            print(f"  Shape: {pred_prob.shape}")
            print(f"  Min/Max: {pred_prob.min():.4f}/{pred_prob.max():.4f}")
            print(f"  Mean: {pred_prob.mean():.4f}")
        
        # 检查 binary_mask
        binary_mask = results.get('full_prediction')
        if binary_mask is not None:
            print(f"\n二值掩码:")
            print(f"  Shape: {binary_mask.shape}")
            print(f"  非零像素: {np.sum(binary_mask > 0)}")
            print(f"  唯一值: {np.unique(binary_mask)}")
        
        # 检查 overlay
        overlay = results.get('overlay')
        if overlay is not None:
            print(f"\n叠加图像:")
            print(f"  Shape: {overlay.shape}")
            print(f"  与原图差异: {np.sum(overlay != results['image'])}")
            
            # 检查红色像素（BGR格式）
            red_pixels = np.sum((overlay[:,:,2] > 200) & (overlay[:,:,0] < 50) & (overlay[:,:,1] < 50))
            print(f"  红色像素数量: {red_pixels}")
            
            # 保存overlay
            output_path = "test_overlay_output.jpg"
            cv2.imwrite(output_path, overlay)
            print(f"\nOverlay 已保存到: {output_path}")
            
            # 同时保存原图对比
            cv2.imwrite("test_original.jpg", results['image'])
            print(f"原图已保存到: test_original.jpg")
            
            # 保存binary_mask
            cv2.imwrite("test_binary_mask.jpg", binary_mask)
            print(f"二值掩码已保存到: test_binary_mask.jpg")
            
            # 保存概率图（用于调试）
            prob_img = (pred_prob * 255).astype(np.uint8)
            cv2.imwrite("test_prob.jpg", prob_img)
            print(f"概率图已保存到: test_prob.jpg")
            
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_overlay()
