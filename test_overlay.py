#!/usr/bin/env python3
"""
测试叠加图像生成
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
    
    # 查找测试图片
    test_images = list(Path(".").glob("**/*.jpg")) + list(Path(".").glob("**/*.png"))
    test_images = [p for p in test_images if 'temp' not in str(p).lower() and 'output' not in str(p).lower()]
    
    if not test_images:
        print("错误: 未找到测试图片")
        return
    
    # 使用第一张图片测试
    test_image = str(test_images[0])
    print(f"测试图片: {test_image}")
    
    # 预测
    try:
        results = predictor.predict_single_image(test_image)
        print(f"预测完成")
        print(f"结果键: {results.keys()}")
        
        # 检查 overlay
        overlay = results.get('overlay')
        if overlay is not None:
            print(f"Overlay shape: {overlay.shape}")
            print(f"Overlay dtype: {overlay.dtype}")
            print(f"Overlay min/max: {overlay.min()}/{overlay.max()}")
            
            # 检查是否有红色线条（BGR格式中红色是0,0,255）
            red_pixels = np.sum((overlay[:,:,2] > 200) & (overlay[:,:,0] < 50) & (overlay[:,:,1] < 50))
            print(f"红色像素数量: {red_pixels}")
            
            # 保存overlay
            output_path = "test_overlay_output.jpg"
            cv2.imwrite(output_path, overlay)
            print(f"Overlay 已保存到: {output_path}")
        else:
            print("错误: Overlay 为 None")
        
        # 检查 binary_mask
        binary_mask = results.get('full_prediction')
        if binary_mask is not None:
            print(f"Binary mask shape: {binary_mask.shape}")
            print(f"Binary mask 非零像素: {np.sum(binary_mask > 0)}")
        else:
            print("错误: Binary mask 为 None")
            
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_overlay()
