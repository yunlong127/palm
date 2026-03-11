#!/usr/bin/env python3
"""
手掌掌纹识别系统 - 网页应用启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 检查依赖
def check_dependencies():
    """检查依赖"""
    try:
        import gradio as gr
        print("✓ Gradio 已安装")
    except ImportError:
        print("✗ Gradio 未安装，请运行: pip install gradio")
        return False
    
    try:
        import cv2
        print("✓ OpenCV 已安装")
    except ImportError:
        print("✗ OpenCV 未安装")
        return False
    
    try:
        import torch
        print("✓ PyTorch 已安装")
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    
    # 检查模型文件
    model_path = Path("checkpoints/checkpoint_epoch_50.pth")
    if model_path.exists():
        print(f"✓ 模型文件存在: {model_path}")
    else:
        print(f"⚠ 模型文件不存在: {model_path}")
    
    return True


def main():
    """主函数"""
    print("手掌掌纹识别系统 - 网页应用")
    print("=" * 50)
    
    if not check_dependencies():
        print("请安装缺失的依赖项后再运行")
        return
    
    print("\n正在启动网页应用...")
    
    try:
        from apps.web_app import create_web_app
        app = create_web_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7865,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"启动网页应用时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
