#!/usr/bin/env python3
"""
手掌掌纹识别系统 - 桌面应用启动脚本
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
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 已安装")
    except ImportError:
        print("✗ PyQt5 未安装，请运行: pip install -r requirements_gui.txt")
        return False
    
    try:
        import cv2
        print("✓ OpenCV 已安装")
    except ImportError:
        print("✗ OpenCV 未安装")
        return False
    
    # 检查模型文件
    model_path = Path("checkpoints/best_three_lines_model.pth")
    if model_path.exists():
        print(f"✓ 模型文件存在: {model_path}")
    else:
        print(f"⚠ 模型文件不存在，请先运行训练脚本")
    
    return True


def main():
    """主函数"""
    print("手掌掌纹识别系统 - 桌面应用")
    print("=" * 50)
    
    if not check_dependencies():
        print("请安装缺失的依赖项后再运行")
        return
    
    print("\n正在启动应用程序...")
    
    try:
        from apps.main_window import main as app_main
        app_main()
    except Exception as e:
        print(f"启动应用程序时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()