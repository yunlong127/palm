"""
手掌掌纹识别系统

一个基于深度学习的自动手掌掌纹识别系统，能够自动检测手掌区域并提取三条主要掌纹线。
"""

__version__ = "1.0.0"
__description__ = "手掌掌纹识别与三大主线绘制系统"
__author__ = "Palm Recognition Project"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Palm Recognition Project"

# 导出主要功能
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 简化的导入
def train_model(data_root="PLSU", model_name="resunet", use_gpu=True):
    """训练模型的快捷函数"""
    from scripts.train import main as train_main
    import sys
    
    sys.argv = [
        sys.argv[0],
        "--data-root", data_root,
        "--model", model_name,
    ]
    
    if use_gpu:
        sys.argv.append("--gpu")
    
    train_main()

def predict_image(image_path, model_path="checkpoints/best_model.pth"):
    """预测图像的快捷函数"""
    from scripts.predict import main as predict_main
    import sys
    
    sys.argv = [
        sys.argv[0],
        "--image", image_path,
        "--model", model_path,
    ]
    
    predict_main()

def launch_web_interface(port=7860):
    """启动Web界面的快捷函数"""
    from scripts.predict import main as predict_main
    import sys
    
    sys.argv = [
        sys.argv[0],
        "--web",
        "--port", str(port),
    ]
    
    predict_main()