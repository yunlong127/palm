import argparse
import yaml
from src.config import Config
from src.data_loader import get_dataloader
from src.models import UNet, ResUNet
from src.evaluator import Evaluator
from src.utils import load_model

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Evaluate palm line recognition model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 数据加载器
    test_loader = get_dataloader(
        config['data.test_dir'],
        config['data.test_mask_dir'],
        config['data.batch_size'],
        config['data.image_size'],
        shuffle=False
    )
    
    # 模型
    model_name = config['model.name']
    if model_name == 'unet':
        model = UNet(
            in_channels=config['model.in_channels'],
            out_channels=config['model.out_channels']
        )
    elif model_name == 'resunet':
        model = ResUNet(
            in_channels=config['model.in_channels'],
            out_channels=config['model.out_channels']
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    # 加载模型权重
    model = load_model(model, args.model_path)
    
    # 评估器
    evaluator = Evaluator(model, test_loader)
    
    # 评估
    metrics = evaluator.evaluate()
    
    # 打印评估指标
    evaluator.print_metrics(metrics)

if __name__ == '__main__':
    main()
