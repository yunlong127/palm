# 掌纹识别系统 (Palm Line Recognition System)

## 项目简介

掌纹识别系统是一个基于深度学习的自动掌纹识别应用，能够自动检测和识别手掌上的主要纹路，包括生命线、智慧线、感情线等。系统采用先进的计算机视觉技术和深度学习模型，实现了从图像采集、预处理、特征提取到结果可视化的完整流程。

### 核心特性

- **智能掌纹检测**：基于深度学习的自动掌纹区域检测
- **多模型支持**：集成UNet和ResUNet两种先进的分割模型
- **实时预测**：支持单张图片和批量图片处理
- **可视化界面**：提供友好的PyQt5桌面应用界面
- **置信度评估**：基于掌纹面积比值的智能置信度计算
- **结果导出**：支持图片和JSON格式的结果导出
- **训练与评估**：完整的模型训练和评估流程

## 项目结构

```
palm-line-recognition/
│
├── apps/                    # 桌面应用模块
│   ├── main_window.py      # 主窗口界面
│   ├── image_processor.py   # 图像处理器
│   ├── results_viewer.py   # 结果查看器
│   └── settings.py         # 应用设置
│
├── src/                     # 核心源代码
│   ├── config.py           # 配置管理
│   ├── data_loader.py      # 数据加载器
│   ├── preprocessor.py     # 图像预处理
│   ├── trainer.py          # 模型训练器
│   ├── evaluator.py        # 模型评估器
│   ├── predictor.py        # 预测器
│   └── models/            # 深度学习模型
│       ├── unet.py         # UNet模型
│       └── resunet.py      # ResUNet模型
│
├── scripts/                 # 执行脚本
│   ├── train_final.py      # 训练脚本
│   ├── predict_final.py    # 预测脚本
│   └── evaluate.py         # 评估脚本
│
├── tests/                   # 测试文件
│   └── test_app.py         # 应用测试
│
├── checkpoints/             # 模型检查点
│   └── best_three_lines_model.pth  # 最佳模型
│
├── PLSU/                   # PLSU数据集
│   ├── img/               # 原始图像
│   └── Mask/              # 标注掩码
│
├── docs/                   # 文档
│   └── design.md          # 技术设计文档
│
├── requirements.txt        # Python依赖
├── run_app.py             # 应用启动脚本
└── README.md              # 项目说明
```

## 安装步骤

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 8GB+ RAM
- 2GB+ 显存 (GPU模式)

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/palm-line-recognition.git
cd palm-line-recognition
```

### 2. 创建虚拟环境

```bash
# 使用conda
conda create -n palm-recognition python=3.9
conda activate palm-recognition

# 或使用venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装PyQt5（桌面应用）
pip install PyQt5
```

### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
python -c "import PyQt5; print('PyQt5安装成功')"
```

## 运行步骤

### 桌面应用模式

#### 1. 启动应用

```bash
python run_app.py
```

#### 2. 使用界面

1. **选择模式**：
   - 单张图片模式：处理单个手掌图像
   - 批量处理模式：处理文件夹中的所有图像

2. **加载图片**：
   - 点击"选择图片"按钮选择单个图像
   - 或点击"选择文件夹"批量加载图像

3. **配置参数**：
   - 模型选择：U-Net (推荐) 或 ResUNet
   - 置信度阈值：0.1-1.0
   - 线条粗细：1-10像素
   - 图像增强：启用/禁用

4. **开始识别**：
   - 点击"开始识别"按钮
   - 等待处理完成

5. **查看结果**：
   - 结果图片：显示识别后的叠加图像
   - 结果信息：总置信度、处理时间、图像尺寸、检测状态
   - 识别建议：基于识别结果的建议
   - 线条概览：详细显示各线条的识别信息

6. **保存结果**：
   - 点击"保存结果"保存为图片
   - 点击"导出JSON"保存为JSON格式

### 命令行模式

#### 单张图片预测

```bash
python scripts/predict_final.py \
    --model_path checkpoints/best_three_lines_model.pth \
    --image_path path/to/your/image.jpg \
    --output_path output/result.jpg
```

#### 批量预测

```bash
python batch_predict.py \
    --model_path checkpoints/best_three_lines_model.pth \
    --input_dir path/to/images \
    --output_dir path/to/results
```

### 模型训练

#### 1. 准备数据

确保数据集按以下结构组织：

```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

#### 2. 配置训练参数

编辑 `src/config.py` 或创建自定义配置文件。

#### 3. 开始训练

```bash
python scripts/train_final.py \
    --data_dir PLSU \
    --model_type unet \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.001
```

#### 4. 监控训练

训练过程中会自动保存检查点到 `checkpoints/` 目录，可以使用TensorBoard监控训练过程：

```bash
tensorboard --logdir logs
```

## 演示步骤

### 快速演示

#### 1. 使用预训练模型

```bash
# 启动应用
python run_app.py

# 在界面中选择一张测试图片
# 点击"开始识别"
# 查看识别结果
```

#### 2. 测试样例

项目包含PLSU数据集的样本，可以使用以下图片进行测试：

```bash
# 测试单张图片
python scripts/predict_final.py \
    --model_path checkpoints/best_three_lines_model.pth \
    --image_path PLSU/img/image1.jpg \
    --output_path test_output.jpg
```

#### 3. 批量演示

```bash
# 创建测试输出目录
mkdir -p batch_results

# 批量处理
python batch_predict.py \
    --model_path checkpoints/best_three_lines_model.pth \
    --input_dir PLSU/img \
    --output_dir batch_results
```

### 功能演示

#### 置信度评估

系统使用掌纹面积与手掌面积的比值来计算置信度：

- **高置信度 (≥70%)**：掌纹清晰，识别准确
- **中等置信度 (30%-70%)**：掌纹一般，可能需要重新拍摄
- **低置信度 (<30%)**：掌纹不清晰，建议重新拍摄

#### 建议生成

系统会根据识别结果自动生成建议：

- 掌纹清晰度建议
- 拍摄角度建议
- 光线条件建议
- 手部姿势建议

## 配置说明

### 模型配置

在 `src/config.py` 中配置模型参数：

```python
# 数据配置
DATA_CONFIG = {
    'image_size': (512, 512),
    'batch_size': 8,
    'num_workers': 4,
}

# 模型配置
MODEL_CONFIG = {
    'in_channels': 3,
    'out_channels': 1,
    'num_classes': 4,  # 背景 + 3条主线
}

# 训练配置
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 10,
}
```

### 应用配置

在 `apps/settings.py` 中配置应用参数：

```python
# 界面配置
WINDOW_CONFIG = {
    'width': 1400,
    'height': 900,
    'title': '手掌掌纹识别系统 v1.0',
}

# 处理配置
PROCESSING_CONFIG = {
    'default_model': 'unet',
    'confidence_threshold': 0.3,
    'line_width': 3,
    'enhance_image': True,
}
```

## 技术栈

### 深度学习框架
- **PyTorch 2.0+**: 核心深度学习框架
- **TorchVision**: 图像处理和变换

### 计算机视觉
- **OpenCV 4.8+**: 图像处理和计算机视觉
- **Pillow**: 图像IO操作
- **scikit-image**: 图像处理算法

### 数据处理
- **NumPy**: 数值计算
- **Pandas**: 数据分析
- **scikit-learn**: 机器学习工具

### 可视化
- **Matplotlib**: 数据可视化
- **Seaborn**: 统计可视化
- **PyQt5**: 桌面应用界面

### 训练工具
- **TensorBoard**: 训练监控
- **WandB**: 实验跟踪
- **tqdm**: 进度条

### 图像增强
- **Albumentations**: 高效图像增强

### 模型架构
- **UNet**: 经典分割网络
- **ResUNet**: 结合残差连接的改进网络
- **EfficientNet**: 高效骨干网络

## 性能指标

### 模型性能

在PLSU数据集上的测试结果：

| 模型 | IoU | Dice | 精确率 | 召回率 | F1分数 |
|------|-----|------|--------|--------|--------|
| UNet | 0.524 | 0.689 | 0.721 | 0.658 | 0.688 |
| ResUNet | 0.531 | 0.695 | 0.728 | 0.667 | 0.696 |

### 处理速度

- **CPU模式**: ~2-3秒/张
- **GPU模式**: ~0.5-1秒/张 (NVIDIA RTX 3060)

## 常见问题

### 1. 模型加载失败

**问题**: 提示模型文件不存在

**解决方案**:
```bash
# 检查模型文件是否存在
ls checkpoints/best_three_lines_model.pth

# 如果不存在，需要先训练模型
python scripts/train_final.py --data_dir PLSU
```

### 2. CUDA内存不足

**问题**: 提示CUDA out of memory

**解决方案**:
- 减小batch_size
- 使用较小的图像尺寸
- 使用CPU模式
- 清理GPU缓存

### 3. 图像识别效果差

**问题**: 识别结果不准确

**解决方案**:
- 确保图像质量良好
- 手掌完全在画面中
- 光线充足
- 手心朝上，手指自然分开
- 尝试调整置信度阈值

### 4. PyQt5安装失败

**问题**: PyQt5安装出错

**解决方案**:
```bash
# 使用conda安装
conda install pyqt

# 或使用pip安装特定版本
pip install PyQt5==5.15.9
```

## 贡献指南

我们欢迎任何形式的贡献！

### 贡献方式

1. **报告Bug**: 在Issues中提交bug报告
2. **功能建议**: 提出新功能或改进建议
3. **代码贡献**: 提交Pull Request
4. **文档改进**: 完善文档和示例

### 开发流程

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 更新相关文档

## 许可证

本项目采用 PolyForm NonCommercial License 1.0.0 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- **PLSU数据集**: 感谢PLSU数据集的提供者
- **PyTorch团队**: 感谢PyTorch框架的支持
- **开源社区**: 感谢所有开源项目的贡献者

## 联系方式

- **项目主页**: https://github.com/yourusername/palm-line-recognition
- **问题反馈**: https://github.com/yourusername/palm-line-recognition/issues
- **邮箱**: your.email@example.com

## 更新日志

### v1.0.0 (2024-01-27)
- 初始版本发布
- 实现UNet和ResUNet模型
- 完成桌面应用界面
- 支持单张和批量处理
- 实现置信度评估
- 添加结果导出功能

---

**注意**: 本项目仅用于研究和教育目的。请确保在使用时遵守相关法律法规和隐私政策。
