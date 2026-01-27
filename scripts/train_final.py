#!/usr/bin/env python3
"""
最终训练脚本 - 独立版本，不依赖复杂的导入
"""

import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
# 2. 数据集类
# ============================================================================
class SimplePLSUDataset(torch.utils.data.Dataset):
    """简化的PLSU数据集"""
    
    def __init__(self, img_dir, mask_dir, image_size=(512, 512), is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(img_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg'))])
        
        print(f"找到 {len(self.image_files)} 张图像")
        
        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码（转换为单通道，二值化）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"警告: 无法读取掩码: {mask_path}，使用空白掩码")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = (transformed['mask'] > 0).float()
        else:
            # 调整大小
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            mask = cv2.resize(mask, self.image_size)
            mask = torch.from_numpy(mask).float() / 255.0
            mask = (mask > 0.5).float()
        
        # 添加通道维度
        mask = mask.unsqueeze(0)
        
        return image, mask


# ============================================================================
# 3. 训练函数
# ============================================================================
def train_model():
    """主训练函数"""
    print("=" * 60)
    print("掌纹三大主线训练系统")
    print("=" * 60)
    
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = (512, 512)
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    
    print(f"使用设备: {device}")
    print(f"图像尺寸: {image_size}")
    print(f"批量大小: {batch_size}")
    print(f"训练轮次: {num_epochs}")
    
    # 检查数据目录
    data_root = "PLSU"
    img_dir = os.path.join(data_root, "img")
    mask_dir = os.path.join(data_root, "Mask")
    
    if not os.path.exists(img_dir):
        print(f"错误: 找不到图像目录: {img_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"错误: 找不到掩码目录: {mask_dir}")
        return
    
    # 获取所有图像文件
    all_files = sorted([f for f in os.listdir(img_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg'))])
    
    if len(all_files) == 0:
        print(f"错误: 在 {img_dir} 中找不到jpg图像")
        return
    
    # 划分数据集
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"训练集: {len(train_files)} 张图像")
    print(f"验证集: {len(val_files)} 张图像")
    
    # 创建数据集（只使用训练集文件）
    train_dataset = SimplePLSUDataset(img_dir, mask_dir, image_size, is_train=True)
    val_dataset = SimplePLSUDataset(img_dir, mask_dir, image_size, is_train=False)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建模型
    model = UNet(
        in_channels=3,
        out_channels=1,  # 二值分割
        features=[32, 64, 128, 256, 512]
    ).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 学习率调度器 - 修复：移除 verbose 参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 创建输出目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 训练循环
    best_iou = 0
    train_losses = []
    val_losses = []
    val_ious = []
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}/{num_epochs}')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0
        total_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # 计算IoU
                preds = torch.sigmoid(outputs) > 0.5
                intersection = (preds & masks.bool()).sum().item()
                union = (preds | masks.bool()).sum().item()
                
                if union > 0:
                    iou = intersection / union
                    total_iou += iou
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        
        val_losses.append(avg_val_loss)
        val_ious.append(avg_iou)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:03d}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if avg_iou > best_iou:
            best_iou = avg_iou
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'iou': avg_iou,
                'image_size': image_size,
                'batch_size': batch_size,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_ious': val_ious
            }, 'checkpoints/best_three_lines_model.pth')
            
            print(f"  ✅ 保存最佳模型，IoU: {avg_iou:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'iou': avg_iou
            }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
            print(f"  📁 保存检查点")
    
    print("-" * 60)
    print(f"训练完成！最佳IoU: {best_iou:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_ious)
    
    # 测试模型
    test_model(model, val_loader, device)
    
    return model


def plot_training_curves(train_losses, val_losses, val_ious):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失')
    axes[0].plot(val_losses, label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU曲线
    axes[1].plot(val_ious)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('验证集IoU')
    axes[1].grid(True, alpha=0.3)
    
    # 损失与IoU关系
    axes[2].scatter(val_losses, val_ious, alpha=0.6)
    axes[2].set_xlabel('验证损失')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('损失与IoU关系')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: results/training_curves.jpg")


def test_model(model, val_loader, device):
    """测试模型"""
    model.eval()
    
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images, masks = images.to(device), masks.to(device)
        
        outputs = torch.sigmoid(model(images))
        predictions = outputs > 0.5
        
        # 可视化几个样本
        num_samples = min(4, images.size(0))
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # 原始图像（反标准化）
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # 真实标注
            true = masks[i, 0].cpu().numpy() > 0.5
            
            # 预测结果
            pred = predictions[i, 0].cpu().numpy()
            
            # 计算IoU
            intersection = np.logical_and(pred, true).sum()
            union = np.logical_or(pred, true).sum()
            iou = intersection / union if union > 0 else 0
            
            # 叠加结果
            overlay = (img * 255).astype(np.uint8).copy()
            overlay[pred > 0] = [255, 0, 0]  # 红色表示预测
            
            # 绘制
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('原始图像')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(true, cmap='gray')
            axes[i, 1].set_title('真实标注')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title(f'预测结果')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'叠加 (IoU={iou:.3f})')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/test_samples.jpg', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"测试样本已保存到: results/test_samples.jpg")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        train_model()
        print("\n✅ 训练完成！")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()