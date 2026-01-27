# data_loader.py - 修改部分
import numpy as np
import cv2
from typing import List

class PLSUDataset:
    """PLSU数据集加载器 - 修改为只处理三大主线"""
    
    def __init__(self, config, image_paths, mask_paths, is_train=True):
        self.config = config
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train
        
        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.Resize(*config.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5
                ),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=config.mean, std=config.std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*config.image_size),
                A.Normalize(mean=config.mean, std=config.std),
                ToTensorV2()
            ])
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标注并提取三条主线
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 分离三条主线，排除命运线
        heart_mask, head_mask, life_mask = self._extract_three_main_lines(mask)
        
        # 合并为三通道掩码
        lines_mask = np.stack([heart_mask, head_mask, life_mask], axis=2)
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=lines_mask)
            image = transformed['image']
            mask = transformed['mask'].permute(2, 0, 1).float()  # [3, H, W]
        else:
            # 调整大小
            image = cv2.resize(image, self.config.image_size)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            lines_mask = cv2.resize(lines_mask, self.config.image_size)
            mask = torch.from_numpy(lines_mask).permute(2, 0, 1).float()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path
        }
    
    def _extract_three_main_lines(self, mask):
        """
        从标注中提取三条主线，排除命运线
        
        策略：
        1. 找到所有连通分量
        2. 根据位置和特征选择三条主线
        3. 排除与生命线平行且靠近的命运线
        """
        # 二值化
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 找到所有连通分量
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        if num_labels <= 1:  # 只有背景
            height, width = mask.shape
            return np.zeros((height, width), dtype=np.uint8), \
                   np.zeros((height, width), dtype=np.uint8), \
                   np.zeros((height, width), dtype=np.uint8)
        
        # 收集所有非背景分量
        components = []
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            area = stats[label, cv2.CC_STAT_AREA]
            
            # 计算特征
            y_coords, x_coords = np.where(component_mask > 0)
            if len(y_coords) == 0:
                continue
                
            mean_y = np.mean(y_coords)
            mean_x = np.mean(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            
            components.append({
                'mask': component_mask,
                'area': area,
                'mean_y': mean_y,
                'mean_x': mean_x,
                'min_y': min_y,
                'max_y': max_y,
                'length': max_y - min_y
            })
        
        # 按垂直位置排序（从上到下）
        components.sort(key=lambda x: x['mean_y'])
        
        # 如果只有3个或更少分量，直接返回
        if len(components) <= 3:
            masks = [c['mask'] for c in components]
            while len(masks) < 3:
                masks.append(np.zeros_like(mask))
            return masks
        
        # 尝试选择三条主线
        # 策略：选择最上方的线作为感情线，最下方的线作为生命线，中间的作为智慧线
        selected = []
        
        # 1. 选择最上方的线作为感情线
        top_component = min(components, key=lambda x: x['mean_y'])
        selected.append(top_component)
        components.remove(top_component)
        
        # 2. 选择最下方的线作为生命线
        bottom_component = max(components, key=lambda x: x['mean_y'])
        selected.append(bottom_component)
        components.remove(bottom_component)
        
        # 3. 从剩余的线中选择中间位置的线作为智慧线
        # 计算中间位置
        mean_y_values = [c['mean_y'] for c in components]
        target_y = (top_component['mean_y'] + bottom_component['mean_y']) / 2
        
        # 选择最接近中间位置的线
        middle_component = min(components, 
                              key=lambda x: abs(x['mean_y'] - target_y))
        selected.append(middle_component)
        
        # 按位置排序（上、中、下）
        selected.sort(key=lambda x: x['mean_y'])
        
        return selected[0]['mask'], selected[1]['mask'], selected[2]['mask']
    # 在 data_loader.py 文件末尾添加 create_dataloaders 函数

    def create_dataloaders(config, train_ratio=0.8, val_ratio=0.2):
        """
        创建训练和验证数据加载器
        
        Args:
            config: 配置对象
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            train_loader, val_loader
        """
        # 获取图像和掩码路径
        img_dir = os.path.join(config.data_root, config.image_dir)
        mask_dir = os.path.join(config.data_root, config.mask_dir)
        
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(img_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg'))])
        
        # 创建图像和掩码路径
        image_paths = []
        mask_paths = []
        
        for img_file in image_files:
            img_path = os.path.join(img_dir, img_file)
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        print(f"总共找到 {len(image_paths)} 个图像-掩码对")
        
        # 划分数据集
        np.random.seed(42)
        indices = np.random.permutation(len(image_paths))
        
        train_end = int(len(image_paths) * train_ratio)
        val_end = train_end + int(len(image_paths) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        
        # 创建数据集
        train_dataset = PLSUDataset(
            config=config,
            image_paths=[image_paths[i] for i in train_indices],
            mask_paths=[mask_paths[i] for i in train_indices],
            is_train=True
        )
        
        val_dataset = PLSUDataset(
            config=config,
            image_paths=[image_paths[i] for i in val_indices],
            mask_paths=[mask_paths[i] for i in val_indices],
            is_train=False
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"训练集: {len(train_dataset)} 张图像")
        print(f"验证集: {len(val_dataset)} 张图像")
        
        return train_loader, val_loader