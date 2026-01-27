import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    """注意力门"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ResUNet(nn.Module):
    """Residual U-Net with Attention"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 4,
                 features: List[int] = [64, 128, 256, 512, 1024]):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 编码器
        self.encoder1 = ResidualBlock(in_channels, features[0])
        self.encoder2 = ResidualBlock(features[0], features[1])
        self.encoder3 = ResidualBlock(features[1], features[2])
        self.encoder4 = ResidualBlock(features[2], features[3])
        
        # 瓶颈层
        self.bottleneck = ResidualBlock(features[3], features[4])
        
        # 注意力门
        self.attention4 = AttentionGate(F_g=features[3], F_l=features[3], F_int=features[2])
        self.attention3 = AttentionGate(F_g=features[2], F_l=features[2], F_int=features[1])
        self.attention2 = AttentionGate(F_g=features[1], F_l=features[1], F_int=features[0])
        self.attention1 = AttentionGate(F_g=features[0], F_l=features[0], F_int=features[0] // 2)
        
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(features[4], features[3])
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(features[3], features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(features[2], features[1])
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(features[1], features[0])
        
        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # 瓶颈层
        bottleneck = self.bottleneck(self.pool(e4))
        
        # 解码器 with attention
        d4 = self.upconv4(bottleneck)
        e4_att = self.attention4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        e3_att = self.attention3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        e2_att = self.attention2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        e1_att = self.attention1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)