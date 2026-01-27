#!/usr/bin/env python3
"""
配置管理器
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class AppSettings:
    """应用程序设置管理器"""
    
    def __init__(self):
        self.settings_file = Path("user_settings.json")
        self.default_settings = {
            'window_geometry': None,
            'window_state': None,
            'model_type': 0,  # 0: U-Net, 1: ResUNet
            'confidence_threshold': 0.3,
            'line_width': 3,
            'enhance_image': True,
            'save_overlay': True,
            'save_json': True,
            'auto_open': True,
            'recent_files': [],
            'output_dir': "results",
            'use_gpu': True,
            'theme': 'light'
        }
    
    def load(self) -> Dict[str, Any]:
        """加载设置"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                    # 合并默认设置（确保所有键都存在）
                    for key, value in self.default_settings.items():
                        if key not in settings:
                            settings[key] = value
                    
                    return settings
            else:
                return self.default_settings.copy()
        except Exception as e:
            print(f"加载设置失败: {e}")
            return self.default_settings.copy()
    
    def save(self, settings: Dict[str, Any]):
        """保存设置"""
        try:
            # 确保输出目录存在
            output_dir = settings.get('output_dir', 'results')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")
    
    def add_recent_file(self, file_path: str):
        """添加最近文件"""
        settings = self.load()
        
        if 'recent_files' not in settings:
            settings['recent_files'] = []
        
        # 移除重复项
        if file_path in settings['recent_files']:
            settings['recent_files'].remove(file_path)
        
        # 添加到开头
        settings['recent_files'].insert(0, file_path)
        
        # 限制数量
        settings['recent_files'] = settings['recent_files'][:10]
        
        self.save(settings)
    
    def clear_recent_files(self):
        """清空最近文件"""
        settings = self.load()
        settings['recent_files'] = []
        self.save(settings)