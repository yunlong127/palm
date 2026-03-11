#!/usr/bin/env python3
"""
手掌掌纹识别网页应用 - 新布局设计
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import gradio as gr

sys.path.append(str(Path(__file__).parent.parent))

from apps.image_processor import ImageProcessor, ProcessResult


class WebApp:
    """网页应用类"""
    
    def __init__(self):
        self.processor = None
        self.model_loaded = False
        self.current_config = {
            'model_type': 'unet',
            'confidence_threshold': 0.3,
            'line_width': 3,
            'enhance_image': True,
            'use_gpu': True,
            'save_overlay': True,
            'save_json': True
        }
        self.current_image_path = None
        self.current_result = None
        self.batch_results = []
        self.batch_mode = False
        self.batch_images = []
        self.current_batch_index = 0
    
    def load_model(self) -> Tuple[bool, str]:
        """加载模型"""
        if self.model_loaded:
            return True, "模型已加载"
        
        try:
            available_models = list(Path("checkpoints").glob("*.pth"))
            if not available_models:
                return False, "未找到模型文件"
            
            model_path = str(sorted(available_models, key=lambda x: x.stat().st_mtime, reverse=True)[0])
            print(f"使用模型: {model_path}")
            
            self.current_config['model_path'] = model_path
            self.processor = ImageProcessor(self.current_config)
            self.processor.model_paths['unet'] = model_path
            self.processor.model_paths['resunet'] = model_path
            
            if self.processor.load_model():
                self.model_loaded = True
                return True, f"模型加载成功: {Path(model_path).name}"
            else:
                return False, "模型加载失败"
        except Exception as e:
            return False, f"模型加载错误: {str(e)}"
    
    def process_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, str, str, str, str]:
        """处理单张图片"""
        if not self.model_loaded:
            success, msg = self.load_model()
            if not success:
                return image, "请选择图片进行识别", "", "", ""
        
        if image is None:
            return np.zeros((400, 400, 3), dtype=np.uint8), "请选择图片进行识别", "", "", ""
        
        try:
            temp_path = "temp_input.jpg"
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, image_bgr)
            self.current_image_path = temp_path
            
            result = self.processor.process_image(temp_path)
            self.current_result = result
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result.success:
                output_image = result.overlay_image if result.overlay_image is not None else image
                result_info = f"总置信度: {result.confidences.get('total', 0):.2f}"
                processing_info = f"处理时间: {result.processing_time:.2f}秒\n"
                if result.image_size:
                    processing_info += f"图像尺寸: {result.image_size[1]}x{result.image_size[0]}\n"
                processing_info += "检测状态: 成功"
                suggestions_text = "\n".join(result.suggestions) if result.suggestions else "无建议"
                lines_table = self.format_lines_table(result)
                
                return output_image, result_info, processing_info, suggestions_text, lines_table
            else:
                return image, "处理失败", "", result.error_message or "未知错误", ""
                
        except Exception as e:
            return image, f"处理错误: {str(e)}", "", "请检查图片格式和内容", ""
    
    def format_lines_table(self, result: ProcessResult) -> str:
        """格式化线条数据为表格文本"""
        lines_data = result.lines_data
        if not lines_data:
            return "暂无线条数据"
        
        table = "| 线条名称 | 置信度 | 点数 | 状态 | 备注 |\n"
        table += "|---------|--------|------|------|------|\n"
        
        line_names = {
            'heart': '感情线',
            'head': '智慧线',
            'life': '生命线'
        }
        
        for line in lines_data:
            name = line_names.get(line['name'], line['name'])
            confidence = f"{line.get('confidence', 0):.2f}"
            points = str(line.get('length', 0))
            status = "✓ 已检测" if line.get('confidence', 0) > 0.5 else "⚠ 低置信度"
            note = line.get('note', '-')
            table += f"| {name} | {confidence} | {points} | {status} | {note} |\n"
        
        return table
    
    def format_lines_html(self, result: ProcessResult) -> str:
        """格式化线条数据为彩色HTML显示"""
        lines_data = result.lines_data
        if not lines_data:
            return '<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'
        
        line_colors = {
            'heart': '#FF6B6B',
            'head': '#4ECDC4',
            'life': '#45B7D1'
        }
        
        line_names = {
            'heart': '感情线',
            'head': '智慧线',
            'life': '生命线'
        }
        
        html = '<div style="padding: 15px;">'
        
        for line in lines_data:
            name = line['name']
            color = line_colors.get(name, '#999')
            display_name = line_names.get(name, name)
            confidence = line.get('confidence', 0)
            points = line.get('length', 0)
            
            html += f'''
            <div style="margin-bottom: 15px; padding: 12px; border-left: 4px solid {color}; background: linear-gradient(90deg, {color}15, transparent); border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: bold; font-size: 16px; color: {color};">{display_name}</span>
                    <span style="font-size: 14px; color: #666;">点数: {points}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 14px; color: #333; margin-right: 10px;">置信度:</span>
                        <span style="font-weight: bold; font-size: 16px; color: {color};">{confidence:.2f}</span>
                    </div>
                    <div style="background: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {confidence * 100}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                <div style="font-size: 13px; color: #666;">
                    状态: {'✓ 已检测' if confidence > 0.5 else '⚠ 低置信度'}
                </div>
            </div>
            '''
        
        html += '</div>'
        return html
    
    def process_batch_images(self, files: List[Any], progress: gr.Progress) -> Tuple[str, str, List[np.ndarray], List[np.ndarray], str, str, str, str, List[Dict]]:
        """批量处理图片"""
        if not self.model_loaded:
            success, msg = self.load_model()
            if not success:
                return "错误", f"模型加载失败: {msg}", [], [], "--", "--", "--", "", []
        
        if not files:
            return "错误", "未选择图片", [], [], "0", "0", "0", "0%", []
        
        self.batch_results = []
        self.batch_images = []
        overlay_images = []
        total = len(files)
        successful = 0
        failed = 0
        
        for i, file in enumerate(files):
            try:
                progress(i / total, desc=f"处理中: {Path(file.name).name}")
                
                img = cv2.imread(file.name)
                if img is None:
                    failed += 1
                    self.batch_results.append({
                        'filename': Path(file.name).name,
                        'success': False,
                        'error': '无法读取图片',
                        'original_image': None,
                        'result': None
                    })
                    self.batch_images.append(None)
                    overlay_images.append(None)
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.batch_images.append(img_rgb)
                
                result = self.processor.process_image(file.name)
                
                if result.success:
                    successful += 1
                    overlay_image = result.overlay_image
                    self.batch_results.append({
                        'filename': Path(file.name).name,
                        'success': True,
                        'confidence': result.confidences.get('total', 0),
                        'processing_time': result.processing_time,
                        'result': result,
                        'original_image': img_rgb,
                        'overlay_image': overlay_image
                    })
                    overlay_images.append(overlay_image)
                else:
                    failed += 1
                    self.batch_results.append({
                        'filename': Path(file.name).name,
                        'success': False,
                        'error': result.error_message,
                        'original_image': img_rgb,
                        'result': None
                    })
                    overlay_images.append(None)
                    
            except Exception as e:
                failed += 1
                self.batch_results.append({
                    'filename': Path(file.name).name if hasattr(file, 'name') else f'图片_{i+1}',
                    'success': False,
                    'error': str(e),
                    'original_image': None,
                    'result': None
                })
                self.batch_images.append(None)
                overlay_images.append(None)
        
        progress(1.0, desc="处理完成")
        
        success_rate = (successful / total * 100) if total > 0 else 0
        batch_stats_text = f"总图片数: {total}\n成功: {successful}\n失败: {failed}\n成功率: {success_rate:.1f}%"
        
        return (
            "完成", 
            batch_stats_text, 
            self.batch_images,
            overlay_images,
            str(total),
            str(successful),
            str(failed),
            f"{success_rate:.1f}%",
            self.batch_results
        )
    
    def get_batch_image(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, str, str, str, str, str]:
        """获取指定索引的批量图片信息"""
        if not self.batch_results or index < 0 or index >= len(self.batch_results):
            return (
                np.zeros((400, 400, 3), dtype=np.uint8),
                np.zeros((400, 400, 3), dtype=np.uint8),
                "--", "--", "--", "--", "",
                '<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'
            )
        
        result_data = self.batch_results[index]
        original_image = self.batch_images[index]
        
        if result_data['success'] and result_data['result']:
            result = result_data['result']
            result_info = f"总置信度: {result.confidences.get('total', 0):.2f}"
            time_val = f"{result.processing_time:.2f}秒"
            size_val = f"{result.image_size[1]}x{result.image_size[0]}" if result.image_size else "--"
            status_val = "成功"
            suggestions = "\n".join(result.suggestions) if result.suggestions else "无建议"
            lines_html = self.format_lines_html(result)
            overlay_image = result_data['overlay_image']
        else:
            result_info = "--"
            time_val = "--"
            size_val = "--"
            status_val = "失败"
            suggestions = result_data.get('error', '处理失败')
            lines_html = '<div style="padding: 20px; text-align: center; color: #999;">处理失败</div>'
            overlay_image = np.zeros_like(original_image) if original_image is not None else np.zeros((400, 400, 3), dtype=np.uint8)
        
        return (
            original_image if original_image is not None else np.zeros((400, 400, 3), dtype=np.uint8),
            overlay_image,
            result_info,
            time_val,
            size_val,
            status_val,
            suggestions,
            lines_html
        )
    
    def update_config(self, model_type_val, conf_thresh, line_w, enhance, save_overlay, save_json):
        """更新配置"""
        self.current_config.update({
            'model_type': 'unet' if 'U-Net' in model_type_val else 'resunet',
            'confidence_threshold': conf_thresh,
            'line_width': int(line_w),
            'enhance_image': enhance,
            'save_overlay': save_overlay,
            'save_json': save_json
        })


def create_web_app():
    """创建 Gradio 网页应用 - 新布局设计"""
    app = WebApp()
    
    custom_css = """
    .gradio-container {
        max-width: 1800px !important;
    }
    .header {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .group-box {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .group-box-title {
        font-weight: bold;
        font-size: 14px;
        color: #333;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #ddd;
    }
    .horizontal-scroll-gallery {
        display: block !important;
        overflow-x: hidden !important;
        overflow-y: scroll !important;
        max-height: 600px !important;
        padding: 10px 0 !important;
        scrollbar-width: thin !important;
        scrollbar-color: #888 #f1f1f1 !important;
    }
    .horizontal-scroll-gallery::-webkit-scrollbar {
        width: 8px !important;
        display: block !important;
    }
    .horizontal-scroll-gallery::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 4px !important;
    }
    .horizontal-scroll-gallery::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 4px !important;
    }
    .horizontal-scroll-gallery::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }
    .horizontal-scroll-gallery .preview {
        display: none !important;
    }
    .horizontal-scroll-gallery .grid-wrap {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
        height: auto !important;
    }
    .horizontal-scroll-gallery .thumbnail-item {
        border: 3px solid transparent !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        width: 100% !important;
        min-height: 150px !important;
        height: auto !important;
    }
    .horizontal-scroll-gallery .thumbnail-item.selected {
        border: 3px solid #667eea !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.6) !important;
    }
    .horizontal-scroll-gallery .thumbnail-item:hover {
        border: 3px solid #a0a0a0 !important;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2) !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="手掌掌纹识别系统 v1.0") as demo:
        gr.HTML("""
        <div class="header">
            <h2>🖐️ 手掌掌纹识别系统 v1.0</h2>
            <p style="margin: 0;">基于深度学习的掌纹三线识别 | Web版</p>
        </div>
        <script>
        (function() {
            'use strict';
            var gallerySetupComplete = false;
            
            function setupGallerySelection() {
                if (gallerySetupComplete) return;
                
                try {
                    var galleries = document.querySelectorAll('.horizontal-scroll-gallery');
                    if (galleries.length === 0) return;
                    
                    galleries.forEach(function(gallery) {
                        if (gallery.dataset.setup === 'true') return;
                        gallery.dataset.setup = 'true';
                        
                        var items = gallery.querySelectorAll('.thumbnail-item');
                        if (items.length > 0) {
                            var hasSelected = gallery.querySelector('.thumbnail-item.selected');
                            if (!hasSelected) {
                                items[0].classList.add('selected');
                            }
                        }
                        
                        gallery.addEventListener('click', function(e) {
                            try {
                                var items = gallery.querySelectorAll('.thumbnail-item');
                                if (items.length === 0) return;
                                
                                items.forEach(function(item) { item.classList.remove('selected'); });
                                var target = e.target.closest('.thumbnail-item');
                                if (target) {
                                    target.classList.add('selected');
                                }
                            } catch (err) {
                                console.log('Gallery click handler error:', err);
                            }
                        });
                    });
                    
                    gallerySetupComplete = true;
                } catch (err) {
                    console.log('Gallery setup error:', err);
                }
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    setTimeout(setupGallerySelection, 500);
                });
            } else {
                setTimeout(setupGallerySelection, 500);
            }
            
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length > 0) {
                        setTimeout(setupGallerySelection, 100);
                    }
                });
            });
            
            observer.observe(document.body || document.documentElement, {
                childList: true,
                subtree: true
            });
        })();
        </script>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">📤 上传区域</div>')
                    
                    with gr.Group(visible=True) as upload_controls:
                        mode_radio = gr.Radio(
                            choices=["批量处理模式"],
                            value="批量处理模式",
                            label="处理模式",
                            container=False
                        )
                        batch_input = gr.File(
                            label="选择图片",
                            file_count="multiple",
                            file_types=["image"],
                            height=100
                        )
                    
                    with gr.Group(visible=False) as original_images_group:
                        original_images = gr.Gallery(
                            label="📋 原图列表（点击选择）",
                            show_label=True,
                            elem_classes="horizontal-scroll-gallery",
                            columns=1,
                            rows=None,
                            height=600,
                            preview=False,
                            object_fit="contain",
                            selected_index=0
                        )
                        clear_batch_btn = gr.Button("清空列表", size="sm")
                    
                    process_btn = gr.Button("开始识别", variant="primary", size="lg", interactive=False)
            
            with gr.Column(scale=1, min_width=400):
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">📷 处理后图片</div>')
                    output_image = gr.Image(
                        label="",
                        type="numpy",
                        height=400
                    )
            
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">💡 识别建议</div>')
                    suggestions_text = gr.Textbox(
                        label="",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">📊 统计信息</div>')
                    stats_html = gr.HTML(
                        value='<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
                    )
                    
                    with gr.Group(visible=False) as batch_stats_group:
                        batch_stats_html = gr.HTML(
                            value='<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总图片数: <strong>0</strong></div><div>成功: <strong>0</strong></div><div>失败: <strong>0</strong></div><div>成功率: <strong>0%</strong></div></div>'
                        )
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">⚙️ 设置区域</div>')
                    model_combo = gr.Dropdown(
                        choices=["U-Net (推荐)", "ResUNet"],
                        value="U-Net (推荐)",
                        label="模型选择"
                    )
                    confidence_spin = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="置信度阈值"
                    )
                    line_width_spin = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="线条粗细"
                    )
                    enhance_check = gr.Checkbox(
                        value=True,
                        label="启用图像增强"
                    )
                    save_overlay_check = gr.Checkbox(
                        value=True,
                        label="保存叠加图像"
                    )
                    save_json_check = gr.Checkbox(
                        value=True,
                        label="保存JSON结果"
                    )
            
            with gr.Column(scale=3, min_width=900):
                with gr.Group(elem_classes="group-box"):
                    gr.HTML('<div class="group-box-title">📈 线条概览</div>')
                    lines_html = gr.HTML(
                        value='<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'
                    )
        
        def on_mode_change(mode):
            """处理模式切换"""
            # 只有批量处理模式
            return (
                gr.update(visible=True),   # upload_controls
                gr.update(visible=False),  # original_images_group
                gr.update(visible=True),   # batch_stats_group
                "开始批量识别"
            )
        
        def on_batch_files_change(files):
            """批量文件选择变化"""
            if not files:
                return gr.update(value=[]), gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)
            
            original_images_list = []
            for f in files:
                if hasattr(f, 'name'):
                    try:
                        # 尝试使用 OpenCV 读取
                        img = cv2.imread(f.name)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            # 尝试使用 PIL 读取（支持更多格式如 webp）
                            from PIL import Image
                            import numpy as np
                            img_pil = Image.open(f.name)
                            img_rgb = np.array(img_pil.convert('RGB'))
                        
                        # 调整图片尺寸
                        max_height = 600
                        max_width = 800
                        
                        if img_rgb.shape[0] > max_height or img_rgb.shape[1] > max_width:
                            scale = min(max_height / img_rgb.shape[0], max_width / img_rgb.shape[1])
                            new_height = int(img_rgb.shape[0] * scale)
                            new_width = int(img_rgb.shape[1] * scale)
                            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        original_images_list.append(img_rgb)
                    except Exception as e:
                        print(f"处理图片失败 {f.name}: {e}")
            
            return gr.update(value=original_images_list), gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)
        
        def on_clear_batch():
            """清空批量列表"""
            app.batch_results = []
            app.batch_images = []
            app.current_batch_index = 0
            
            stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
            batch_stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总图片数: <strong>0</strong></div><div>成功: <strong>0</strong></div><div>失败: <strong>0</strong></div><div>成功率: <strong>0%</strong></div></div>'
            
            return (
                gr.update(value=None), 
                gr.update(value=[], selected_index=0), 
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(interactive=False),
                gr.update(value=np.zeros((400, 400, 3), dtype=np.uint8)),
                gr.update(value=stats_html_val),
                gr.update(value=""),
                gr.update(value='<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'),
                gr.update(value=batch_stats_html_val),
                gr.update(visible=False)
            )
        
        def on_select_original_image(evt: gr.SelectData):
            """选择原图"""
            if not app.batch_results:
                stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
                return (
                    gr.update(value=np.zeros((400, 400, 3), dtype=np.uint8)),
                    gr.update(value=stats_html_val),
                    gr.update(value=""),
                    gr.update(value='<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>')
                )
            
            try:
                index = evt.index
            except (AttributeError, KeyError, TypeError):
                index = 0
            
            if index is None or index < 0 or index >= len(app.batch_results):
                stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
                return (
                    gr.update(value=np.zeros((400, 400, 3), dtype=np.uint8)),
                    gr.update(value=stats_html_val),
                    gr.update(value=""),
                    gr.update(value='<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>')
                )
            
            app.current_batch_index = index
            
            original, overlay, result_info, time_val, size_val, status_val, suggestions, lines_html_val = app.get_batch_image(index)
            
            stats_html_val = f'<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>{result_info}</strong></div><div>处理时间: <strong>{time_val}</strong></div><div>图像尺寸: <strong>{size_val}</strong></div><div>检测状态: <strong>{status_val}</strong></div></div>'
            
            return (
                gr.update(value=overlay),
                gr.update(value=stats_html_val),
                gr.update(value=suggestions),
                gr.update(value=lines_html_val)
            )
        
        def on_process_single(image):
            """处理单张图片"""
            if image is None:
                stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
                return (
                    gr.update(value=np.zeros((400, 400, 3), dtype=np.uint8)),
                    gr.update(value=stats_html_val),
                    gr.update(value=""),
                    gr.update(value='<div style="padding: 20px; text-align: center; color: #999;">请选择图片</div>')
                )
            
            output_img, result_info, processing_info, suggestions, table = app.process_single_image(image)
            
            time_val = "--"
            size_val = "--"
            status_val = "失败"
            
            if "处理时间" in processing_info:
                lines = processing_info.split('\n')
                for line in lines:
                    if "处理时间" in line:
                        time_val = line.split(': ')[1] if ': ' in line else "--"
                    elif "图像尺寸" in line:
                        size_val = line.split(': ')[1] if ': ' in line else "--"
                    elif "检测状态" in line:
                        status_val = line.split(': ')[1] if ': ' in line else "失败"
            
            stats_html_val = f'<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>{result_info}</strong></div><div>处理时间: <strong>{time_val}</strong></div><div>图像尺寸: <strong>{size_val}</strong></div><div>检测状态: <strong>{status_val}</strong></div></div>'
            lines_html_val = app.format_lines_html(app.current_result) if app.current_result and app.current_result.success else '<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'
            
            return (
                gr.update(value=output_img),
                gr.update(value=stats_html_val),
                gr.update(value=suggestions),
                gr.update(value=lines_html_val)
            )
        
        def on_process_batch(files, progress=gr.Progress()):
            """处理批量图片"""
            if not files:
                stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>--</strong></div><div>处理时间: <strong>--</strong></div><div>图像尺寸: <strong>--</strong></div><div>检测状态: <strong>--</strong></div></div>'
                batch_stats_html_val = '<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总图片数: <strong>0</strong></div><div>成功: <strong>0</strong></div><div>失败: <strong>0</strong></div><div>成功率: <strong>0%</strong></div></div>'
                return (
                    gr.update(value=np.zeros((400, 400, 3), dtype=np.uint8)),
                    gr.update(value=stats_html_val),
                    gr.update(value=""),
                    gr.update(value='<div style="padding: 20px; text-align: center; color: #999;">请选择图片</div>'),
                    gr.update(value=batch_stats_html_val)
                )
            
            status, stats, original_images, overlay_images, total, success, failed, rate, batch_results = app.process_batch_images(files, progress)
            
            if app.batch_results and app.batch_results[0]['success']:
                first_result = app.batch_results[0]['result']
                result_info = f"{first_result.confidences.get('total', 0):.2f}"
                time_val = f"{first_result.processing_time:.2f}秒"
                size_val = f"{first_result.image_size[1]}x{first_result.image_size[0]}" if first_result.image_size else "--"
                status_val = "成功"
                suggestions = "\n".join(first_result.suggestions) if first_result.suggestions else "无建议"
                lines_html_val = app.format_lines_html(first_result)
                first_overlay = overlay_images[0] if overlay_images else np.zeros((400, 400, 3), dtype=np.uint8)
            else:
                result_info = "--"
                time_val = "--"
                size_val = "--"
                status_val = "--"
                suggestions = ""
                lines_html_val = '<div style="padding: 20px; text-align: center; color: #999;">暂无线条数据</div>'
                first_overlay = np.zeros((400, 400, 3), dtype=np.uint8)
            
            app.current_batch_index = 0
            
            stats_html_val = f'<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总置信度: <strong>{result_info}</strong></div><div>处理时间: <strong>{time_val}</strong></div><div>图像尺寸: <strong>{size_val}</strong></div><div>检测状态: <strong>{status_val}</strong></div></div>'
            batch_stats_html_val = f'<div style="padding: 10px; font-size: 14px; line-height: 1.8;"><div>总图片数: <strong>{total}</strong></div><div>成功: <strong>{success}</strong></div><div>失败: <strong>{failed}</strong></div><div>成功率: <strong>{rate}</strong></div></div>'
            
            return (
                gr.update(value=first_overlay),
                gr.update(value=stats_html_val),
                gr.update(value=suggestions),
                gr.update(value=lines_html_val),
                gr.update(value=batch_stats_html_val)
            )
        
        def on_process(mode, batch_files):
            """根据模式处理图片"""
            # 只有批量处理模式
            result = on_process_batch(batch_files)
            return result + (gr.update(visible=True),)
        
        mode_radio.change(
            on_mode_change,
            inputs=[mode_radio],
            outputs=[upload_controls, original_images_group, batch_stats_group, process_btn]
        )
        
        batch_input.change(
            on_batch_files_change,
            inputs=[batch_input],
            outputs=[original_images, original_images_group, upload_controls, process_btn]
        )
        
        clear_batch_btn.click(
            on_clear_batch,
            outputs=[
                batch_input, 
                original_images, 
                original_images_group, 
                upload_controls, 
                process_btn,
                output_image,
                stats_html,
                suggestions_text,
                lines_html,
                batch_stats_html,
                batch_stats_group
            ]
        )
        
        original_images.select(
            on_select_original_image,
            outputs=[
                output_image,
                stats_html,
                suggestions_text,
                lines_html
            ]
        )
        
        config_inputs = [model_combo, confidence_spin, line_width_spin, enhance_check, save_overlay_check, save_json_check]
        for inp in config_inputs:
            inp.change(
                app.update_config,
                inputs=config_inputs,
                outputs=[]
            )
        
        process_btn.click(
            on_process,
            inputs=[mode_radio, batch_input],
            outputs=[
                output_image,
                stats_html,
                suggestions_text,
                lines_html,
                batch_stats_html,
                batch_stats_group
            ]
        )
    
    return demo


if __name__ == '__main__':
    app = WebApp()
    demo = create_web_app()
    demo.launch(server_name="0.0.0.0", server_port=7865)
