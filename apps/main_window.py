#!/usr/bin/env python3
"""
手掌掌纹识别桌面应用 - 主窗口
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QTabWidget, QGroupBox, QGridLayout, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QProgressBar, QStatusBar, QSplitter, QAction,
    QToolBar, QMenuBar, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor

from .image_processor import ImageProcessor, ProcessResult
from .settings import AppSettings


class ProcessingThread(QThread):
    """处理线程"""
    
    progress_signal = pyqtSignal(int, str)  # 进度值, 状态消息
    result_signal = pyqtSignal(ProcessResult)  # 处理结果
    error_signal = pyqtSignal(str)  # 错误消息
    
    def __init__(self, image_path, config):
        super().__init__()
        self.image_path = image_path
        self.config = config
        self.processor = ImageProcessor(config)
    
    def run(self):
        try:
            self.progress_signal.emit(10, "正在加载模型...")
            if not self.processor.load_model():
                self.error_signal.emit("模型加载失败")
                return
            
            self.progress_signal.emit(30, "正在处理图像...")
            result = self.processor.process_image(self.image_path)
            
            self.progress_signal.emit(90, "正在生成结果...")
            self.result_signal.emit(result)
            self.progress_signal.emit(100, "处理完成")
            
        except Exception as e:
            self.error_signal.emit(str(e))


class BatchProcessingThread(QThread):
    """批量处理线程"""
    
    progress_signal = pyqtSignal(int, str, int, int)  # 进度值, 状态消息, 当前, 总数
    file_result_signal = pyqtSignal(str, ProcessResult)  # 文件名, 处理结果
    batch_complete_signal = pyqtSignal(dict)  # 批量统计
    
    def __init__(self, image_paths, config):
        super().__init__()
        self.image_paths = image_paths
        self.config = config
        self.processor = ImageProcessor(config)
    
    def run(self):
        try:
            total = len(self.image_paths)
            if total == 0:
                self.error_signal.emit("没有要处理的图片")
                return
            
            self.progress_signal.emit(10, "正在加载模型...", 0, total)
            if not self.processor.load_model():
                self.error_signal.emit("模型加载失败")
                return
            
            batch_stats = {
                'total': total,
                'successful': 0,
                'failed': 0,
                'results': []
            }
            
            for i, image_path in enumerate(self.image_paths):
                self.progress_signal.emit(
                    10 + int(80 * i / total),
                    f"正在处理: {Path(image_path).name}",
                    i + 1,
                    total
                )
                
                try:
                    result = self.processor.process_image(image_path)
                    batch_stats['results'].append({
                        'filename': Path(image_path).name,
                        'success': result.success,
                        'confidences': result.confidences,
                        'processing_time': result.processing_time,
                        'error_message': result.error_message
                    })
                    
                    if result.success:
                        batch_stats['successful'] += 1
                    else:
                        batch_stats['failed'] += 1
                    
                    self.file_result_signal.emit(image_path, result)
                    
                except Exception as e:
                    batch_stats['results'].append({
                        'filename': Path(image_path).name,
                        'success': False,
                        'error_message': str(e)
                    })
                    batch_stats['failed'] += 1
            
            self.progress_signal.emit(95, "正在保存结果...", total, total)
            self.batch_complete_signal.emit(batch_stats)
            self.progress_signal.emit(100, "批量处理完成", total, total)
            
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.settings = AppSettings()
        self.current_image_path = None
        self.current_result = None
        self.batch_results = []
        self.batch_mode = False
        
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_connections()
        
        # 设置窗口属性
        self.setWindowTitle("手掌掌纹识别系统 v1.0")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(self.style().standardIcon(getattr(self.style(), 'SP_FileIcon')))
        
        # 加载上次的设置
        self.load_settings()
    
    def setup_ui(self):
        """设置UI界面"""
        # 中心窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 分隔线 - 包含左侧和右侧面板
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧显示区域
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 上部：结果图片和信息区域
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        
        # 结果图片显示
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.image_display.setText("请选择图片进行识别")
        top_layout.addWidget(self.image_display, 2)
        
        # 右侧：结果信息和建议
        info_section = QWidget()
        info_layout = QVBoxLayout(info_section)
        
        # 结果信息
        self.results_info_group = QGroupBox("结果信息")
        results_info_layout = QVBoxLayout()
        
        # 总置信度显示
        self.total_confidence_group = QGroupBox("总置信度")
        total_confidence_layout = QVBoxLayout()
        
        self.total_confidence_label = QLabel("总置信度: --")
        
        total_confidence_layout.addWidget(self.total_confidence_label)
        self.total_confidence_group.setLayout(total_confidence_layout)
        results_info_layout.addWidget(self.total_confidence_group)
        
        # 处理信息
        self.info_group = QGroupBox("处理信息")
        info_layout_grid = QGridLayout()
        
        info_layout_grid.addWidget(QLabel("处理时间:"), 0, 0)
        self.time_label = QLabel("--")
        info_layout_grid.addWidget(self.time_label, 0, 1)
        
        info_layout_grid.addWidget(QLabel("图像尺寸:"), 1, 0)
        self.size_label = QLabel("--")
        info_layout_grid.addWidget(self.size_label, 1, 1)
        
        info_layout_grid.addWidget(QLabel("检测状态:"), 2, 0)
        self.status_label = QLabel("--")
        info_layout_grid.addWidget(self.status_label, 2, 1)
        
        self.info_group.setLayout(info_layout_grid)
        results_info_layout.addWidget(self.info_group)
        
        self.results_info_group.setLayout(results_info_layout)
        info_layout.addWidget(self.results_info_group)
        
        # 建议区域
        self.suggestions_group = QGroupBox("识别建议")
        suggestions_layout = QVBoxLayout()
        
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        self.suggestions_text.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")
        suggestions_layout.addWidget(self.suggestions_text)
        
        self.suggestions_group.setLayout(suggestions_layout)
        info_layout.addWidget(self.suggestions_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("保存结果")
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self.save_results)
        
        self.export_json_btn = QPushButton("导出JSON")
        self.export_json_btn.setEnabled(False)
        self.export_json_btn.clicked.connect(self.export_json)
        
        button_layout.addWidget(self.save_result_btn)
        button_layout.addWidget(self.export_json_btn)
        info_layout.addLayout(button_layout)
        
        # 批量统计
        self.batch_stats_group = QGroupBox("批量统计")
        batch_stats_layout = QVBoxLayout()
        
        self.batch_total_label = QLabel("总图片数: 0")
        self.batch_success_label = QLabel("成功: 0")
        self.batch_failed_label = QLabel("失败: 0")
        self.batch_success_rate_label = QLabel("成功率: 0%")
        
        batch_stats_layout.addWidget(self.batch_total_label)
        batch_stats_layout.addWidget(self.batch_success_label)
        batch_stats_layout.addWidget(self.batch_failed_label)
        batch_stats_layout.addWidget(self.batch_success_rate_label)
        self.batch_stats_group.setLayout(batch_stats_layout)
        self.batch_stats_group.setVisible(False)
        info_layout.addWidget(self.batch_stats_group)
        
        info_layout.addStretch()
        info_section.setLayout(info_layout)
        top_layout.addWidget(info_section, 1)
        
        right_layout.addWidget(top_section)
        
        # 下部：线条概览区域
        self.lines_overview_group = QGroupBox("线条概览")
        lines_layout = QVBoxLayout()
        
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(5)
        self.lines_table.setHorizontalHeaderLabels([
            "线条名称", "置信度", "点数", "状态", "备注"
        ])
        
        lines_layout.addWidget(self.lines_table)
        self.lines_overview_group.setLayout(lines_layout)
        right_layout.addWidget(self.lines_overview_group)
        
        splitter.addWidget(right_panel)
        
        splitter.setSizes([250, 1150])
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 模式选择
        mode_group = QGroupBox("处理模式")
        mode_layout = QVBoxLayout()
        
        self.single_mode_btn = QPushButton("单张图片模式")
        self.single_mode_btn.setCheckable(True)
        self.single_mode_btn.setChecked(True)
        self.single_mode_btn.clicked.connect(self.set_single_mode)
        
        self.batch_mode_btn = QPushButton("批量处理模式")
        self.batch_mode_btn.setCheckable(True)
        self.batch_mode_btn.clicked.connect(self.set_batch_mode)
        
        mode_layout.addWidget(self.single_mode_btn)
        mode_layout.addWidget(self.batch_mode_btn)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 单张图片控制
        self.single_group = QGroupBox("单张图片")
        single_layout = QVBoxLayout()
        
        self.select_image_btn = QPushButton("选择图片")
        self.select_image_btn.clicked.connect(self.select_image)
        
        self.image_label = QLabel("未选择图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
        
        single_layout.addWidget(self.select_image_btn)
        single_layout.addWidget(self.image_label)
        self.single_group.setLayout(single_layout)
        layout.addWidget(self.single_group)
        
        # 批量处理控制
        self.batch_group = QGroupBox("批量处理")
        batch_layout = QVBoxLayout()
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        
        self.batch_list = QListWidget()
        self.batch_list.setMaximumHeight(150)
        
        self.clear_batch_btn = QPushButton("清空列表")
        self.clear_batch_btn.clicked.connect(self.clear_batch_list)
        
        batch_layout.addWidget(self.select_folder_btn)
        batch_layout.addWidget(self.batch_list)
        batch_layout.addWidget(self.clear_batch_btn)
        self.batch_group.setLayout(batch_layout)
        self.batch_group.setVisible(False)
        layout.addWidget(self.batch_group)
        
        # 处理按钮
        self.process_btn = QPushButton("开始识别")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)
        
        # 配置选项
        config_group = QGroupBox("配置选项")
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("模型选择:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["U-Net (推荐)", "ResUNet"])
        config_layout.addWidget(self.model_combo, 0, 1)
        
        config_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.3)
        self.confidence_spin.setSingleStep(0.1)
        config_layout.addWidget(self.confidence_spin, 1, 1)
        
        config_layout.addWidget(QLabel("线条粗细:"), 2, 0)
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(3)
        config_layout.addWidget(self.line_width_spin, 2, 1)
        
        self.enhance_check = QCheckBox("启用图像增强")
        self.enhance_check.setChecked(True)
        config_layout.addWidget(self.enhance_check, 3, 0, 1, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 输出选项
        output_group = QGroupBox("输出选项")
        output_layout = QVBoxLayout()
        
        self.save_overlay_check = QCheckBox("保存叠加图像")
        self.save_overlay_check.setChecked(True)
        output_layout.addWidget(self.save_overlay_check)
        
        self.save_json_check = QCheckBox("保存JSON结果")
        self.save_json_check.setChecked(True)
        output_layout.addWidget(self.save_json_check)
        
        self.auto_open_check = QCheckBox("处理后自动打开结果")
        self.auto_open_check.setChecked(True)
        output_layout.addWidget(self.auto_open_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self):
        """创建右侧显示面板"""
        # 这个方法现在已经不再使用，因为 image_display 已经移到了 setup_ui 方法中
        panel = QWidget()
        layout = QVBoxLayout(panel)
        return panel
    
    def create_results_panel(self):
        """创建建议窗口"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 建议标签
        self.results_label = QLabel("识别建议")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(self.results_label)
        
        # 建议文本
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        self.suggestions_text.setMaximumHeight(200)
        self.suggestions_text.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")
        layout.addWidget(self.suggestions_text)
        
        # 结果信息区域
        self.result_info_group = QGroupBox("结果信息")
        result_info_layout = QVBoxLayout()
        
        # 总置信度显示
        self.total_confidence_group = QGroupBox("总置信度")
        total_confidence_layout = QVBoxLayout()
        
        self.total_confidence_label = QLabel("总置信度: --")
        
        total_confidence_layout.addWidget(self.total_confidence_label)
        self.total_confidence_group.setLayout(total_confidence_layout)
        result_info_layout.addWidget(self.total_confidence_group)
        
        # 处理信息
        self.info_group = QGroupBox("处理信息")
        info_layout = QGridLayout()
        
        info_layout.addWidget(QLabel("处理时间:"), 0, 0)
        self.time_label = QLabel("--")
        info_layout.addWidget(self.time_label, 0, 1)
        
        info_layout.addWidget(QLabel("图像尺寸:"), 1, 0)
        self.size_label = QLabel("--")
        info_layout.addWidget(self.size_label, 1, 1)
        
        info_layout.addWidget(QLabel("检测状态:"), 2, 0)
        self.status_label = QLabel("--")
        info_layout.addWidget(self.status_label, 2, 1)
        
        self.info_group.setLayout(info_layout)
        result_info_layout.addWidget(self.info_group)
        
        self.result_info_group.setLayout(result_info_layout)
        layout.addWidget(self.result_info_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.save_result_btn = QPushButton("保存结果")
        self.save_result_btn.setEnabled(False)
        self.save_result_btn.clicked.connect(self.save_results)
        
        self.export_json_btn = QPushButton("导出JSON")
        self.export_json_btn.setEnabled(False)
        self.export_json_btn.clicked.connect(self.export_json)
        
        button_layout.addWidget(self.save_result_btn)
        button_layout.addWidget(self.export_json_btn)
        layout.addLayout(button_layout)
        
        # 批量统计
        self.batch_stats_group = QGroupBox("批量统计")
        batch_stats_layout = QVBoxLayout()
        
        self.batch_total_label = QLabel("总图片数: 0")
        self.batch_success_label = QLabel("成功: 0")
        self.batch_failed_label = QLabel("失败: 0")
        self.batch_success_rate_label = QLabel("成功率: 0%")
        
        batch_stats_layout.addWidget(self.batch_total_label)
        batch_stats_layout.addWidget(self.batch_success_label)
        batch_stats_layout.addWidget(self.batch_failed_label)
        batch_stats_layout.addWidget(self.batch_success_rate_label)
        self.batch_stats_group.setLayout(batch_stats_layout)
        self.batch_stats_group.setVisible(False)
        layout.addWidget(self.batch_stats_group)
        
        layout.addStretch()
        return panel
    
    def setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开图片", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_image)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction("打开文件夹", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("保存结果", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        export_action = QAction("导出所有结果", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_all_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 处理菜单
        process_menu = menubar.addMenu("处理")
        
        process_action = QAction("开始识别", self)
        process_action.setShortcut("Ctrl+R")
        process_action.triggered.connect(self.start_processing)
        process_menu.addAction(process_action)
        
        batch_process_action = QAction("批量识别", self)
        batch_process_action.setShortcut("Ctrl+Shift+R")
        batch_process_action.triggered.connect(self.start_batch_processing)
        process_menu.addAction(batch_process_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        show_original_action = QAction("显示原图", self)
        show_original_action.setCheckable(True)
        show_original_action.setChecked(True)
        show_original_action.triggered.connect(self.toggle_original_view)
        view_menu.addAction(show_original_action)
        
        show_overlay_action = QAction("显示叠加图", self)
        show_overlay_action.setCheckable(True)
        show_overlay_action.setChecked(True)
        show_overlay_action.triggered.connect(self.toggle_overlay_view)
        view_menu.addAction(show_overlay_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_action = QAction("使用说明", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        
        # 打开按钮
        open_action = QAction("打开", self)
        open_action.triggered.connect(self.select_image)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # 处理按钮
        process_action = QAction("识别", self)
        process_action.triggered.connect(self.start_processing)
        toolbar.addAction(process_action)
        
        toolbar.addSeparator()
        
        # 保存按钮
        save_action = QAction("保存", self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)
        
        # 导出按钮
        export_action = QAction("导出", self)
        export_action.triggered.connect(self.export_json)
        toolbar.addAction(export_action)
    
    def setup_connections(self):
        """设置信号连接"""
        # 处理线程
        self.processing_thread = None
        self.batch_thread = None
    
    def set_single_mode(self):
        """设置为单张图片模式"""
        self.single_mode_btn.setChecked(True)
        self.batch_mode_btn.setChecked(False)
        self.single_group.setVisible(True)
        self.batch_group.setVisible(False)
        self.batch_mode = False
        self.batch_stats_group.setVisible(False)
        self.process_btn.setText("开始识别")
    
    def set_batch_mode(self):
        """设置为批量处理模式"""
        self.single_mode_btn.setChecked(False)
        self.batch_mode_btn.setChecked(True)
        self.single_group.setVisible(False)
        self.batch_group.setVisible(True)
        self.batch_mode = True
        self.batch_stats_group.setVisible(True)
        self.process_btn.setText("开始批量识别")
    
    def select_image(self):
        """选择图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择手掌图片",
            "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """加载图片"""
        self.current_image_path = file_path
        
        # 显示图片信息
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # 调整尺寸以适应显示
            scaled_pixmap = pixmap.scaled(
                400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")
            
            # 在主显示区域显示原图
            self.display_original_image()
            
            # 更新状态
            self.status_bar.showMessage(f"已加载: {Path(file_path).name}")
            self.process_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "错误", "无法加载图片")
    
    def display_original_image(self):
        """显示原图"""
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                # 调整尺寸以适应显示
                scaled_pixmap = pixmap.scaled(
                    self.image_display.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_display.setPixmap(scaled_pixmap)
    
    def select_folder(self):
        """选择文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择包含手掌图片的文件夹"
        )
        
        if folder_path:
            self.load_folder(folder_path)
    
    def load_folder(self, folder_path):
        """加载文件夹中的所有图片"""
        self.batch_list.clear()
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        folder = Path(folder_path)
        
        image_files = []
        for format in supported_formats:
            image_files.extend(folder.glob(f"*{format}"))
            image_files.extend(folder.glob(f"*{format.upper()}"))
        
        for file_path in image_files:
            item = QListWidgetItem(str(file_path.name))
            item.setData(Qt.UserRole, str(file_path))
            self.batch_list.addItem(item)
        
        self.batch_total_label.setText(f"总图片数: {len(image_files)}")
        self.status_bar.showMessage(f"已加载 {len(image_files)} 张图片")
        self.process_btn.setEnabled(len(image_files) > 0)
    
    def clear_batch_list(self):
        """清空批量列表"""
        self.batch_list.clear()
        self.process_btn.setEnabled(False)
        self.batch_total_label.setText("总图片数: 0")
    
    def start_processing(self):
        """开始处理"""
        if self.batch_mode:
            self.start_batch_processing()
        else:
            self.start_single_processing()
    
    def start_single_processing(self):
        """开始单张图片处理"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先选择图片")
            return
        
        # 禁用按钮
        self.process_btn.setEnabled(False)
        self.select_image_btn.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建处理线程
        config = {
            'model_type': 'unet' if self.model_combo.currentIndex() == 0 else 'resunet',
            'confidence_threshold': self.confidence_spin.value(),
            'line_width': self.line_width_spin.value(),
            'enhance_image': self.enhance_check.isChecked(),
            'save_overlay': self.save_overlay_check.isChecked(),
            'save_json': self.save_json_check.isChecked()
        }
        
        self.processing_thread = ProcessingThread(self.current_image_path, config)
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.result_signal.connect(self.process_result)
        self.processing_thread.error_signal.connect(self.process_error)
        self.processing_thread.finished.connect(self.processing_finished)
        
        self.status_bar.showMessage("正在处理图片...")
        self.processing_thread.start()
    
    def start_batch_processing(self):
        """开始批量处理"""
        if self.batch_list.count() == 0:
            QMessageBox.warning(self, "警告", "请先添加要处理的图片")
            return
        
        # 获取所有图片路径
        image_paths = []
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            image_paths.append(item.data(Qt.UserRole))
        
        # 禁用按钮
        self.process_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.clear_batch_btn.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建批量处理线程
        config = {
            'model_type': 'unet' if self.model_combo.currentIndex() == 0 else 'resunet',
            'confidence_threshold': self.confidence_spin.value(),
            'line_width': self.line_width_spin.value(),
            'enhance_image': self.enhance_check.isChecked(),
            'save_overlay': self.save_overlay_check.isChecked(),
            'save_json': self.save_json_check.isChecked(),
            'batch_mode': True
        }
        
        self.batch_thread = BatchProcessingThread(image_paths, config)
        self.batch_thread.progress_signal.connect(self.update_batch_progress)
        self.batch_thread.file_result_signal.connect(self.process_batch_file_result)
        self.batch_thread.batch_complete_signal.connect(self.process_batch_complete)
        self.batch_thread.error_signal.connect(self.process_error)
        self.batch_thread.finished.connect(self.batch_processing_finished)
        
        self.status_bar.showMessage("开始批量处理...")
        self.batch_thread.start()
    
    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
    
    def update_batch_progress(self, value, message, current, total):
        """更新批量处理进度"""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(f"{message} ({current}/{total})")
    
    def process_result(self, result):
        """处理单个结果"""
        self.current_result = result
        
        # 显示结果图像
        if result.success and result.overlay_image is not None:
            image = QImage(
                result.overlay_image.data,
                result.overlay_image.shape[1],
                result.overlay_image.shape[0],
                result.overlay_image.strides[0],
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.image_display.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
        
        # 更新建议文本
        self.update_suggestions_display(result)
        
        # 更新结果信息
        self.update_confidence_display(result)
        self.update_info_display(result)
        self.update_lines_table(result)
        
        # 启用按钮
        self.save_result_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
    
    def process_batch_file_result(self, file_path, result):
        """处理批量文件结果"""
        # 可以在这里更新批量处理进度
        pass
    
    def process_batch_complete(self, batch_stats):
        """批量处理完成"""
        # 更新批量统计
        total = batch_stats['total']
        successful = batch_stats['successful']
        failed = batch_stats['failed']
        success_rate = (successful / total * 100) if total > 0 else 0
        
        self.batch_total_label.setText(f"总图片数: {total}")
        self.batch_success_label.setText(f"成功: {successful}")
        self.batch_failed_label.setText(f"失败: {failed}")
        self.batch_success_rate_label.setText(f"成功率: {success_rate:.1f}%")
        
        # 保存批量结果
        self.save_batch_results(batch_stats)
        
        QMessageBox.information(
            self, "批量处理完成",
            f"批量处理完成!\n"
            f"总图片数: {total}\n"
            f"成功: {successful}\n"
            f"失败: {failed}\n"
            f"成功率: {success_rate:.1f}%"
        )
    
    def save_batch_results(self, batch_stats):
        """保存批量结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("batch_results") / f"batch_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存统计信息
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
        
        # 生成报告
        report_file = output_dir / "report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("批量掌纹识别结果报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {timestamp}\n")
            f.write(f"总图片数: {batch_stats['total']}\n")
            f.write(f"成功识别: {batch_stats['successful']}\n")
            f.write(f"识别失败: {batch_stats['failed']}\n")
            f.write(f"成功率: {success_rate:.1f}%\n\n")
            
            f.write("详细结果:\n")
            for result in batch_stats['results']:
                status = "成功" if result['success'] else "失败"
                f.write(f"{result['filename']}: {status}\n")
                if not result['success'] and 'error_message' in result:
                    f.write(f"  错误: {result['error_message']}\n")
    
    def process_error(self, error_message):
        """处理错误"""
        QMessageBox.critical(self, "处理错误", f"处理过程中发生错误:\n{error_message}")
        self.status_bar.showMessage("处理失败")
    
    def processing_finished(self):
        """单张处理完成"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.select_image_btn.setEnabled(True)
        self.status_bar.showMessage("处理完成")
    
    def batch_processing_finished(self):
        """批量处理完成"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.clear_batch_btn.setEnabled(True)
        self.status_bar.showMessage("批量处理完成")
    
    def update_suggestions_display(self, result):
        """更新建议显示"""
        if result.suggestions:
            suggestions_text = "💡 识别建议:\n\n"
            for suggestion in result.suggestions:
                suggestions_text += f"  • {suggestion}\n"
        else:
            suggestions_text = "💡 暂无建议\n\n请确保手掌清晰可见，光线充足。"
        
        self.suggestions_text.setText(suggestions_text)
    
    def update_confidence_display(self, result):
        """更新置信度显示"""
        if result.success and result.confidences:
            total_conf = result.confidences.get('total', 0)
            
            # 设置颜色：红色表示低置信度，绿色表示高置信度
            def get_color(confidence):
                if confidence >= 0.7:
                    return "green"
                elif confidence >= 0.3:
                    return "orange"
                else:
                    return "red"
            
            self.total_confidence_label.setText(
                f"总置信度: {total_conf:.0%}"
            )
            self.total_confidence_label.setStyleSheet(f"color: {get_color(total_conf)};")
        else:
            self.total_confidence_label.setText("总置信度: --")
            self.total_confidence_label.setStyleSheet("")
    
    def update_info_display(self, result):
        """更新处理信息显示"""
        self.time_label.setText(f"{result.processing_time:.2f}秒")
        
        if result.image_size:
            self.size_label.setText(f"{result.image_size[0]}×{result.image_size[1]}")
        else:
            self.size_label.setText("--")
        
        if result.success:
            self.status_label.setText("✅ 成功")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText("❌ 失败")
            self.status_label.setStyleSheet("color: red;")
    
    def update_lines_table(self, result):
        """更新线条表格"""
        if not result.success or not hasattr(result, 'lines_data') or not result.lines_data:
            self.lines_table.setRowCount(0)
            return
        
        self.lines_table.setRowCount(len(result.lines_data))
        
        for row, line_data in enumerate(result.lines_data):
            # 线条名称
            name_map = {
                'heart': '感情线',
                'head': '智慧线',
                'life': '生命线'
            }
            name = name_map.get(line_data['name'], line_data['name'])
            
            # 置信度
            confidence = line_data.get('confidence', 0)
            
            # 点数
            points = len(line_data.get('points', []))
            
            # 状态
            if confidence >= 0.7:
                status = "良好"
                status_color = "green"
            elif confidence >= 0.3:
                status = "一般"
                status_color = "orange"
            else:
                status = "较差"
                status_color = "red"
            
            # 备注
            notes = "掌纹清晰" if confidence >= 0.5 else "可能需要重新拍摄"
            
            # 设置表格项
            self.lines_table.setItem(row, 0, QTableWidgetItem(name))
            self.lines_table.setItem(row, 1, QTableWidgetItem(f"{confidence:.0%}"))
            self.lines_table.setItem(row, 2, QTableWidgetItem(str(points)))
            
            status_item = QTableWidgetItem(status)
            status_item.setForeground(Qt.white)
            status_item.setBackground(QColor(status_color))
            self.lines_table.setItem(row, 3, status_item)
            
            self.lines_table.setItem(row, 4, QTableWidgetItem(notes))
    
    def save_results(self):
        """保存结果"""
        if not self.current_result:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果",
            f"palm_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "图片文件 (*.jpg *.jpeg *.png)"
        )
        
        if file_path:
            try:
                if self.current_result.overlay_image is not None:
                    cv2.imwrite(file_path, self.current_result.overlay_image)
                    QMessageBox.information(self, "保存成功", f"结果已保存到:\n{file_path}")
                else:
                    QMessageBox.warning(self, "保存失败", "没有可保存的图像")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存失败:\n{str(e)}")
    
    def export_json(self):
        """导出JSON结果"""
        if not self.current_result:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON",
            f"palm_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                result_dict = {
                    'success': self.current_result.success,
                    'confidences': self.current_result.confidences,
                    'processing_time': self.current_result.processing_time,
                    'error_message': self.current_result.error_message,
                    'suggestions': self.current_result.suggestions,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 添加线条数据
                if hasattr(self.current_result, 'lines_data'):
                    result_dict['lines'] = self.current_result.lines_data
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
                
                QMessageBox.information(self, "导出成功", f"JSON结果已导出到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出错误", f"导出失败:\n{str(e)}")
    
    def export_all_results(self):
        """导出所有结果"""
        QMessageBox.information(self, "功能开发中", "批量导出功能正在开发中")
    
    def toggle_original_view(self):
        """切换显示原图"""
        if self.current_image_path and hasattr(self, 'show_original_action'):
            if self.show_original_action.isChecked():
                self.display_original_image()
    
    def toggle_overlay_view(self):
        """切换显示叠加图"""
        if self.current_result and hasattr(self, 'show_overlay_action'):
            if self.show_overlay_action.isChecked() and self.current_result.overlay_image is not None:
                self.display_overlay_image()
    
    def display_overlay_image(self):
        """显示叠加图像"""
        if self.current_result and self.current_result.overlay_image is not None:
            image = QImage(
                self.current_result.overlay_image.data,
                self.current_result.overlay_image.shape[1],
                self.current_result.overlay_image.shape[0],
                self.current_result.overlay_image.strides[0],
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.image_display.size(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <h2>手掌掌纹识别系统 v1.0</h2>
        <p>一个基于深度学习的自动手掌掌纹识别系统</p>
        <p>主要功能：</p>
        <ul>
            <li>自动检测手掌区域</li>
            <li>提取并识别三条主要掌纹线</li>
            <li>在原图上叠加显示结果</li>
            <li>提供置信度和失败原因分析</li>
        </ul>
        <p>支持：单张图片识别和批量处理</p>
        <p>© 2024 掌纹识别项目组</p>
        """
        QMessageBox.about(self, "关于", about_text)
    
    def show_help(self):
        """显示帮助对话框"""
        help_text = """
        <h2>使用说明</h2>
        <h3>基本操作：</h3>
        <ol>
            <li><b>选择模式：</b>单张图片模式或批量处理模式</li>
            <li><b>加载图片：</b>选择手掌图片或包含图片的文件夹</li>
            <li><b>开始识别：</b>点击"开始识别"按钮</li>
            <li><b>查看结果：</b>在右侧查看识别结果和置信度</li>
        </ol>
        
        <h3>拍摄建议：</h3>
        <ul>
            <li>确保手掌完全在画面中</li>
            <li>手心朝上，手指自然分开</li>
            <li>光线充足但避免反光</li>
            <li>背景尽量简单</li>
            <li>图像清晰，不模糊</li>
        </ul>
        
        <h3>常见问题：</h3>
        <ul>
            <li><b>识别失败：</b>请确保手掌清晰可见，调整拍摄角度</li>
            <li><b>置信度低：</b>尝试重新拍摄，改善光照条件</li>
            <li><b>无法加载图片：</b>检查图片格式是否支持（jpg、png等）</li>
        </ul>
        
        <p>如需更多帮助，请联系技术支持。</p>
        """
        QMessageBox.information(self, "使用说明", help_text)
    
    def load_settings(self):
        """加载设置"""
        try:
            settings = self.settings.load()
            if settings:
                # 加载上次的设置
                pass
        except Exception as e:
            print(f"加载设置失败: {e}")
    
    def save_settings(self):
        """保存设置"""
        try:
            settings = {
                'model_type': self.model_combo.currentIndex(),
                'confidence_threshold': self.confidence_spin.value(),
                'line_width': self.line_width_spin.value(),
                'enhance_image': self.enhance_check.isChecked(),
                'save_overlay': self.save_overlay_check.isChecked(),
                'save_json': self.save_json_check.isChecked(),
                'auto_open': self.auto_open_check.isChecked()
            }
            self.settings.save(settings)
        except Exception as e:
            print(f"保存设置失败: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.save_settings()
        event.accept()


def main():
    """主函数"""
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setApplicationName("手掌掌纹识别系统")
    app.setOrganizationName("PalmRecognition")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()