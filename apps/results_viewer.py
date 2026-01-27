#!/usr/bin/env python3
"""
结果查看器 - 显示详细结果
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QGroupBox, QGridLayout, QWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
import json


class ResultsViewer(QDialog):
    """结果查看器对话框"""
    
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setup_ui()
        self.load_result()
    
    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle("识别结果详情")
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout(self)
        
        # 标签页
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 结果概览标签页
        self.overview_tab = self.create_overview_tab()
        self.tabs.addTab(self.overview_tab, "结果概览")
        
        # 线条详情标签页
        self.lines_tab = self.create_lines_tab()
        self.tabs.addTab(self.lines_tab, "线条详情")
        
        # 原始数据标签页
        self.raw_data_tab = self.create_raw_data_tab()
        self.tabs.addTab(self.raw_data_tab, "原始数据")
        
        # 按钮
        button_layout = QHBoxLayout()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        
        export_btn = QPushButton("导出JSON")
        export_btn.clicked.connect(self.export_json)
        
        button_layout.addStretch()
        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_overview_tab(self):
        """创建结果概览标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 状态显示
        status_group = QGroupBox("识别状态")
        status_layout = QGridLayout()
        
        self.status_label = QLabel()
        status_layout.addWidget(QLabel("状态:"), 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        
        self.time_label = QLabel()
        status_layout.addWidget(QLabel("处理时间:"), 1, 0)
        status_layout.addWidget(self.time_label, 1, 1)
        
        self.size_label = QLabel()
        status_layout.addWidget(QLabel("图像尺寸:"), 2, 0)
        status_layout.addWidget(self.size_label, 2, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 置信度显示
        confidence_group = QGroupBox("掌纹置信度")
        confidence_layout = QGridLayout()
        
        self.heart_label = QLabel()
        self.head_label = QLabel()
        self.life_label = QLabel()
        
        confidence_layout.addWidget(QLabel("感情线:"), 0, 0)
        confidence_layout.addWidget(self.heart_label, 0, 1)
        confidence_layout.addWidget(QLabel("智慧线:"), 1, 0)
        confidence_layout.addWidget(self.head_label, 1, 1)
        confidence_layout.addWidget(QLabel("生命线:"), 2, 0)
        confidence_layout.addWidget(self.life_label, 2, 1)
        
        confidence_group.setLayout(confidence_layout)
        layout.addWidget(confidence_group)
        
        # 建议
        suggestions_group = QGroupBox("建议")
        suggestions_layout = QVBoxLayout()
        
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        self.suggestions_text.setMaximumHeight(100)
        suggestions_layout.addWidget(self.suggestions_text)
        
        suggestions_group.setLayout(suggestions_layout)
        layout.addWidget(suggestions_group)
        
        layout.addStretch()
        return widget
    
    def create_lines_tab(self):
        """创建线条详情标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 线条数据表格
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(5)
        self.lines_table.setHorizontalHeaderLabels([
            "线条名称", "置信度", "点数", "状态", "备注"
        ])
        
        layout.addWidget(self.lines_table)
        layout.addStretch()
        return widget
    
    def create_raw_data_tab(self):
        """创建原始数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.raw_data_text = QTextEdit()
        self.raw_data_text.setReadOnly(True)
        self.raw_data_text.setFont(QFont("Consolas", 10))
        
        layout.addWidget(self.raw_data_text)
        return widget
    
    def load_result(self):
        """加载结果数据"""
        # 更新概览标签页
        if self.result.success:
            self.status_label.setText("✅ 成功")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("❌ 失败")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.time_label.setText(f"{self.result.processing_time:.2f} 秒")
        
        if self.result.image_size:
            self.size_label.setText(f"{self.result.image_size[0]} × {self.result.image_size[1]}")
        else:
            self.size_label.setText("未知")
        
        # 更新置信度
        if self.result.confidences:
            for line_name, confidence in self.result.confidences.items():
                label_name = f"{line_name}_label"
                if hasattr(self, label_name):
                    label = getattr(self, label_name)
                    label.setText(f"{confidence:.0%}")
                    
                    # 设置颜色
                    if confidence >= 0.7:
                        color = "green"
                    elif confidence >= 0.3:
                        color = "orange"
                    else:
                        color = "red"
                    
                    label.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # 更新建议
        if self.result.suggestions:
            suggestions_text = "\n".join(f"• {suggestion}" for suggestion in self.result.suggestions)
            self.suggestions_text.setText(suggestions_text)
        
        # 更新线条表格
        self.update_lines_table()
        
        # 更新原始数据
        self.update_raw_data()
    
    def update_lines_table(self):
        """更新线条表格"""
        if not self.result.lines_data:
            return
        
        self.lines_table.setRowCount(len(self.result.lines_data))
        
        for row, line_data in enumerate(self.result.lines_data):
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
    
    def update_raw_data(self):
        """更新原始数据"""
        result_dict = {
            'success': self.result.success,
            'confidences': self.result.confidences,
            'processing_time': self.result.processing_time,
            'error_message': self.result.error_message,
            'suggestions': self.result.suggestions,
            'image_size': self.result.image_size,
            'lines_data': self.result.lines_data
        }
        
        raw_data = json.dumps(result_dict, ensure_ascii=False, indent=2)
        self.raw_data_text.setText(raw_data)
    
    def export_json(self):
        """导出JSON数据"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出JSON数据", "palm_lines_data.json", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                result_dict = {
                    'success': self.result.success,
                    'confidences': self.result.confidences,
                    'processing_time': self.result.processing_time,
                    'error_message': self.result.error_message,
                    'suggestions': self.result.suggestions,
                    'image_size': self.result.image_size,
                    'lines_data': self.result.lines_data
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, ensure_ascii=False, indent=2)
                
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "导出成功", f"数据已导出到:\n{file_path}")
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "导出失败", f"导出失败:\n{str(e)}")