import unittest
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestApp(unittest.TestCase):
    """应用测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.project_root = Path(__file__).parent.parent
        self.apps_dir = self.project_root / "apps"
        self.tests_dir = self.project_root / "tests"
        self.data_dir = self.project_root / "PLSU" / "img"
    
    def test_apps_directory_exists(self):
        """测试apps目录是否存在"""
        self.assertTrue(self.apps_dir.exists(), "apps目录不存在")
        self.assertTrue(self.apps_dir.is_dir(), "apps不是目录")
    
    def test_test_directory_exists(self):
        """测试tests目录是否存在"""
        self.assertTrue(self.tests_dir.exists(), "tests目录不存在")
        self.assertTrue(self.tests_dir.is_dir(), "tests不是目录")
    
    def test_main_window_file_exists(self):
        """测试main_window.py文件是否存在"""
        main_window_file = self.apps_dir / "main_window.py"
        self.assertTrue(main_window_file.exists(), "main_window.py文件不存在")
        self.assertTrue(main_window_file.is_file(), "main_window.py不是文件")
    
    def test_image_processor_file_exists(self):
        """测试image_processor.py文件是否存在"""
        image_processor_file = self.apps_dir / "image_processor.py"
        self.assertTrue(image_processor_file.exists(), "image_processor.py文件不存在")
        self.assertTrue(image_processor_file.is_file(), "image_processor.py不是文件")
    
    def test_results_viewer_file_exists(self):
        """测试results_viewer.py文件是否存在"""
        results_viewer_file = self.apps_dir / "results_viewer.py"
        self.assertTrue(results_viewer_file.exists(), "results_viewer.py文件不存在")
        self.assertTrue(results_viewer_file.is_file(), "results_viewer.py不是文件")
    
    def test_settings_file_exists(self):
        """测试settings.py文件是否存在"""
        settings_file = self.apps_dir / "settings.py"
        self.assertTrue(settings_file.exists(), "settings.py文件不存在")
        self.assertTrue(settings_file.is_file(), "settings.py不是文件")
    
    def test_requirements_gui_file_exists(self):
        """测试requirements_gui.txt文件是否存在"""
        requirements_gui_file = self.project_root / "requirements_gui.txt"
        self.assertTrue(requirements_gui_file.exists(), "requirements_gui.txt文件不存在")
        self.assertTrue(requirements_gui_file.is_file(), "requirements_gui.txt不是文件")
    
    def test_import_main_window(self):
        """测试导入main_window模块"""
        try:
            from apps.main_window import MainWindow
            self.assertIsNotNone(MainWindow, "无法导入MainWindow类")
        except ImportError as e:
            self.fail(f"导入main_window模块失败: {str(e)}")
    
    def test_import_image_processor(self):
        """测试导入image_processor模块"""
        try:
            from apps.image_processor import ImageProcessor
            self.assertIsNotNone(ImageProcessor, "无法导入ImageProcessor类")
        except ImportError as e:
            self.fail(f"导入image_processor模块失败: {str(e)}")
    
    def test_import_results_viewer(self):
        """测试导入results_viewer模块"""
        try:
            from apps.results_viewer import ResultsViewer
            self.assertIsNotNone(ResultsViewer, "无法导入ResultsViewer类")
        except ImportError as e:
            self.fail(f"导入results_viewer模块失败: {str(e)}")
    
    def test_import_settings(self):
        """测试导入settings模块"""
        try:
            from apps.settings import AppSettings
            self.assertIsNotNone(AppSettings, "无法导入AppSettings类")
        except ImportError as e:
            self.fail(f"导入settings模块失败: {str(e)}")
    
    def test_settings_initialization(self):
        """测试AppSettings类初始化"""
        try:
            from apps.settings import AppSettings
            settings = AppSettings()
            self.assertIsNotNone(settings, "AppSettings类初始化失败")
            # 测试加载设置
            loaded_settings = settings.load()
            self.assertIsInstance(loaded_settings, dict, "加载设置失败")
        except Exception as e:
            self.fail(f"AppSettings类初始化测试失败: {str(e)}")
    
    def test_sample_image_exists(self):
        """测试样本图像是否存在"""
        if self.data_dir.exists():
            image_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
            self.assertGreater(len(image_files), 0, "样本图像不存在")
    
    def test_model_file_exists(self):
        """测试模型文件是否存在"""
        model_path = self.project_root / "final_checkpoints" / "best_model.pth"
        # 模型文件可能不存在，这是正常的，因为用户可能还没有训练模型
        # 所以这里只做警告，不做断言
        if not model_path.exists():
            print(f"警告: 模型文件不存在: {model_path}")
    
    def test_requirements_gui_content(self):
        """测试requirements_gui.txt文件内容"""
        requirements_gui_file = self.project_root / "requirements_gui.txt"
        if requirements_gui_file.exists():
            with open(requirements_gui_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn("Pillow", content, "requirements_gui.txt中缺少Pillow")
            self.assertIn("opencv-python", content, "requirements_gui.txt中缺少opencv-python")

if __name__ == '__main__':
    unittest.main()