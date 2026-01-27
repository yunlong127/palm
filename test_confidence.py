from apps.image_processor import ImageProcessor
import cv2

# 测试置信度计算
config = {'model_type': 'unet', 'use_gpu': True}
processor = ImageProcessor(config)
processor.load_model()

# 预测图像
results = processor.predictor.predict_single_image('PLSU/img/image1.jpg')

# 计算置信度
confidences = processor.calculate_confidences(results)
print('置信度:', confidences)

# 计算面积信息
pred_mask = results.get('full_prediction')
if pred_mask is not None:
    _, binary_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    palm_line_area = cv2.countNonZero(binary_mask)
    hand_area = processor.calculate_hand_area(binary_mask)
    area_ratio = palm_line_area / hand_area
    print('掌纹面积:', palm_line_area)
    print('手掌面积:', hand_area)
    print('面积比值:', area_ratio)
