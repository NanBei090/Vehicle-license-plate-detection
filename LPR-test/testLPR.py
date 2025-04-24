import cv2
import numpy as np
from hyperlpr3 import LicensePlateCatcher
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt  # 使用matplotlib来显示图像

# 1. 加载图像
img = cv2.imread("images/test04.png")

# 2. 使用 hyperlpr3 进行车牌识别
lpc = LicensePlateCatcher()
results = lpc(img)

# 3. 输出结果
print("识别结果：", results)

# 4. 如果有车牌识别结果，进行可视化
if results:
    for result in results:
        plate_text = result[0]  # 车牌号
        score = result[1]  # 置信度
        bbox = result[3]  # 边界框 [x1, y1, x2, y2]

        print(f"车牌号: {plate_text}, 置信度: {score}")
        print(f"边界框: {bbox}")

        # 5. 在图像上绘制识别到的车牌框
        # 转换为 RGB 格式的 PIL 图像
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)

        # 画出车牌框（红色）
        draw.rectangle([tuple(bbox[:2]), tuple(bbox[2:])], outline='red', width=3)

        # 6. 转换回 OpenCV 格式的图像
        img_result = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # 7. 使用 matplotlib 显示结果
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))  # 转换为RGB显示
        plt.axis('off')  # 不显示坐标轴
        plt.show()
else:
    print("未识别到车牌")
