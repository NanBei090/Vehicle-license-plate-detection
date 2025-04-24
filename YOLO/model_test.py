from ultralytics import YOLO


def detect_and_show(image_path, model_path):
    """
    加载YOLO模型，对单张图像进行推理并显示结果
    """
    # 加载训练好的模型
    model = YOLO(model_path)

    # 对图像进行推理
    results = model(image_path)

    # 显示检测结果
    results[0].show()  # 显示带有检测框的图像


def main():
    # 设置模型路径和图像路径
    model_path = "./runs/detect/yolo11s-vehicle3/weights/best.pt"  # 训练好的模型路径
    image_path = "a.jpg"  # 要进行推理的图像路径

    # 对单张图像进行推理并显示结果
    detect_and_show(image_path, model_path)


if __name__ == "__main__":
    main()
