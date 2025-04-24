from ultralytics import YOLO


def detect_and_return_labels_and_coords(image_path,model):
    # 对图像进行推理
    results = model(image_path)

    # 获取检测结果
    boxes = results[0].boxes.xyxy  # 检测框的坐标 (x1, y1, x2, y2)
    labels = results[0].boxes.cls  # 类别标签
    confidences = results[0].boxes.conf  # 每个框的置信度

    # 返回每个检测框的类别标签、坐标和置信度
    output = []
    for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
        x1, y1, x2, y2 = box  # 提取框的左上角和右下角坐标
        class_label = model.names[int(label)]  # 获取对应的类别名称
        output.append({
            'class': class_label,
            'confidence': confidence.item(),
            'coordinates': (x1.item(), y1.item(), x2.item(), y2.item())  # 将坐标转换为标准格式
        })
        print(
            f"检测框 {i + 1}: 类别 = {label}:{class_label}, 置信度 = {confidence:.2f}, 坐标 = ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

    return output


def main():
    # 设置模型路径和图像路径
    model_path = "./weights/best.pt"  # 训练好的模型路径
    image_path = "test.jpg"  # 要进行推理的图像路径
    model = YOLO(model_path)
    # 获取检测结果并返回
    results = detect_and_return_labels_and_coords(image_path,model)

    # 输出结果
    print("\n检测结果:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
