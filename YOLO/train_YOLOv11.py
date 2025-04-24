from ultralytics import YOLO

def main():
    # 加载预训练 YOLOv5 模型（你可以选择其他模型，如 yolov5n.pt、yolov5m.pt 等）
    model = YOLO("yolo11s.pt")

    # 开始训练
    model.train(
        data="./dataset/data.yaml",     # 数据集配置文件
        epochs=100,                     # 训练周期数
        imgsz=640,                      # 图像大小
        batch=16,                       # 批次大小
        device=0,                       # 使用 GPU 编号或 'cpu'
        name="yolo11s-vehicle",         # 训练保存路径名称
        workers=4,                      # 加载数据的线程数
        save_period=10,                 # 每个周期保存一次模型
    )

    # 获取训练过程中保存的最佳模型
    best_model_path = model.best.weights
    print(f"最佳模型路径：{best_model_path}")

    # 在验证集上评估模型
    metrics = model.val()

    # 示例推理
    results = model("inference.jpg")  # 推理图像路径
    results[0].show()

    # 导出模型（可选 ONNX、TorchScript、CoreML 等）
    export_path = model.export(format="onnx")
    print(f"模型已导出至: {export_path}")

if __name__ == "__main__":
    main()
