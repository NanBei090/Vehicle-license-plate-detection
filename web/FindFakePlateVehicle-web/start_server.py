from myModel import *
from flask import Flask, render_template, redirect, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from myUtil import isFakePlate
from datetime import timedelta
import pandas as pd
from PIL import Image
import torch
from hyperlpr3 import LicensePlateCatcher  # 车牌识别库
from ultralytics import YOLO

#设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}

#vehicle_info_database = pd.read_csv('vehicle-database.csv')

cwd_path = os.getcwd()
yolo_weights_path ="./weights/best.pt"
v_color_model_path = cwd_path + "/weights/vehicleColor.pth"
v_type_model_path = cwd_path + "/weights/vehicleType.pth"

v_type_names = ["bus", "car", "minibus", "truck"]
v_color_names = ["black", "blue", "brown", "green", "red", "silver", "white", "yellow"]
global vehicle_info_database


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def prepare_service():
    global vehicle_info_database
    vehicle_info_database = pd.read_csv('vehicle-database.csv')
    return vehicle_info_database

# 初始化设备： GPU 或者 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 调用模型 # Load the trained model weights
model_yolo = YOLO(yolo_weights_path)

model_color = MiniVGGNet(num_classes=8)
model_type = MiniVGGNet(num_classes=4)
model_color.load_state_dict(torch.load(v_color_model_path))
model_type.load_state_dict(torch.load(v_type_model_path))
model_color.eval()
model_type.eval()

model_type.to(device)
model_color.to(device)

def detectVehicle(source_, model_yolo):
    # 对图像进行推理
    results = model_yolo(source_)

    # 获取检测结果
    boxes = results[0].boxes.xyxy  # 检测框的坐标 (x1, y1, x2, y2)
    if boxes is None or len(boxes) == 0:
        return "未识别", 0, 0, 0, 0  # 默认返回值，避免 None

    labels = results[0].boxes.cls  # 类别标签
    confidences = results[0].boxes.conf  # 每个框的置信度

    # 返回每个检测框的类别标签、坐标和置信度
    output = []
    for i, (box, label, confidence) in enumerate(zip(boxes, labels, confidences)):
        x1, y1, x2, y2 = box  # 提取框的左上角和右下角坐标
        class_label = model_yolo.names[int(label)]  # 获取对应的类别名称
        output.append({
            'class': class_label,
            'confidence': confidence.item(),
            'coordinates': (x1.item(), y1.item(), x2.item(), y2.item())  # 将坐标转换为标准格式
        })
        print(
            f"检测框 {i + 1}: 类别 = {label}:{class_label}, 置信度 = {confidence:.2f}, 坐标 = ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

        return class_label, x1.item(), y1.item(), x2.item(), y2.item()


# 预测模块，输入图片，分析车牌信息和子品牌，跟车辆信息库对比查询，判定是否为嫌疑车辆
def predict(imgPath, vehicle_info_database):
    predictResult = "未识别"
    plateNo = "未识别"
    carBrandZh = "未识别"
    v_type = "未识别"
    v_color = "未识别"

    original_image = cv2.imread(imgPath)
    carBrandZh, left, top, right, bottom = detectVehicle(imgPath, model_yolo)
    # 取整用于后续的图像分割
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    print(carBrandZh, left, top, right, bottom)
    # 图像分割
    crop_img = original_image[top:bottom, left:right]
    cv2.imwrite("crop_img.jpg", crop_img)
    time.sleep(0.2)

    # continue to recognize vehicle type and colorr
    if carBrandZh == '未识别':
        return plateNo, v_type, v_color, carBrandZh, predictResult

    image = Image.open("crop_img.jpg")

    # Apply the transformation on the image
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        outputs_type = model_type(image_tensor)
        _, predicted_idx = torch.max(outputs_type.data, 1)
        v_type = v_type_names[predicted_idx.item()]

        outputs_color = model_color(image_tensor)
        _, predicted_idx = torch.max(outputs_color.data, 1)
        v_color = v_color_names[predicted_idx.item()]

    lcp=LicensePlateCatcher()
    plateInfo = lcp(crop_img)

    if plateInfo:
        plateNo = plateInfo[0][0]
        inputCarInfo = [plateNo, carBrandZh]
        # print(inputCarInfo)
        isFake, true_car_brand = isFakePlate(inputCarInfo, vehicle_info_database)
        if isFake:
            predictResult = "这是一辆套牌车"
        else:
            predictResult = "这是一辆正常车"
    else:
        plateNo = "未识别"
        #carBrandZh = "未识别"
        predictResult = "车牌未识别，无法判定"

    return plateNo, v_type, v_color, carBrandZh, predictResult


# ------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
vehicle_info_database = prepare_service()


@app.route('/prepare')
def warm_up():
    vehicle_info_database = prepare_service()
    return redirect('/')


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def analyze():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        img_path = 'static/images/test.jpg'
        plate_no, v_type, v_color, car_brand, predict_result = predict(img_path, vehicle_info_database)
        context = ["车牌号：" + plate_no, "车型：" + v_type, "车辆颜色：" + v_color, "车辆品牌：" + car_brand,
                   "结论：" + predict_result]

        return render_template('index.html', context=context, val1=time.time())

    return render_template('index.html')

if __name__ == '__main__':
    # app.debug = True
    app.run(port=8090, debug=True)
