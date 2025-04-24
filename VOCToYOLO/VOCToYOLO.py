import os
import xml.etree.ElementTree as ET

# 输入和输出路径
xml_dir = "./data/annotations"       # VOC 标签文件夹
img_dir = "./data/images"            # 原始图像路径（获取宽高用）
output_dir = "./data/labels"         # 输出 YOLO 格式标签

# 收集所有类别
class_set = set()
for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    for obj in root.iter("object"):
        class_set.add(obj.find("name").text)

classes = sorted(list(class_set))
print("所有类别：", classes)

# 保存车辆classes
with open("classes.txt", "w", encoding="utf-8") as f:
    for cls in classes:
        f.write(f"{cls}\n")

os.makedirs(output_dir, exist_ok=True)

def convert_box(size, box):
    """转换边界框坐标 (x_min, x_max, y_min, y_max) ➜ YOLO 格式"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    # 图像尺寸
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    # 输出 TXT 标签文件名
    txt_filename = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
    with open(txt_filename, "w") as f:
        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)

            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text)
            )
            bb = convert_box((w, h), b)
            f.write(f"{cls_id} {' '.join([str(round(a, 6)) for a in bb])}\n")

print("VOC to YOLO 格式转换完成！")
