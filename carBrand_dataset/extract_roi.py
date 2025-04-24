annotations_path = './data/annotations' # 标注文件的目录
vehicle_path = './data/img' # 原始图像的目录
extracted_path = './data/extracted' # 截取后存储图像的目录


# 读取xml，得到坐标列表    
from xml.dom import minidom
def get_object_location(xml_path):
    document_tree = minidom.parse(xml_path)
    bndbox = document_tree.getElementsByTagName("bndbox")
    location_list = []
    for box in bndbox:
        location = []
        xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
        location.extend([xmin, ymin, xmax, ymax])
        location_list.append(location)
        
    return location_list


# 根据坐标列表截取图像
import cv2
def get_image_roi(image_path, xmin, ymin, xmax, ymax):
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    image = cv2.imread(image_path)
    roi = image[ymin:ymax, xmin:xmax]
    return roi
    
    
# 遍历车辆目录
import os
for _, _, file_list in os.walk(vehicle_path):
    pass


image_counter = 0
progress_counter = 0
for each_file in file_list:
    progress_counter += 1
    if progress_counter % 50 == 0:
        print('正在处理第%d/%d张图像......' %(progress_counter, len(file_list)))
    each_vehicle_path = vehicle_path + '/' + each_file
    each_annotation_path = annotations_path + '/' +  each_file.split('.')[0] + '.xml'
    
    # 标注文件可能不存在
    if not os.path.exists(each_annotation_path):
        continue
    
    location_list = get_object_location(each_annotation_path)
    
    if not os.path.exists(extracted_path):
       os.makedirs(extracted_path)
    for each_location in location_list:
        image_counter += 1
        xmin,ymin,xmax,ymax = each_location[0],each_location[1],each_location[2],each_location[3]
        roi = get_image_roi(each_vehicle_path, xmin, ymin, xmax, ymax)
        cv2.imwrite(extracted_path+'/'+str(image_counter)+'.jpg', roi)
