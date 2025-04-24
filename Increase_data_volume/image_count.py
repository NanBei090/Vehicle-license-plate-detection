import os

root_dirs = {
    "车辆颜色数据": r"G:\Projects\car-system\vehicle_color_data\vehicle_color",
    "车辆类型数据": r"G:\Projects\car-system\vehicle_type_data\images"
}

# 支持的图片后缀
image_extensions = ['.jpg', '.png', '.jpeg']

print("各子文件夹图片数量统计：\n")

# 遍历每一个目录
for dataset_name, root_dir in root_dirs.items():
    print(f"【{dataset_name}】")

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            count = 0
            for filename in os.listdir(subfolder_path):
                if os.path.splitext(filename)[-1].lower() in image_extensions:
                    count += 1
            print(f"{subfolder} 文件夹中共有图片 {count} 张")

    print("-" * 40)
