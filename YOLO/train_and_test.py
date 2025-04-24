import os
import random
import shutil

# 设置路径
image_dir = './data/images'
label_dir = './data/labels'

output_image_dir = './dataset/images'
output_label_dir = './dataset/labels'

train_ratio = 0.8  # 训练集比例，剩下的为验证集

# 获取所有图片文件名（不带扩展名）
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_basenames = [os.path.splitext(f)[0] for f in image_files]

# 打乱并划分
random.seed(42)
random.shuffle(image_basenames)
split_index = int(len(image_basenames) * train_ratio)
train_basenames = image_basenames[:split_index]
val_basenames = image_basenames[split_index:]

# 创建输出目录
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_label_dir, split), exist_ok=True)

# 拷贝文件函数
def copy_files(basenames, split):
    for name in basenames:
        # 复制图像
        for ext in ['.jpg', '.png', '.jpeg']:
            src_img = os.path.join(image_dir, name + ext)
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(output_image_dir, split, name + ext))
                break
        # 复制标签
        src_label = os.path.join(label_dir, name + '.txt')
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(output_label_dir, split, name + '.txt'))

# 执行复制
copy_files(train_basenames, 'train')
copy_files(val_basenames, 'val')

print("✅ 数据划分完成，训练集数量：{}, 验证集数量：{}".format(len(train_basenames), len(val_basenames)))
