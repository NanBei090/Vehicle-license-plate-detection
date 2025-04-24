import cv2
import numpy as np
import os

def process_train_set(data_dir, save_dir):
    filelist = os.listdir(data_dir)
    num = 0
    for item in filelist:
        if item.endswith('.jpg') or item.endswith('.png'):
            img_full_path = os.path.join(os.path.abspath(data_dir), item)


            # 加载图片一次
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"[跳过] 无法读取图像：{img_full_path}")
                continue

            # 增强操作
            FlipOperation(img, str(num), save_dir)
            Shift(img, str(num), save_dir)
            Rotation(img, str(num), save_dir)

            num += 1
            print(f"The {num} th")
    print("Image augment finished!")

def FlipOperation(img, img_name, save_dir):
    middle_name = 'Flip_'
    os.makedirs(save_dir, exist_ok=True)

    try:
        flipped1 = cv2.flip(img, 1)
        save_name1 = os.path.join(save_dir, f"{middle_name}Hor_{img_name}.jpg")
        cv2.imwrite(save_name1, flipped1)

        flipped2 = cv2.flip(img, 0)
        save_name2 = os.path.join(save_dir, f"{middle_name}Vec_{img_name}.jpg")
        cv2.imwrite(save_name2, flipped2)
    except Exception as e:
        print(f"[错误] FlipOperation 失败: {e}")

def Shift(img, img_name, save_dir):
    middle_name = 'Shift_'
    os.makedirs(save_dir, exist_ok=True)

    try:
        rows, cols = img.shape[:2]
        M = np.array([[1, 0, -100], [0, 1, -12]], dtype=np.float32)
        dst = cv2.warpAffine(img, M, (cols, rows))
        save_name = os.path.join(save_dir, f"{middle_name}{img_name}.jpg")
        cv2.imwrite(save_name, dst)
    except Exception as e:
        print(f"[错误] Shift 失败: {e}")

def Rotation(img, img_name, save_dir):
    middle_name = 'Rotation_'
    os.makedirs(save_dir, exist_ok=True)

    try:
        rows, cols = img.shape[:2]
        R_image = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        dst = cv2.warpAffine(img, R_image, (cols, rows))
        save_name = os.path.join(save_dir, f"{middle_name}{img_name}.jpg")
        cv2.imwrite(save_name, dst)
    except Exception as e:
        print(f"[错误] Rotation 失败: {e}")


data_dir = "./data/yellow"
save_dir = "./data/yellow_augment/"
process_train_set(data_dir, save_dir)
