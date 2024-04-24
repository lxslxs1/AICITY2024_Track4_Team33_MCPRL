import cv2
import os
import random
import shutil
def crop_and_save_objects(img_path, label_path, output_dir,img_name):
    # 读取图像
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义类别和对应的文件夹名称
    class_folders = {2: 'car', 0: 'bus', 4: 'truck'}

    # 为每个类别创建输出文件夹
    for class_id in class_folders.values():
        class_dir = os.path.join(output_dir, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # 读取标注文件并裁剪出每个目标
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            i=0
            a=int(random.random()*3+1)
            for line in file.readlines():
                i=i+1
                if i%a==0:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())

                    # 计算边界框的像素坐标
                    x_min = int((x_center - bbox_width / 2) * width)
                    y_min = int((y_center - bbox_height / 2) * height)
                    x_max = int((x_center + bbox_width / 2) * width)
                    y_max = int((y_center + bbox_height / 2) * height)

                    # 裁剪目标
                    cropped_img = img[y_min:y_max, x_min:x_max]

                    # 根据类别保存裁剪的图像
                    class_folder = class_folders.get(class_id)
                    if class_folder:
                        # 构建保存路径
                        save_path = os.path.join(output_dir, class_folder, f'{img_name}_{int(class_id)}_{x_min}_{y_min}.jpg')
                        # 保存图像
                        cv2.imwrite(save_path, cropped_img)
                        print(f'Saved {save_path}')

# 示例用法
# img_path = '/home/sedlight/lxs/AICITY/yolov9-main/Fisheye8k/train/images/camera3_A_0.png'  # 图像路径
# label_path = '/home/sedlight/lxs/AICITY/yolov9-main/Fisheye8k/train/labels/camera3_A_0.txt'  # 标注文件路径
output_dir = '/home/sedlight/lxs/AICITY/yolov9-main/Fisheye8k/cars/train_all/neg_new'  # 输出目录
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
img_dir='/home/sedlight/lxs/AICITY/yolov9-main/Fisheye8k/val/images'
label_dir="/home/sedlight/lxs/AICITY/yolov9-main/runs/neg"
j=0 
for img in sorted(os.listdir(img_dir)):
    j=j+1
    b=int(random.random()*5+1)
    if j%b==0:
        if img.endswith('.jpg') or img.endswith('.png'):
            img_name=img.split(".")[0]
            img_path=os.path.join(img_dir,img)
            label_path=os.path.join(label_dir,img_name+".txt")
            crop_and_save_objects(img_path, label_path, output_dir,img_name)
    

