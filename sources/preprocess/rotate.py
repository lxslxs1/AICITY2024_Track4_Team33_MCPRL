import os
from PIL import Image
import tqdm
def read_yolo_labels(label_path, image_width, image_height):
    bboxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height
            bboxes.append((class_id,x_min, y_min, x_max, y_max))
    return bboxes

def convert_bbox_to_yolo_format(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return [x_center, y_center, width, height]

def process_images_and_labels(input_dir,input_txt_dir, output_dir):
    for filename in tqdm.tqdm(os.listdir(input_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image_name=filename.split(".")[0]
            label_path = os.path.join(input_txt_dir,image_name+".txt")
            
            image = Image.open(image_path)
            image_width, image_height = image.size
            bboxes = read_yolo_labels(label_path, image_width, image_height)
            
            for angle in [180]:
            # for angle in [90, 180, 270]:#我将验证集的转换之后的图像一起训练
                rotated_image = image.rotate(angle, expand=True)
                output_imgs_dir= output_dir+"/images"
                output_labels_dir= output_dir+"/labels"
                if os.path.exists(output_imgs_dir)==0:
                    os.mkdir(output_imgs_dir)
                if os.path.exists(output_labels_dir)==0:
                    os.mkdir(output_labels_dir)
                rotated_image_path = os.path.join(output_imgs_dir, f"{os.path.splitext(filename)[0]}|{angle}.png")
                rotated_image.save(rotated_image_path)
                
                rotated_width, rotated_height = rotated_image.size
                rotated_bboxes = update_bboxes_and_convert_to_yolo(bboxes, (image_width, image_height), rotated_image.size, angle)
                yolo_label_path = os.path.join(output_labels_dir, f"{image_name}|{angle}.txt")
                with open(yolo_label_path, 'w') as f:
                    for bbox,yolo_bbox in zip(bboxes,rotated_bboxes):
                        f.write(str(int(bbox[0]))+' '+' '.join(map(str, yolo_bbox)) + '\n')

def update_bboxes_and_convert_to_yolo(bboxes, original_size, rotated_size, angle):
    original_width, original_height = original_size
    rotated_width, rotated_height = rotated_size
    yolo_bboxes = []

    for bbox in bboxes:
        bbox=bbox[1:]
        if angle == 0:
            new_bbox =bbox
        elif angle == 90:
            new_bbox = (bbox[1], original_width - bbox[2], bbox[3], original_width - bbox[0])
        elif angle == 180:
            new_bbox = (original_width - bbox[2], original_height - bbox[3], original_width - bbox[0], original_height - bbox[1])
        elif angle == 270:
            new_bbox = (original_height - bbox[3], bbox[0], original_height - bbox[1], bbox[2])

        yolo_bbox = convert_bbox_to_yolo_format(new_bbox, rotated_width, rotated_height)
        yolo_bboxes.append(yolo_bbox)

    return yolo_bboxes

# 示例用法
input_dir = '/home/sy/lxs/AICITY/data/Fisheye8k/split_val/images' # 输入文件夹路径
input_txt_dir="/home/sy/lxs/AICITY/data/Fisheye8k/split_val/labels"
output_dir = '/home/sy/lxs/AICITY/data/Fisheye8k/split_val' # 输出文件夹路径
process_images_and_labels(input_dir,input_txt_dir, output_dir)
