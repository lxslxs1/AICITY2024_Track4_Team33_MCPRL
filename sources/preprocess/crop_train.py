# -*-coding:utf-8-*-
import os
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A
import tqdm
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2,height=640,width=640):
    """Visualizes a single bounding box on the image"""
    bbox1=bbox[:4]
    # x_min, y_min, w, h =bbox1
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    bbox_xyxy=xywhn2xyxy(bbox1,width,height)
    # print(bbox_xyxy)
    x_min, y_min, x_max,y_max =int(bbox_xyxy[0]),int(bbox_xyxy[1]),int(bbox_xyxy[2]),int(bbox_xyxy[3])
    # print(x_min, y_min,  x_max,y_max )

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    height=img.shape[0]
    width=img.shape[1]
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name,height=height,width=width)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


path = r"/home/sy/lxs/AICITY/data/Fisheye8k/val/images"
path_labels=r"/home/sy/lxs/AICITY/data/Fisheye8k/val/labels"

# path =r"/disk2/lxs/yolov7/test_Albumentations/images"
# path_labels=r"/disk2/lxs/yolov7/test_Albumentations/labels"
listdir = os.listdir(path)

pic_num=0

for i in tqdm.tqdm(listdir):
    if i.split('.')[1] == "png" or i.split('.')[1] == "JPG" or i.split('.')[1] == "jpg" :
        print(i)
        filepath = os.path.join(path, i)
        filename = i.split('.')[0]
        filename_labels=os.path.join(path_labels,'{}.txt'.format(filename)) #大图的标签路径

        # img = cv2.imread(filepath) #读取大图
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # filename_labels="/disk2/lxs/yolov7/test_Albumentations/labels/9999999_00885_d_0000406.txt"
    num = len(open(filename_labels,'rU').readlines())
    with open(filename_labels, "r") as f:
        bboxes= [[] for i in range(num)]
        category_ids=[]
        count=0
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            type=int(line.split(' ')[0]) #每一个bbox的类型
            for i in line.split(" ")[1:5]:bboxes[count].append(float(i))
            bboxes[count].append(int(line.split(" ")[0]))
            category_ids.append(int(line.split(" ")[0]))
            count=count+1

    # category_id_to_name = {0:'pedestrian', 1:'people', 2:'bicycle', 3:'car', 4:'van', 5:'truck', 6:'tricycle', 7:'awning-tricycle', 8:'bus',9: 'motor'}
    # visualize(image, bboxes, category_ids, category_id_to_name)

    [h,w] = image.shape[:2] #大图的h,w
    dw,dh=int(w*0.125),int(h*0.125)
    for pic in range(4):
        cor=[]
        
        if pic==0:
            cor=[0,0,w//2+dw,h//2+dh]
            transform = A.Compose(
                [A.Crop(x_min=0, y_min=0, x_max=int(w//2+dw), y_max=int(h//2+dh))],
                bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids'],min_visibility=0.3),
            )

        if pic==1:
            cor=[w//2-dw, 0, w, h//2+dh]
            transform = A.Compose(
                [A.Crop(x_min=w//2-dw, y_min=0, x_max=w, y_max=h//2+dh)],
                bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids'],min_visibility=0.3),
            )

        if pic==2:
            cor=[0, h//2-dh, w//2+dw, h]
            transform = A.Compose(
                [A.Crop(x_min=0, y_min=h//2-dh, x_max=w//2+dw, y_max=h)],
                bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids'],min_visibility=0.3),
            )

        if pic==3:
            cor=[w//2-dw, h//2-dh, w,h]
            transform = A.Compose(
                [A.Crop(x_min=w//2-dw, y_min=h//2-dh, x_max=w, y_max=h)],
                bbox_params=A.BboxParams(format='yolo',label_fields=['category_ids'],min_visibility=0.3),
            )

        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        # visualize(
        #     transformed['image'],
        #     transformed['bboxes'],
        #     transformed['category_ids'],
        #     category_id_to_name
        # )
        newdir='/home/sy/lxs/AICITY/data/Fisheye8k/split_val/images'
        newdir_labels='/home/sy/lxs/AICITY/data/Fisheye8k/split_val/labels'
            # 新建split文件夹用于保存
        # newdir = os.path.join(path, 'images_split')
        if (os.path.exists(newdir) == False):
            os.mkdir(newdir)
        
        # newdir_labels = os.path.join(path_labels, 'labels_split')
        if (os.path.exists(newdir_labels) == False):
            os.mkdir(newdir_labels)




        path_new_labels= os.path.join(newdir_labels, filename) + "|{}_{}.txt".format(pic,"_".join((str(j) for j in cor)))
        f = open(path_new_labels,'a')
        category_ids_length=len(transformed['category_ids'])
        for k in range(category_ids_length):
            for m in [4,0,1,2,3]:
                f.write(str(transformed['bboxes'][k][m]))
                f.write(' ')
            f.write('\n') # 实现换行的功能
        f.close()

        path_new= os.path.join(newdir, filename) + "|{}_{}.png".format(pic,"_".join((str(j) for j in cor)))
        transformed['image'] = cv2.cvtColor(transformed['image'], cv2.COLOR_BGR2RGB)
        cv2.imwrite(path_new, transformed['image'])

    pic_num=pic_num+1

print("all_pic:",pic_num)