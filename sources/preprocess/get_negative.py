import cv2
import json

# 加载预测结果和真实标注数据
with open("/home/sy/lxs/AICITY/Co-DETR/my_models/0320/val/valbbox.json", "r") as f:
    data = json.load(f)

with open("/home/sy/lxs/AICITY/data/Fisheye8k/val/format_val.json", "r") as f1:
    gt = json.load(f1)['annotations']

# 定义获取图像名称的函数
def get_image_name(image_id):
    img_name = f"{image_id}.png"
    return img_name

# 定义计算IoU的函数
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

# 定义过滤条件
def filter_objects(obj):
    return obj["score"] > 0.4

# 将真实数据中的bbox以图像ID为键存储
gt_bboxes = {g["image_id"]: [g["bbox"] for g in gt] if "bbox" in g else [] for g in gt}

# 过滤后的结果
filtered_objects = list(filter(filter_objects, data))

# 假设图像存储的基本路径
base_image_path = '/home/sy/lxs/AICITY/data/Fisheye8k/val/images/'
cropped_image_dir = "/home/sy/lxs/AICITY/data/Fisheye8k/val/cars/negative/"
i = 0
for obj in filtered_objects[:2000000]:
    i += 1
    if i % 50 == 0:
        img_name = get_image_name(obj["image_id"])
        image_path = base_image_path + img_name
        image = cv2.imread(image_path)
        bbox = [int(coord) for coord in obj["bbox"]]
        is_false_positive = True
        
        # 检查是否为 false positive
        for gt_bbox in gt_bboxes.get(str(obj["image_id"]), []):
            if compute_iou(bbox, gt_bbox) >= 0.5:
                is_false_positive = False
                break
        
        if is_false_positive:
            x, y, w, h = bbox
            cropped_image = image[y:y+h, x:x+w]
            cropped_image_path = cropped_image_dir + f"cropped_{obj['image_id']}_{obj['category_id']}_{x}_{y}_{w}.jpg"
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Saved cropped image to {cropped_image_path}")
