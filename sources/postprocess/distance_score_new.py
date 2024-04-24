import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
import os
def determine_region(distance, num_distance_bins=3):
    """
    Determine the region index for a given normalized distance.
    
    Parameters:
    distance (float): Normalized distance of the detection from the image center.
    num_distance_bins (int): The number of distance bins or regions.
    
    Returns:
    int: The region index (0, 1, or 2) that the distance falls into.
    """
    gap = 1.0 / num_distance_bins
    region = int(distance // gap)
    return region

def get_image_id(img_name):
    img_name = img_name.split('.')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIdx = int(img_name.split('_')[0].split('camera')[1])
    sceneIdx = sceneList.index(img_name.split('_')[1])
    frameIdx = int(img_name.split('_')[2])
    imageId = int(str(cameraIdx)+str(sceneIdx)+str(frameIdx))
    return imageId

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_histogram(scores, bins=10):
    hist, _ = np.histogram(scores, bins=bins, range=(0, 1))
    return hist

def match_histograms_old(src_scores, target_hist, src_hist):
    # 简化的直方图匹配方法，真实使用时可能需要更复杂的逻辑
    # 这里使用OpenCV的normalize方法作为直方图匹配的一个简单代理
    src_cdf = np.cumsum(src_hist).astype(np.float32)
    src_cdf = src_cdf / src_cdf[-1]  # 归一化
    target_cdf = np.cumsum(target_hist).astype(np.float32)
    target_cdf = target_cdf / target_cdf[-1]  # 归一化

    # 创建一个查找表
    lookup_table = np.searchsorted(target_cdf, src_cdf)
    min_score, max_score = 0, 1
    scale = (max_score - min_score) / (len(lookup_table) - 1)
    adjusted_scores = np.interp(src_scores, np.arange(len(lookup_table)) * scale + min_score, lookup_table * scale + min_score)

    return adjusted_scores
def match_histograms(src_scores, target_hist_counts, src_hist_counts):
    # 将频数转换为频率
    src_hist_freq = src_hist_counts / np.sum(src_hist_counts)
    target_hist_freq = target_hist_counts / np.sum(target_hist_counts)
    
    # 计算累积分布函数 (CDF)
    src_cdf = np.cumsum(src_hist_freq).astype(np.float32)
    target_cdf = np.cumsum(target_hist_freq).astype(np.float32)
    
    # 创建查找表
    lookup_table = np.searchsorted(target_cdf, src_cdf)
    
    # 插值得分
    min_score, max_score = 0, 1
    scale = (max_score - min_score) / (len(lookup_table) - 1)
    adjusted_scores = np.interp(src_scores, np.arange(len(lookup_table)) * scale + min_score, lookup_table * scale + min_score)
    
    # 确保调整后的得分不会超出[0, 1]的范围
    adjusted_scores = np.clip(adjusted_scores, min_score, max_score)

    return adjusted_scores


# 示例：加载图像尺寸数据
image_size = {}
# 示例图像目录和JSON文件路径，请替换为实际路径
image_dir = '/home/sy/lxs/1080/AICITY/yolov9-main/Fisheye8k/test/images'
json_file_path = '/home/sy/lxs/AICITY/InternImage/detection/mmdetection-2.28.1/tools/analysis_tools/fuse_result/codetr_0320_fusion_results.json'

# 加载图像尺寸数据
for image in tqdm(os.listdir(image_dir), desc='Loading image sizes'):
    w, h = get_image_size(os.path.join(image_dir, image))
    image_id = get_image_id(image)
    image_size[image_id] = (w, h)

# 加载检测结果
detections = load_json(json_file_path)

# 计算检测结果的归一化距离并按区域分组
detections_by_region = {0: [], 1: [], 2: []}
for detection in detections:
    image_id = detection['image_id']
    bbox = detection['bbox']
    image_w, image_h = image_size[image_id]
    center_x = bbox[0] + bbox[2] / 2.0
    center_y = bbox[1] + bbox[3] / 2.0
    distance_max = np.sqrt((image_w / 2) ** 2 + (image_h / 2) ** 2)
    distance = np.sqrt((center_x - image_w / 2) ** 2 + (center_y - image_h / 2) ** 2) / distance_max
    region = determine_region(distance, num_distance_bins=3)
    detections_by_region[region].append(detection)

# # 收集每个区域的得分
scores_by_region = {region: [d['score'] for d in detections] for region, detections in detections_by_region.items()}

# # 计算得分的直方图
histograms = {region: calculate_histogram(scores, bins=200) for region, scores in scores_by_region.items()}

# 选择目标直方图并匹配
target_histogram = histograms[1]
adjusted_scores_by_region = {}
for region, scores in scores_by_region.items():
    if region == 1:  # 第二区域不调整
        adjusted_scores_by_region[region] = scores
    else:
        src_hist = histograms[region]
        adjusted_scores_by_region[region] = match_histograms(np.array(scores), target_histogram, src_hist).tolist()


# 定义得分区间和距离区间
num_score_bins = 10
score_bins = np.linspace(0, 1, num_score_bins+1 )
num_distance_bins = 3
distance_bins = np.linspace(0, 1, num_distance_bins + 1)

# 初始化计数数组
counts_before = np.zeros((num_distance_bins, num_score_bins), dtype=int)
counts_after = np.zeros((num_distance_bins, num_score_bins), dtype=int)

# 填充计数数组 - 调整前
for region, scores in scores_by_region.items():
    score_indices = np.digitize(scores, score_bins) - 1
    for indice in score_indices:
        counts_before[region, indice] += 1

# 填充计数数组 - 调整后
for region, scores in adjusted_scores_by_region.items():
    score_indices = np.digitize(scores, score_bins) - 1
    for indice in score_indices:
        counts_after[region, indice] += 1
adjusted_scores_by_region[1] = np.array(scores_by_region[1])
# 转换计数为频率
counts_frequency_before = counts_before / counts_before.sum(axis=1, keepdims=True)
counts_frequency_after = counts_after / counts_after.sum(axis=1, keepdims=True)


plt.style.use('seaborn-paper')

# 计算频率的代码在这里省略，请使用上面优化后的代码进行计算

# 创建图和子图
fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
ax1, ax2 = axes
bar_width = 0.35
index = np.arange(num_distance_bins)
# fig, ax = plt.subplots(figsize=(10, 8))

# 设置子图1：调整前
for i in range(num_score_bins):
    ax1.bar(index, counts_frequency_before[:, i], bar_width,
            label=f'{score_bins[i]:.2f}-{score_bins[i+1]:.2f}',
            bottom=np.sum(counts_frequency_before[:, :i], axis=1),
            color=plt.cm.rainbow(i / num_score_bins))

ax1.set_title('(a) Before Adjustment',y=-0.15)
ax1.set_xlabel('Distance Intervals')
ax1.set_ylabel('Frequency')
ax1.set_xticks(index)
ax1.set_xticklabels([f'{distance_bins[i]:.2f}-{distance_bins[i+1]:.2f}' for i in range(num_distance_bins)])

# 设置子图2：调整后
for i in range(num_score_bins):
    ax2.bar(index, counts_frequency_after[:, i], bar_width,
            label=f'{score_bins[i]:.2f}-{score_bins[i+1]:.2f}',
            bottom=np.sum(counts_frequency_after[:, :i], axis=1),
            color=plt.cm.rainbow(i / num_score_bins))

ax2.set_title('(b) After Adjustment', y=-0.15)
ax2.set_xlabel('Distance Intervals')
# ax2.set_ylabel('Frequency') # Y轴标签在共享，所以不需要设置第二次
ax2.set_xticks(index)
ax2.set_xticklabels([f'{distance_bins[i]:.2f}-{distance_bins[i+1]:.2f}' for i in range(num_distance_bins)])

# 添加图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, title='Score Intervals')

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.9])  # 为顶部的图例留出空间

# 显示图形
# plt.show()
plt.savefig('/home/sy/lxs/AICITY/data/Fisheye8k/codetr_0331_intern_distance_vs_score.png')


