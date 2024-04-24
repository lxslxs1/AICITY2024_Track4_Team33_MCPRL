import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

delete_list=[]
def filter_detections(detections):
    filtered_detections = []

    # Group detections by camera and time segment
    camera_id_dic={}
    detections_grouped = {}
    for det in detections:
        image_id=det['image_id']
        camera_id = str(det['image_id'])[:2]
        time_segment = str(det['image_id'])[2:3]
        frame_seq = int(str(det['image_id'])[3:])
        key = (camera_id, time_segment)
        if key not in detections_grouped:
            detections_grouped[key] = {}
        if frame_seq not in detections_grouped[key]:
            detections_grouped[key][frame_seq]=[]
        detections_grouped[key][frame_seq].append(det)
        # detections_grouped[key].append((frame_seq, det))
    # for 
    # Sort detections within each group by frame sequence
    for (camera_time,frame_seq) in detections_grouped.items():
        print('*************',camera_time)
        frame_seq_list=[i for i in frame_seq.keys()]
        frame_seq_list=sorted(frame_seq_list)
        for k in range(len(frame_seq_list)):
            frame=frame_seq_list[k]
            ner_frame_list=[ frame_seq_list[k+j] for j in range(-15,20) if (k+j<len(frame_seq_list) and k+j>=0 and j!=0)]
            ner_frame_list=ner_frame_list[:20]
            for dets in frame_seq[frame]:
                is_duplicate = False
                match_counts = 0
                for ner_frame in ner_frame_list:
                    if is_duplicate:
                        break
                    boxes=frame_seq[ner_frame]
                    for temp in boxes:
                        if calculate_iou(dets['bbox'], temp['bbox'])>0.65:
                            match_counts += 1
                            if match_counts >3 and dets['score']<=0.3 :
                                is_duplicate = True
                                print(f'*****************已删除{dets}')
                                break
                            if match_counts >10 :
                                is_duplicate = True
                                print(f'*****************已删除{dets}')
                                break    
                if not is_duplicate :
                    filtered_detections.append(dets)
                else:delete_list.append(dets)
    return filtered_detections

# 示例文件路径
file_path = '/home/sy/lxs/AICITY/data/Fisheye8k/results/codetr/01234_0321_post_new_2_delete_dense_delete_dynamic3_2_1.json'

# 加载、过滤、保存
detections = load_json(file_path)
label1_detections=[]
other_detections=[]
for det in detections:
    if int(det['category_id']) == 3:
        label1_detections.append(det)
    else:
        other_detections.append(det)
filtered_detections = filter_detections(label1_detections)
filtered_detections =filtered_detections +other_detections
output_path=file_path.split('.json')[0]+"_delete_dynamic3.json"
output_path_dele=file_path.split('.json')[0]+"_only_dete.json"
with open(output_path, 'w') as f:
    print(f'***************长度为{len(filtered_detections)}')
    json.dump(filtered_detections, f, indent=4)
with open(output_path_dele, 'w') as f:
    print(f'***************长度为{len(delete_list)}')
    json.dump(delete_list, f, indent=4)

