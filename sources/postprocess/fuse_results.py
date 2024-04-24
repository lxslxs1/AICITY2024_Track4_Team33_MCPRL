import argparse

from mmengine.fileio import dump, load
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tqdm
from PIL import Image
import os
# import sys
# sys.path.append('/home/sy/lxs/AICITY/InternImage/detection/mmdetection-2.28.1')
from wbf import weighted_boxes_fusion
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
        width, height = img.size
        return width, height
image_size={}
for image in tqdm.tqdm(os.listdir('/home/sy/lxs/1080/AICITY/yolov9-main/Fisheye8k/test/images')):
    w,h=get_image_size(os.path.join('/home/sy/lxs/1080/AICITY/yolov9-main/Fisheye8k/test/images',image))
    image_id=get_image_id(image)
    image_size[image_id]=[w,h]

def parse_args():
    parser = argparse.ArgumentParser(description='Fusion image \
        prediction results using Weighted \
        Boxes Fusion from multiple models.')
    parser.add_argument(
        '--pred-results',
        type=str,
        default=['/home/sy/lxs/AICITY/InternImage/detection/myworkdirs/fisheye_raw/intern_new_changeid.json', '/home/sy/lxs/AICITY/InternImage/detection/myworkdirs/fisheye_raw/intern_rotate_changeid.json' ],
        nargs='+')
        # help='files of prediction results \
        #             from multiple models, json format.')
    parser.add_argument('--annotation', type=str, default='/home/sy/lxs/AICITY/data/Fisheye8k/test/format_test.json',help='annotation file path')
    parser.add_argument(
        '--weights',
        type=float,
        nargs='*',
        default=None,
        help='weights for each model, '
        'remember to correspond to the above prediction path.')
    parser.add_argument(
        '--fusion-iou-thr',
        type=float,
        default=0.65,
        help='IoU value for boxes to be a match in wbf.')
    parser.add_argument(
        '--skip-box-thr',
        type=float,
        default=0.2,
        help='exclude boxes with score lower than this variable in wbf.')
    parser.add_argument(
        '--conf-type',
        type=str,
        default='avg',
        help='how to calculate confidence in weighted boxes in wbf.')
    parser.add_argument(
        '--eval-single',
        action='store_true',
        help='whether evaluate each single model result.')
    parser.add_argument(
        '--save-fusion-results',
        default=True)
        # action='store_true',
        # help='whether save fusion result')
    parser.add_argument(
        '--only_file1_small',
        default=False)
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/home/sy/lxs/AICITY/InternImage/detection/mmdetection-2.28.1/tools/analysis_tools/fuse_result/',
        help='Output directory of images or prediction results.')

    args = parser.parse_args()

    return args

def convert_xywh_nxyxy(box,w,h):
    w,h=float(w),float(h)
    new=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
    res=[new[0]/w,new[1]/h,new[2]/w,new[3]/h]
    return res
def convert_nxyxy_xywh(box,w,h):
    w,h=float(w),float(h)
    new=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
    res=[new[0]*w,new[1]*h,new[2]*w,new[3]*h]
    return res
def main():
    image_size={}
    for image in tqdm.tqdm(os.listdir('/home/sy/lxs/1080/AICITY/yolov9-main/Fisheye8k/test/images')):
        w,h=get_image_size(os.path.join('/home/sy/lxs/1080/AICITY/yolov9-main/Fisheye8k/test/images',image))
        image_id=get_image_id(image)
        image_size[image_id]=[w,h]
    args = parse_args()

    # assert len(args.models_name) == len(args.pred_results), \
    #     'the quantities of model names and prediction results are not equal'

    cocoGT = COCO(args.annotation)

    predicts_raw = []

    models_name = ['model_' + str(i) for i in range(len(args.pred_results))]

    for model_name, path in \
            zip(models_name, args.pred_results):
        pred = load(path)
        
        if args.only_file1_small:
            if model_name=='model_0':
                for det in pred:
                    box=det['bbox']
                    if box[2]*box[3]>64*64:
                        continue
                    else:
                        predicts_raw.append(det)
            else:
                predicts_raw.append(pred)
        else:
            predicts_raw.append(pred)

        if args.eval_single:
            print_log(f'Evaluate {model_name}...')
            cocoDt = cocoGT.loadRes(pred)
            coco_eval = COCOeval(cocoGT, cocoDt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    predict = {
        str(image_id): {
            'bboxes_list': [[] for _ in range(len(predicts_raw))],
            'scores_list': [[] for _ in range(len(predicts_raw))],
            'labels_list': [[] for _ in range(len(predicts_raw))]
        }
        for image_id in cocoGT.getImgIds()
    }

    for i, pred_single in enumerate(predicts_raw):
        for pred in pred_single:
            p = predict[str(pred['image_id'])]
            w,h=image_size[pred['image_id']][0],image_size[pred['image_id']][1]
            pred['bbox']=convert_xywh_nxyxy(pred['bbox'],w,h)
            p['bboxes_list'][i].append(pred['bbox'])
            p['scores_list'][i].append(pred['score'])
            p['labels_list'][i].append(pred['category_id'])

    result = []
    prog_bar = ProgressBar(len(predict))
    for image_id, res in predict.items():
        w,h=image_size[int(image_id)][0],image_size[int(image_id)][1]
        bboxes, scores, labels = weighted_boxes_fusion(
            res['bboxes_list'],
            res['scores_list'],
            res['labels_list'],
            weights=args.weights,
            iou_thr=args.fusion_iou_thr,
            skip_box_thr=args.skip_box_thr,
            conf_type=args.conf_type)

        for bbox, score, label in zip(bboxes, scores, labels):
            bbox=bbox.numpy().tolist()
            bbox=convert_nxyxy_xywh(bbox,w,h)
            result.append({
                'bbox': bbox,
                'category_id': int(label),
                'image_id': int(image_id),
                'score': float(score)
            })

        prog_bar.update()

    if args.save_fusion_results:
        out_file = args.out_dir + '/fusion_results.json'
        dump(result, file=out_file,indent=4)
        print_log(
            f'Fusion results have been saved to {out_file}.', logger='current')

    print_log('Evaluate fusion results using wbf...')
    cocoDt = cocoGT.loadRes(result)
    # coco_eval = COCOeval(cocoGT, cocoDt, iouType='bbox')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()


if __name__ == '__main__':
    main()