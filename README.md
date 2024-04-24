# AICITY2024_Track4_Team33_MCPRL
The solutions ranked sixth in Track 4 (Road Object Detection in Fish-Eye Cameras) of the NVIDIA [AI City Challenge](https://www.aicitychallenge.org/) at the CVPR 2024 Workshop.

## Installation
Here is the list of libraries used in this project:

- [Co-DETR](https://github.com/Sense-X/Co-DETR) 
- [InternImage](https://github.com/OpenGVLab/InternImage?tab=readme-ov-file)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

## Solution pipelines
1. Download the training datasets from [fisheye8k](https://scidm.nchc.org.tw/en/dataset/fisheye8k). 

2. Data process
- detection

  a. crop and rotate images 
  `bash sources/preprocess/crop.sh`
 
  b. convert gt.json to yolo.txt, codetr.json and InternImage.json
  `bash sources/preprocess/convet.sh`
- classify vehicles

  a.get positive samples to train the network for classifying vehicles. 
 `python sources/preprocess/crop_cars.py`
 
  b.get negative samples to train the network for classifying vehicles. 
 `python sources/preprocess/get_negative.py`

3. Detection Models
- Co-DETR

  Use [Co-DETR](https://github.com/Sense-X/Co-DETR) to train day and night scenes. Choose [Objects365 pre-trained Co-DETR](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv) as the initial weight.

Configs: 
```
sources/models_config/codetr_day.py
sources/models_config/codetr_night.py
```
- InternImage

  Use [InternImage](https://github.com/OpenGVLab/InternImage?tab=readme-ov-file)  to train images. Choose [cascade_internimage_xl_fpn_3x_coco.pth](https://cdn-lfs.huggingface.co/repos/29/b8/29b884d43d991fb1da1715a1ff9ec2e0f0c0bee808c6c6988adcf442954ffdf5/9214c6c9af906d1fc5c8f33f48911be204aa231039191eb89a7bf0579ce57003?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27cascade_internimage_xl_fpn_3x_coco.pth%3B+filename%3D%22cascade_internimage_xl_fpn_3x_coco.pth%22%3B&Expires=1713939383&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzkzOTM4M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy8yOS9iOC8yOWI4ODRkNDNkOTkxZmIxZGExNzE1YTFmZjllYzJlMGYwYzBiZWU4MDhjNmM2OTg4YWRjZjQ0Mjk1NGZmZGY1LzkyMTRjNmM5YWY5MDZkMWZjNWM4ZjMzZjQ4OTExYmUyMDRhYTIzMTAzOTE5MWViODlhN2JmMDU3OWNlNTcwMDM%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=x8XWDQ2h6Vf5nE0OAA3Gd1xXZ0UpoCIK0s6tY7DjSnpsjFYTf-MkWTYkzECNCjxkQZWfdLOS1hj6%7E-PXNe9xgm3DgxBlQ3hvFmSgSOEmzDhCvR-mgJU3pZDs2RqWg1cYgqSJkKeNUeY0%7EZL9o3WDfpsj5s%7EoKPyq%7E8Qht90rpAi0nY3HVznizR5DCUs%7EHvK3-XzYa9NyNtI-bJrmCizi3Sf12ikDXm1lp73irPm4h3aN5ZK2HjtbR0SZLT9pnGmFx-f1uPyGFQT-WuKCnz6I%7E%7E0gA7PP%7EVZnJA7h9M8H3WSOCjM6EEwBlc%7E-s2nLAr1BZumuOZcwQDGjQcay9nmiow__&Key-Pair-Id=KVTP0A1DKRTAX) as the initial weight. 

Config:
`sources/models_config/internimage.py`

- YOLOv8



  Use [YOLOv8](https://github.com/ultralytics/ultralytics) to train images. Choose [yolov8x](https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/03d042aa-d6cf-4ac8-a3ac-778b55d9ee73?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240421T062828Z&X-Amz-Expires=300&X-Amz-Signature=57ff0fd61dac3466004dd618e3dc80191feab8254901c9cc51bba7e642a775a6&X-Amz-SignedHeaders=host&actor_id=59115350&key_id=0&repo_id=521807533&response-content-disposition=attachment%3B%20filename%3Dyolov8x.pt&response-content-type=application%2Foctet-stream) as the initial weight.

Configs: 
```
sources/models_config/fisheye.yaml
sources/models_config/yolov8x.yaml
```
4. Post_Process

- Vehicles Classifier Module

  Use [YOLOv8](https://github.com/ultralytics/ultralytics) to train vehicles classification model. Choose [YOLOv8s-cls](https://objects.githubusercontent.com/github-production-release-asset-2e65be/521807533/5e2908c1-83a7-498e-9b97-13e03b151148?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240421T085016Z&X-Amz-Expires=300&X-Amz-Signature=1218a6b9664f1ac2928c3b50023080e88cca7e226dc7992f9af18e75fd556aa2&X-Amz-SignedHeaders=host&actor_id=59115350&key_id=0&repo_id=521807533&response-content-disposition=attachment%3B%20filename%3Dyolov8s-cls.pt&response-content-type=application%2Foctet-stream) as the initial weight.

Config: `sources/models_config/YOLOv8s-cls.yaml`

- Static Objects Processing

`bash sources/postprocess/new_post.py`

- Confidence Score Refine

`bash sources/postprocess/distance_score_new.py`

- WBF

  Put `fuse_results.py` into `mmdetection-2.28.1/tools/analysis_tools/`  and then
`bash sources/postprocess/fuse.sh`
