# Ego-Exo4D RESNET Model 
Implementation of hand-ego-pose-potter, a 3D hand pose estimation baseline model based on RESNET.


## Data preparation
Follow instructions [here](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation) to get:
- ground truth annotation files in `$gt_output_dir/annotation/manual` or `$gt_output_dir/annotation/auto` if using automatic annotations,
referred as `gt_anno_dir` below
- corresponding undistorted Aria images in `$gt_output_dir/image/undistorted`, 
referred as `aria_img_dir` below

## Setup

- Follow the instructions below to set up the environment for model training and inference.
```
conda create -n resnet_hand_pose python=3.9.16 -y
conda activate resnet_hand_pose
pip install -r requirement.txt
```
- Install [pytorch](https://pytorch.org/get-started/previous-versions/). The model is tested with `pytorch==2.1.0` and `torchvision==0.16.0`. 

## Navigate to the RESNET model directory
```
cd models/RESNET
```
## Training

Run command below to perform training on manual data with pretrained POTTER_cls weight:
```
python3 train.py \
    --gt_anno_dir <gt_anno_dir> \
    --aria_img_dir <aria_img_dir>

Example:
python3 train.py --gt_anno_dir /workspace/project/ego-exo_gt_data/annotation/manual \
                 --aria_img_dir /workspace/project/ego-exo_gt_data/image/undistorted

```

## Inference

Run command below to perform inference. It will be stored at `output/inference_output` by default. 
```
python3 inference.py --gt_anno_dir <gt_anno_dir> \
                     --aria_img_dir <aria_img_dir> \
                     --output_dir <inference output> \
                     --pretrained_ckpt <ckpt> \
                     -- split <train/val/test> 

Example: 
python3 inference.py --gt_anno_dir /workspace/project/ego-exo_gt_data/annotation/manual \
                     --aria_img_dir /workspace/project/ego-exo_gt_data/image/undistorted \
                     --output_dir /workspace/project/ego-exo4d-egopose/handpose/models/RESNET/output/report_inference_outputs/exp1a \
                     --pretrained_ckpt /workspace/project/ego-exo4d-egopose/handpose/models/RESNET/output/ego4d/2024-06-05-16-59/RESNET-HandPose-ego4d.pt \
                     --split val
```

The output format is: 
```
{
    "<take_uid>": {
        "<frame_number>": {
                "left_hand_3d": [],
                "right_hand_3d": []     
        }
        ...
    }
    ...
}
```
## Evaluation

Navigate to the evaluation directory
```
cd ../evaluation
```

Run command below to get evaluation metrics
```
python3 evaluate.py --gt_path <gt_anno_file_path> \
                    --offset <because predicted outputs are offset by wrist pos>
                    --pred_path <pred_anno_file_path>
Example: 
python3 evaluate.py --gt_path /workspace/project/ego-exo_gt_data/annotation/manual/ego_pose_gt_anno_val_public.json \
                    --offset \
                    --pred_path /workspace/project/ego-exo4d-egopose/handpose/models/RESNET/output/report_inference_outputs/exp1a/ego_pose_pred_anno_val.json
```


## Note
For the 21 keypoints annotation in each hand, its index and label are listed as below:
```
{0: Wrist,
 1: Thumb_1, 2: Thumb_2, 3: Thumb_3, 4: Thumb_4,
 5: Index_1, 6: Index_2, 7: Index_3, 8: Index_4,
 9: Middle_1, 10: Middle_2, 11: Middle_3, 12: Middle_4,
 13: Ring_1, 14: Ring_2, 15: Ring_3, 16: Ring_4,
 17: Pinky_1, 18: Pinky_2, 19: Pinky_3, 20: Pinky_4}
```
The 21 keypoints for right hand are visualized below, with left hand has symmetric keypoints position. 

<img src="assets/hand_index.png" width ="350" height="400">
