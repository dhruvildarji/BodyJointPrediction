# Ego-Exo4D Hand Ego Pose Benchmark Evaluation

Here is an example for ego hand pose model evaluation.   
For challenge participants, please submit model outputs to [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2249/overview)
We also proved a [dummy ground truth file](https://drive.google.com/file/d/1WGnd7aPXeVRZsTfSybGcl5uIC1ZC_bQp/view?usp=sharing) to check the model output format for submission. 

## File requirements
### Model inference results 
The inference output needs to be saved as a single JSON file with the following format:
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
### Ground truth annotation JSON file 
Ground truth annotation for test split is not released to public. Please download a [dummy ground truth file](https://drive.google.com/file/d/1WGnd7aPXeVRZsTfSybGcl5uIC1ZC_bQp/view?usp=sharing) for test split.  
Note that the dummy ground truth is generated by random numbers. It is not meant to evaluate the method but just to check the format.

## Evaluation

Evaluate the model performance based on prediction inference output JSON file (`<pred_path>`) and ground truth JSON file (`<gt_path>`). Remember to set `offset` if the user inference output is offset by hand wrist. 
```
python3 evaluate.py \
    --pred_path <pred_path> \
    --gt_path <gt_path> 
    --offset
```