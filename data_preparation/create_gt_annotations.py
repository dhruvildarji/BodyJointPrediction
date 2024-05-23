import os
import json 
import numpy as np
from dataloader import HandPoseDataLoader

from utils import (
    get_ego_aria_cam_name,
    world_to_cam
)

if __name__ == "__main__":
    dataset_root_dir = "/workspace/project/data"
    splits = ["train", "val"]
    loader = HandPoseDataLoader(dataset_root_dir)

    keypoints = [
        'right_wrist', 'right_thumb_1', 'right_thumb_2', 'right_thumb_3', 'right_thumb_4',
        'right_index_1', 'right_index_2', 'right_index_3', 'right_index_4',
        'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_middle_4',
        'right_ring_1', 'right_ring_2', 'right_ring_3', 'right_ring_4',
        'right_pinky_1', 'right_pinky_2', 'right_pinky_3', 'right_pinky_4',
        'left_wrist', 'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb_4',
        'left_index_1', 'left_index_2', 'left_index_3', 'left_index_4',
        'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_middle_4',
        'left_ring_1', 'left_ring_2', 'left_ring_3', 'left_ring_4',
        'left_pinky_1', 'left_pinky_2', 'left_pinky_3', 'left_pinky_4'
    ]

    kpt_idx_mapping = {kpt: i for i, kpt in enumerate(keypoints)}

    for split in splits:
        print(f"Processing annotations in {split}")
        gt_annotations_dir = os.path.join(dataset_root_dir, f"annotations/ego_pose/{split}/hand/gt_annotation")

        for uid, annotation in loader.annotations[split].items():
            print(f"\tProcessing annotations for {uid}")
            
            take = [t for t in loader.takes_metadata if t["take_uid"] == uid]
            take = take[0]
            aria_cam_name = get_ego_aria_cam_name(take)

            gt_annotations_3d = {}

            if uid in loader.camera_poses[split].keys():
                campera_pose = loader.camera_poses[split][uid]
            else:
                print(f"Missing camera pose for take: {uid}")
                continue

            for frame_id, ann_data in annotation.items():
                extrinsics = np.array(campera_pose[aria_cam_name]["camera_extrinsics"][frame_id])
                frame_ann_3d = ann_data[0]["annotation3D"]

                kpts_3d_world = np.full((42, 3), np.nan)
                for kpt, coord in frame_ann_3d.items():
                    if kpt in kpt_idx_mapping:
                        idx = kpt_idx_mapping[kpt]
                        kpts_3d_world[idx] = [coord["x"], coord["y"], coord["z"]]

                        # Transform
                        keypoints_3d_cam = world_to_cam(kpts_3d_world, extrinsics)

                gt_annotations_3d[frame_id] = {}
                for i, keypoint in enumerate(keypoints):
                    if not np.isnan(keypoints_3d_cam[i]).any():
                        gt_annotations_3d[frame_id][keypoint] = {
                            "x": keypoints_3d_cam[i, 0],
                            "y": keypoints_3d_cam[i, 1],
                            "z": keypoints_3d_cam[i, 2]
                        }

            # Write json for gt_annotations_3d for that uid
            output_json_path = os.path.join(gt_annotations_dir, f"{uid}.json")
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(gt_annotations_3d, f, indent=2)

                