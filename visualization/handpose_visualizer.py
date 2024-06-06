import argparse
import cv2
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import sys

sys.path.append('..')
from data_preparation.utils.utils import get_ego_aria_cam_name, cam_to_img


class HandPoseVisualizer:
    def __init__(self, egoexo_output_dir, gt_output_dir, pred_output_dir, visualization_dir, anno_type="manual", pred_offset=False, seed=None):
        self.egoexo_output_dir = egoexo_output_dir
        self.gt_output_dir = gt_output_dir
        self.pred_output_dir = pred_output_dir
        self.visualization_dir = visualization_dir
        self.offset = pred_offset

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        assert anno_type in [
            "manual", "auto"], f"Invalid anno_type: {anno_type}"
        self.anno_type = anno_type

        self.takes = json.load(
            open(os.path.join(self.egoexo_output_dir, "takes.json")))

    def project_3d_to_2d(self, two_hand_kpts_3d, intrinsics):
        """
        Project 3D hand keypoints to 2D using camera intrinsics.

        Parameters:
        - two_hand_kpts_3d: dict with keys 'left' and 'right', each containing 3D keypoints
        - intrinsics: camera intrinsics matrix

        Returns:
        - projected 2D keypoints
        """
        assert isinstance(two_hand_kpts_3d, dict) and len(
            two_hand_kpts_3d) == 2
        hand_kpts_2d = {}

        for hand_order in ["left", "right"]:
            kpts_3d = two_hand_kpts_3d[hand_order]
            if len(kpts_3d) > 0:
                hand_kpts_2d[hand_order] = cam_to_img(kpts_3d, intrinsics)
            else:
                hand_kpts_2d[hand_order] = kpts_3d

        return hand_kpts_2d

    def vis_2d_hand_pose(self, img, gt_hand_kpts_2d, pred_hand_kpts_2d, take_name, frame_idx, split, label="Projected", save_to_file=False):
        """
        Visualize the ground truth and predicted 2D hand keypoints on the same image.

        Parameters:
        - img: the image to plot the keypoints on
        - gt_hand_kpts_2d: dict with keys 'left' and 'right', each containing ground truth 2D keypoints
        - pred_hand_kpts_2d: dict with keys 'left' and 'right', each containing predicted 2D keypoints
        - take_name: name of the take
        - frame_idx: index of the frame
        - label: If the 2d keypoints are projected from 3d or gt 2d.
        - save_to_file: whether to save the plot to a file
        """
        finger_index = np.array([[0, 1, 2, 3, 4],
                                 [0, 5, 6, 7, 8],
                                 [0, 9, 10, 11, 12],
                                 [0, 13, 14, 15, 16],
                                 [0, 17, 18, 19, 20]])
        color_dict = {0: 'tab:blue', 1: 'tab:orange',
                      2: 'tab:green', 3: 'tab:red', 4: 'tab:purple'}
        gt_color = 'tab:gray'
        pred_color = 'tab:blue'

        assert isinstance(gt_hand_kpts_2d, dict) and len(gt_hand_kpts_2d) == 2
        assert isinstance(pred_hand_kpts_2d, dict) and len(
            pred_hand_kpts_2d) == 2

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        for hand_order in ["left", "right"]:
            gt_one_hand_kpts_2d = gt_hand_kpts_2d[hand_order]
            pred_one_hand_kpts_2d = pred_hand_kpts_2d[hand_order]

            if len(gt_one_hand_kpts_2d) > 0:
                for i, each_finger_index in enumerate(finger_index):
                    curr_finger_kpts = gt_one_hand_kpts_2d[each_finger_index]
                    plt.plot(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], marker='o', markersize=2,
                             color=gt_color, label='Ground Truth' if i == 0 and hand_order == 'left' else "")

            if len(pred_one_hand_kpts_2d) > 0:
                for i, each_finger_index in enumerate(finger_index):
                    curr_finger_kpts = pred_one_hand_kpts_2d[each_finger_index]
                    # plt.plot(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], marker='o', markersize=2,
                    #          color=color_dict[i], label='Prediction' if i == 0 and hand_order == 'left' else "")
                    plt.plot(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], marker='o', markersize=2,
                             color=pred_color, label='Prediction' if i == 0 and hand_order == 'left' else "")

        plt.title(
            f"[{split.capitalize()}] {label} {take_name} - frame_idx={frame_idx}", fontsize=10)
        plt.axis("off")
        plt.legend()

        if save_to_file:
            split_dir = os.path.join(self.visualization_dir, split)
            take_dir = os.path.join(split_dir, take_name)
            frame_dir = os.path.join(take_dir, str(frame_idx))
            os.makedirs(frame_dir, exist_ok=True)
            plt.savefig(os.path.join(
                frame_dir, f'gt_and_pred_{label}_hand_pose.png'))
        else:
            plt.show()

    def vis_gt_and_pred_3d_hand_pose(self, gt_two_hand_kpts_3d, pred_two_hand_kpts_3d, take_name, frame_idx, split, save_to_file=False):
        """
        Visualize the ground truth and predicted 3D hand keypoints on the same plot.

        Parameters:
        - gt_two_hand_kpts_3d: dict with keys 'left' and 'right', each containing ground truth 3D keypoints
        - pred_two_hand_kpts_3d: dict with keys 'left' and 'right', each containing predicted 3D keypoints
        - take_name: name of the take
        - frame_idx: index of the frame
        - save_to_file: whether to save the plot to a file
        """
        finger_index = np.array([[0, 1, 2, 3, 4],
                                 [0, 5, 6, 7, 8],
                                 [0, 9, 10, 11, 12],
                                 [0, 13, 14, 15, 16],
                                 [0, 17, 18, 19, 20]])
        color_dict = {0: 'tab:blue', 1: 'tab:orange',
                      2: 'tab:green', 3: 'tab:red', 4: 'tab:purple'}
        gt_color = 'tab:gray'
        pred_color = 'tab:blue'

        assert isinstance(gt_two_hand_kpts_3d, dict) and len(
            gt_two_hand_kpts_3d) == 2
        assert isinstance(pred_two_hand_kpts_3d, dict) and len(
            pred_two_hand_kpts_3d) == 2

        fig = plt.figure(figsize=plt.figaspect(0.5))
        fig.suptitle(f"[{split.capitalize()}] 3D hand poses {take_name} - frame_idx={frame_idx}", fontsize=12)
        for i, hand_order in enumerate(["left", "right"]):
            gt_one_hand_kpts_3d = gt_two_hand_kpts_3d[hand_order]
            pred_one_hand_kpts_3d = pred_two_hand_kpts_3d[hand_order]
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.set_title(f"3D plot - {hand_order} hand")

            if len(gt_one_hand_kpts_3d) > 0:
                for f_ith, each_finger_index in enumerate(finger_index):
                    curr_finger_kpts = gt_one_hand_kpts_3d[each_finger_index]
                    ax.scatter(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], curr_finger_kpts[:, 2],
                               color=gt_color, label='Ground Truth' if f_ith == 0 else "")
                    ax.plot3D(
                        curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], curr_finger_kpts[:, 2], color=gt_color)

            if len(pred_one_hand_kpts_3d) > 0:
                for f_ith, each_finger_index in enumerate(finger_index):
                    curr_finger_kpts = pred_one_hand_kpts_3d[each_finger_index]
                    # ax.scatter(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], curr_finger_kpts[:, 2],
                    #            color=color_dict[f_ith], label='Prediction' if f_ith == 0 and hand_order == 'left' else "")
                    # ax.plot3D(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1],
                    #           curr_finger_kpts[:, 2], color=color_dict[f_ith])

                    ax.scatter(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1], curr_finger_kpts[:, 2],
                               color=pred_color, label='Prediction' if f_ith == 0 else "")
                    ax.plot3D(curr_finger_kpts[:, 0], curr_finger_kpts[:, 1],
                              curr_finger_kpts[:, 2], color=pred_color)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_aspect('equal')
            ax.legend()

        if save_to_file:
            split_dir = os.path.join(self.visualization_dir, split)
            take_dir = os.path.join(split_dir, take_name)
            frame_dir = os.path.join(take_dir, str(frame_idx))
            os.makedirs(frame_dir, exist_ok=True)
            plt.savefig(os.path.join(
                frame_dir, 'gt_and_pred_3d_hand_pose.png'))
        else:
            plt.show()

    def load_split_data(self, split):
        gt_anno_path = os.path.join(
            self.gt_output_dir, f"annotation/{self.anno_type}/ego_pose_gt_anno_{split}_public.json")
        assert os.path.exists(gt_anno_path), f"{gt_anno_path} doesn't exist"
        gt_anno = json.load(open(gt_anno_path))

        pred_anno_path = os.path.join(
            self.pred_output_dir, f"ego_pose_pred_anno_{split}.json")
        assert os.path.exists(
            pred_anno_path), f"{pred_anno_path} doesn't exist"
        pred_anno = json.load(open(pred_anno_path))

        take_to_uid = {each_take['take_name']: each_take['take_uid']
                       for each_take in self.takes if each_take["take_uid"] in gt_anno.keys()}
        uid_to_take = {uid: take for take, uid in take_to_uid.items()}

        return gt_anno, pred_anno, take_to_uid, uid_to_take

    def visualize_splits(self, splits=["train", "val", "test"], num_takes_per_split=10, save_to_file=False):

        for split in splits:
            assert split in ["train", "val", "test"], f"Invalid split: {split}"

            gt_anno, pred_anno, take_to_uid, uid_to_take = self.load_split_data(
                split)
            selected_takes = random.sample(list(take_to_uid.keys()), min(
                num_takes_per_split, len(take_to_uid)))

            for take_name in selected_takes:
                take_uid = take_to_uid[take_name]
                take_gt_anno = gt_anno[take_uid]
                take_pred_anno = pred_anno[take_uid]

                frame_idx = random.choice(list(take_gt_anno.keys()))

                self.visualize_frame(take_name, take_uid, frame_idx,
                                     take_gt_anno[frame_idx], take_pred_anno[frame_idx], split, save_to_file)

    def visualize_frame(self, take_name, take_uid, frame_idx, gt_anno_frame, pred_anno_frame, split, save_to_file=False):
        """
        Visualize a specific frame with ground truth and predicted annotations.

        Args:
            take_name (str): name of the take
            take_uid (str): unique identifier of the take
            frame_idx (int): index of the frame
            gt_anno_frame (dict): ground truth annotations for the frame
            pred_anno_frame (dict): predicted annotations for the frame
            split (str): dataset split (train, val, test)
            save_to_file (bool, optional): whether to save the visualization to a file. Defaults to False.
        """

        # print("--------------------------------------------------------------------------------")
        # print(f"take_name: {take_name}")
        # print(f"take_uid: {take_uid}")
        # print(f"frame_idx: {frame_idx}")
        # print(f"gt_anno_frame: {gt_anno_frame}")
        # print(f"pred_anno_frame: {pred_anno_frame}")
        # print(f"split: {split}")
        # print(f"save_to_file: {save_to_file}")
        # print("--------------------------------------------------------------------------------")
        img_dir = os.path.join(
            self.gt_output_dir, f"image/undistorted/{split}")
        img_path = os.path.join(
            img_dir, take_name, f"{int(frame_idx):06d}.jpg")
        img = np.array(Image.open(img_path))

        gt_two_hand_kpts_3d = {
            "right": np.array(gt_anno_frame['right_hand_3d']).astype(np.float32),
            "left": np.array(gt_anno_frame['left_hand_3d']).astype(np.float32)
        }

        pred_two_hand_kpts_3d = {
            "right": np.array(pred_anno_frame['right_hand_3d']).astype(np.float32),
            "left": np.array(pred_anno_frame['left_hand_3d']).astype(np.float32)
        }

        if self.offset and len(gt_two_hand_kpts_3d["right"]) > 0:
            pred_two_hand_kpts_3d["right"] += gt_two_hand_kpts_3d["right"][0]
        if self.offset and len(gt_two_hand_kpts_3d["left"]) > 0:
            pred_two_hand_kpts_3d["left"] += gt_two_hand_kpts_3d["left"][0]

        cam_pose_dir = os.path.join(
            self.egoexo_output_dir, f"annotations/ego_pose/{split}/camera_pose")
        cam_pose_path = os.path.join(cam_pose_dir, f"{take_uid}.json")
        cam_pose = json.load(open(cam_pose_path))
        aria_name = get_ego_aria_cam_name(
            [t for t in self.takes if t['take_name'] == take_name][0])
        intrinsics = np.array(
            cam_pose[aria_name]["camera_intrinsics"]).astype(np.float32)

        gt_two_hands_kpts_2d_proj = self.project_3d_to_2d(
            gt_two_hand_kpts_3d, intrinsics)
        pred_two_hands_kpts_2d_proj = self.project_3d_to_2d(
            pred_two_hand_kpts_3d, intrinsics)

        self.vis_2d_hand_pose(img, gt_two_hands_kpts_2d_proj, pred_two_hands_kpts_2d_proj,
                              take_name, frame_idx, split, label="Projected 3D", save_to_file=save_to_file)
        self.vis_gt_and_pred_3d_hand_pose(
            gt_two_hand_kpts_3d, pred_two_hand_kpts_3d, take_name, frame_idx, split, save_to_file=save_to_file)


def main(args):
    visualizer = HandPoseVisualizer(
        args.egoexo_output_dir,
        args.gt_output_dir,
        args.pred_output_dir,
        args.visualization_dir,
        args.anno_type,
        args.offset
    )
    visualizer.visualize_splits(
        splits=args.splits, num_takes_per_split=args.num_takes_per_split, save_to_file=args.save_to_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D hand pose annotations and predictions.")
    parser.add_argument("--egoexo_output_dir", type=str, required=True,
                        help="Directory containing EgoExo dataset outputs.")
    parser.add_argument("--gt_output_dir", type=str, required=True,
                        help="Directory containing ground truth annotations.")
    parser.add_argument("--pred_output_dir", type=str, required=True,
                        help="Directory containing predicted annotations.")
    parser.add_argument("--visualization_dir", type=str,
                        required=True, help="Directory to save visualizations.")
    parser.add_argument("--anno_type", type=str, default="manual",
                        choices=["manual", "auto"], help="Type of annotation (manual or auto).")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"], help="Dataset splits to visualize.")
    parser.add_argument("--num_takes_per_split", type=int,
                        default=2, help="Number of takes to visualize per split.")
    parser.add_argument("--offset", action="store_true",
                        help="Whether the predicted annotations are offset by the wrist position.")
    parser.add_argument("--save_to_file", action="store_true",
                        help="Whether to save the visualizations to files.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)


"""
Example usage:
python3 handpose_visualizer.py --egoexo_output_dir /workspace/project/ego-exo_data \
                               --gt_output_dir /workspace/project/ego-exo_gt_data \
                               --pred_output_dir /workspace/project/ego-exo4d-egopose/handpose/models/RESNET/output/inference_output \
                               --visualization_dir /workspace/project/ego-exo4d-egopose/handpose/models/RESNET/output/visualizations \
                               --splits train val \
                               --num_takes_per_split 2 \
                               --offset \
                               --seed 10 \
                               --save_to_file
"""
