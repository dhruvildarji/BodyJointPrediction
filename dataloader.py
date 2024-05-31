import os
import json
import cv2
import glob
from pathlib import Path
import pickle

class HandPoseDataLoader:
    def __init__(self, root_dir, data_type="train", use_pickle=False):
        self.root_dir = root_dir
        self.json_dir = os.path.join(root_dir, "annotations", "ego_pose", data_type, "camera_pose")
        self.video_dir = os.path.join(root_dir, "takes")
        self.hand_pose_annotations = os.path.join(root_dir, "annotations", "ego_pose", data_type, "hand", "annotation")
        self.hand_pose_automatics = os.path.join(root_dir, "annotations", "ego_pose", data_type, "hand", "automatics")
        # self.pickle_file_path = os.path.join(root_dir, "data", "data.pkl")
        self.json_files = self.find_json_files()
        self.data = {}
        self.load_data()
        self.video_files = {}
        self.video_caps = {}
        self.load_videos()
        self.current_index = 0
        self.video_dirs = []
        self.data_type = data_type
        if not use_pickle:
            self.hand_data = self.load_hand_data()
            self.init_videos()
            # self.save_file_pickle()

    def __iter__(self):
        return self

    def convert_generators(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_generators(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return type(obj)(self.convert_generators(v) for v in obj)
        elif isinstance(obj, type((lambda: (yield))())):  # Check if it's a generator
            return list(obj)  # Convert generator to list
        else:
            return obj

    def save_file_pickle(self):
        data_to_save = self.convert_generators(self.data)
        with open(self.pickle_file_path, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Data saved to {self.pickle_file_path}")

    def find_json_files(self):
        json_files = []
        for json_filename in os.listdir(self.json_dir):
            if json_filename.endswith('.json'):
                json_files.append(os.path.join(self.json_dir, json_filename))
        return json_files

    def find_videos_with_214(self):
        pattern = os.path.join(self.video_dir, '**', '*214*.mp4')
        video_files_ = glob.glob(pattern, recursive=True)
        files_ = {}
        for filename in video_files_:
            file_name_ = Path(filename).parent.parent.name

            if file_name_ in self.data["take_name"]:
                if file_name_ not in self.video_files:
                    files_[file_name_] = filename

        return files_

    def get_video_dir(self):
        self.video_dirs = os.listdir(self.video_dir)

    def load_hand_data(self):
        for files in os.listdir(self.hand_pose_annotations):
            if files.endswith('.json'):
                file = files.split(".")[0]
                if file in self.data["take_uid"]:
                    int_ = self.get_int_from_dict("take_uid", file)
                    self.data[int_]["hand_pose"] = {}
                    with open(os.path.join(self.hand_pose_annotations, f"{file}.json"), 'r') as file_:
                        json_data = json.load(file_)
                        self.data[int_]["hand_pose"] = json_data

    def get_int_from_dict(self, dictionary_key, target_string):
        for key, value in self.data.items():
            if dictionary_key in value:
                if value[dictionary_key] == target_string:
                    return key
        return None

    def init_videos(self):
        for idx, file_ in self.video_files.items():
            cap = cv2.VideoCapture(file_)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            c = 0
            int_ = self.get_int_from_dict("take_name", idx)
            self.data[int_]["frames"] = []

            for frame_num in self.data[int_]["hand_pose"].keys():
                c = c + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
                ret, frame = cap.read()
                if ret:
                    self.data[int_]["frames"].append(frame)

                if c == 25:
                    break
            cap.release()

    def load_data(self):
        self.data["take_name"] = []
        self.data["take_uid"] = []
        for idx, file_path in enumerate(self.json_files):
            self.data[idx] = {}
            self.data[idx]["cam_pose"] = []
            with open(file_path, 'r') as file:
                json_data = json.load(file)

    def load_videos(self):
        self.video_files = self.find_videos_with_214()

    def get_intrinsics(self, file_name):
        if file_name in self.data and "camera_intrinsics" in self.data[file_name]:
            return self.data[file_name]["camera_intrinsics"]
        else:
            return None
    
    def get_distortion(self, file_name):
        if file_name in self.data and "distortion" in self.data[file_name]:
            return self.data[file_name]["distortion"]
        else:
            return None

    def get_hand_poses(self, file_name):
        if file_name in self.data and "hand_pose" in self.data[file_name]:
            return self.data[file_name]["hand_pose"]
        else:
            return None

    def release(self):
        for cap in self.video_caps.values():
            cap.release()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        while self.current_index < len(self.data["take_name"]):
            file_name = self.data["take_name"][self.current_index]
            intrinsics = self.get_intrinsics(file_name)
            if intrinsics is not None:
                item = {
                    "type": self.data_type,
                    "file_name": file_name,
                    "intrinsics": intrinsics,
                    "hand_pose": self.get_hand_poses(file_name),
                    "distortion": self.get_distortion(file_name),
                    "video_file": self.video_files[file_name],
                    "frames": self.data[self.current_index]["frames"]
                }
                self.current_index += 1
                return item
            else:
                print("Ignoring file without camera intrinsics:", file_name)
                self.current_index += 1
        raise StopIteration


import torch
from torch.utils.data import Dataset

class HandPoseDataset(Dataset):
    def __init__(self, hand_pose_loader):
        self.data = []
        for item in hand_pose_loader:
            if item is not None and item.get("intrinsics") is not None:
                frames = item['frames']
                hand_poses = item['hand_pose']
                if frames is not None and hand_poses is not None:
                    for frame, hand_pose in zip(frames, hand_poses.values()):
                        self.data.append((frame, hand_pose))
            else:
                print("Ignoring null item or missing camera intrinsics.")
                print("File name:", item.get("file_name"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame, hand_pose = self.data[idx]
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize and permute for channels
        hand_pose = torch.tensor(hand_pose, dtype=torch.float32)
        return frame, hand_pose
