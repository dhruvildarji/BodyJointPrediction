import os
import json
import cv2
import glob
from pathlib2 import Path

class HandPoseDataLoader:
    def __init__(self, root_dir, data_type = "train"):
        self.root_dir = root_dir
        self.json_dir = f"{root_dir}/data/annotations/ego_pose/{data_type}/camera_pose"
        self.video_dir = f"{root_dir}/data/takes/"
        self.json_files = self.find_json_files()
        self.data = {}
        self.load_data()
        self.video_files = {}
        self.video_caps = {}
        self.load_videos()
        self.current_index = 0
        self.video_dirs = []
        self.data_type = data_type

    def find_json_files(self):
        json_files = []
        for json_filename in os.listdir(self.json_dir):
            if json_filename.endswith('.json'):
                json_files.append(f"{self.json_dir}/{json_filename}")
        print(len(json_files))
        return json_files

    def find_videos_with_214(self):
        pattern = os.path.join(self.video_dir, '**', '*214*.mp4')
        video_files_ = glob.glob(pattern, recursive=True)
        files_ = {}
        for filename in video_files_:
            file_name_ = Path(filename).parent.parent.name
            if file_name_ in self.data["take_name"]:
                print(filename, file_name_)
                if file_name_ not in self.video_files:
                    files_[file_name_] = filename
                
        return files_
    
    def get_video_dir(self):
        self.video_dirs = os.listdir(self.video_dir)


    def load_data(self):
        self.data["take_name"] = []
        self.data["take_uid"] = []
        for file_path in self.json_files:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                print(json_data.keys())
                for frame_keys in json_data.keys():
                    if "metadata" in frame_keys:
                        take_name = json_data["metadata"]["take_name"]
                        self.data[take_name] = {}
                        self.data[take_name]["cam_pose"] = []
                        self.data["take_name"].append(json_data["metadata"]["take_name"])
                        self.data["take_uid"].append(json_data["metadata"]["take_uid"])                
                    elif "aria" in frame_keys:                       
                        self.data[take_name]["camera_intrinsics"] = json_data[frame_keys]["camera_intrinsics"]
                        print("aria " ,json_data[frame_keys]["camera_intrinsics"])
                        self.data[take_name]["cam_pose"].append(json_data[frame_keys][i] for i in json_data[frame_keys].keys() if i.isdigit())
                    else:
                        self.data[take_name]["distortion"] = json_data[frame_keys]["distortion_coeffs"]

    def load_videos(self):
        self.video_files = self.find_videos_with_214()
        print( self.video_files)

    def get_intrinsics(self, file_name):
        return self.data[file_name]["camera_intrinsics"]
    
    def get_distortion(self, file_name):
        return self.data[file_name]["distortion"]

    def get_hand_poses(self, file_name):
        return self.data[file_name]["cam_pose"]

    def get_frames(self, video_name, frame_id):
        if video_name not in self.video_caps:
            raise ValueError(f"Video {video_name} not found")
        
        cap = self.video_caps[video_name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_id} from video {video_name}")
        return frame

    def release(self):
        for cap in self.video_caps.values():
            cap.release()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self.data["take_name"]):
            print(self.current_index , len(self.data["take_name"]))
            file_name = self.data["take_name"][self.current_index]
            item = {
                "type": self.data_type,
                "file_name": file_name,
                "intrinsics": self.get_intrinsics(file_name),
                "hand_poses": self.get_hand_poses(file_name),
                "distortion": self.get_distortion(file_name),
                "video_file" : self.video_files[file_name]
            }
        else:
            raise StopIteration
        
        self.current_index += 1
        return item