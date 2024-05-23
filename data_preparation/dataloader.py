import os
import json

class HandPoseDataLoader:
    def __init__(self, data_root_dir, splits=["train", "val"]):
        self.root_dir = data_root_dir
        self.splits = splits
        self.takes_metadata = None
        self.annotations = {}
        self.camera_poses = {}

        self._load_all_data()

    def _load_all_data(self):
        for split in self.splits:
            self.takes_metadata = self._load_takes_metadata()
            self.annotations[split] = self._load_annotations(split)
            self.camera_poses[split] = self._load_camera_poses(split)

    def _load_takes_metadata(self):
        file_path = os.path.join(self.root_dir, "takes.json") 
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def _load_annotations(self, split):
        annotations = {}
        annotation_dir = os.path.join(self.root_dir, f"annotations/ego_pose/{split}/hand/annotation")
        for file_name in os.listdir(annotation_dir):
            if file_name.endswith('.json'):
                take_uid = file_name.split('.')[0]
                file_path = os.path.join(annotation_dir, file_name)
                with open(file_path, 'r') as f:
                    annotations[take_uid] = json.load(f)
        return annotations

    def _load_camera_poses(self, split):
        camera_poses = {}
        camera_pose_dir = os.path.join(self.root_dir, f"annotations/ego_pose/{split}/camera_pose")
        for file_name in os.listdir(camera_pose_dir):
            if file_name.endswith('.json'):
                take_uid = file_name.split('.')[0]
                file_path = os.path.join(camera_pose_dir, file_name)
                with open(file_path, 'r') as f:
                    camera_poses[take_uid] = json.load(f)
        return camera_poses
    
    def _load_ego_takes(self, split): 
        pass

# Example usage
if __name__ == "__main__":
    dataset_root_dir = "/workspace/project/data"
    splits = ["train", "val"]
    loader = HandPoseDataLoader(dataset_root_dir)

    for split in splits:
        print(f"{split} annotations keys: {loader.annotations[split].keys()}")
        print(f"{split} camera pose keys: {loader.camera_poses[split].keys()}")

        uid = list(loader.annotations[split].keys())[0]
        print(f"{split} number of annotatated frames in : {len(loader.annotations[split][uid])}")
        print(f"{split} number of camera extrinsics: {len(loader.camera_poses[split][uid]['aria02']['camera_extrinsics'])}")
        break

    