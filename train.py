import torch
from torch.utils.data import Dataset, DataLoader
from utils.DataLoader import HandPoseDataLoader
import pickle
import os

class HandPoseDataset(Dataset):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data = self.load_data_from_pickle_or_loader()

    def load_data_from_pickle_or_loader(self):
        pickle_file_path = self.data_loader.pickle_file_path
        print(pickle_file_path)
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as file:
                data = pickle.load(file)
            print("Data loaded from pickle file.")
        else:
            data = list(self.data_loader)  # Load all data into memory using data loader
            with open(pickle_file_path, 'wb') as file:
                pickle.dump(data, file)
            print(f"Data saved to {pickle_file_path}.")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self.data)}")

        item = self.data[idx]
        hand_pose = item['hand_pose']
        
        # Convert video frames and hand poses to tensors
        frames = [torch.tensor(frame) for frame in item['frames']]
        list1 = []
        for i in hand_pose.keys():
            hand_ = hand_pose[i]
            hand_ = hand_[0]
            for joints in hand_["annotation3D"]:
                list_ = []
                list_.append(hand_["annotation3D"][joints]["x"])
                list_.append(hand_["annotation3D"][joints]["y"])
                list_.append(hand_["annotation3D"][joints]["z"])
                list1.append(list_)
        
        hand_pose = torch.tensor(list1)
        
        sample = {
            'frames': frames,
            'hand_pose': hand_pose,
            'intrinsics': torch.tensor(item['camera_intrinsics']),
            'distortion': torch.tensor(item['distortion'])
        }
        
        return sample

# Initialize your HandPoseDataLoader
root_dir = "/mnt/d/git/dataset/handPose"
data_loader = HandPoseDataLoader(root_dir, data_type="train", use_pickle=True)
print(data_loader.keys())
# Create an instance of the custom Dataset
dataset = HandPoseDataset(data_loader)

# Create a DataLoader instance
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
print(dataloader)
# Iterate through the DataLoader
# Debugging: Print the length of the dataset
print(f"Length of dataset: {len(dataset)}")

# Debugging: Iterate through the DataLoader with additional print statements
try:
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx}")
        # Your training code here
except Exception as e:
    print(f"An error occurred: {e}")