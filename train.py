import torch
from torch.utils.data import Dataset, DataLoader
from utils.DataLoader import HandPoseDataLoader
import pickle
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

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
# Create an instance of the custom Dataset
dataset = HandPoseDataset(data_loader)

print(dataset.data.keys())

# Function to convert tensor to OpenCV image
def tensor_to_cv2_image(tensor):
    # Convert tensor to a numpy array and then to an OpenCV image
    image = tensor.numpy().transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

# Define the custom crop function
def custom_crop(image):
    width, height = image.size
    
    # Calculate the crop dimensions
    top_crop = int(height * 0.25)  # 25% from top
    left_crop = int(width * 0.05)  # 5% from left
    right_crop = int(width * 0.95)  # 5% from right (keeping 95% width)
    bottom_crop = height  # Keep bottom as it is

    # Perform the crop
    return image.crop((left_crop, top_crop, right_crop, bottom_crop))


for i in dataset:
    for frame in i["frames"]:
        # cv2.imshow("frame", np.array(frame))
        # cv2.waitKey(1)
        frame = np.array(frame)

        # Convert the image from BGR (OpenCV default) to RGB
        cv2_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV image to a PIL image
        pil_image = Image.fromarray(cv2_image_rgb)

        # Define the transformation pipeline with the custom crop
        transform = transforms.Compose([
            transforms.Lambda(custom_crop),  # Apply the custom crop
            transforms.ToTensor(),           # Convert the cropped image to a tensor
        ])

        # Apply the transformation
        transformed_image = transform(pil_image)

        # Convert the first transformed corner crop to an OpenCV image
        cv2_transformed_image = tensor_to_cv2_image(transformed_image)

        # Display the image using OpenCV
        cv2.imshow('Transformed Image', cv2_transformed_image)
        k  = cv2.waitKey(0)  # Wait for a key press to close the image window
        # if k == 27:
        #     break
    # break



# Create a DataLoader instance
# batch_size = 1
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# print(dataloader)
# Iterate through the DataLoader
# Debugging: Print the length of the dataset
print(f"Length of dataset: {len(dataset)}")

# Debugging: Iterate through the DataLoader with additional print statements
# try:
#     for batch_idx, batch in enumerate(dataloader):
#         print(f"Processing batch {batch_idx}")
#         # Your training code here
# except Exception as e:
#     print(f"An error occurred: {e}")