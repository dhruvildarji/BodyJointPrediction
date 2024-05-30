import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from BodyJointPrediction.dataloader import HandPoseDataLoader, HandPoseDataset
from BodyJointPrediction.train import SimpleCNN

def load_ground_truth(gt_path):
    with open(gt_path, 'r') as file:
        ground_truth = json.load(file)
    return ground_truth

def evaluate_model(model, dataloader, ground_truth):
    model.eval()
    total_error = 0.0
    total_joints = 0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            
            # Convert outputs and labels to numpy for easier manipulation
            outputs_np = outputs.numpy()
            labels_np = labels.numpy()

            for i in range(outputs_np.shape[0]):
                gt_pose = ground_truth[i]
                pred_pose = outputs_np[i]
                error = np.linalg.norm(gt_pose - pred_pose, axis=1)
                total_error += error.sum()
                total_joints += error.shape[0]

    mean_error = total_error / total_joints
    return mean_error

def main():
    root_dir = '/Users/rdhara/Downloads/ego-exo4d-egopose/handpose/cs231project/dataset'
    gt_path = '/path/to/ground_truth.json'  # Update this path with the actual ground truth file path

    hand_pose_loader = HandPoseDataLoader(root_dir)
    dataset = HandPoseDataset(hand_pose_loader)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    model = SimpleCNN()
    model.load_state_dict(torch.load('hand_pose_model.pth'))  # Load the trained model weights

    ground_truth = load_ground_truth(gt_path)
    mean_error = evaluate_model(model, dataloader, ground_truth)

    print(f'Mean Error per Joint: {mean_error:.4f}')

if __name__ == "__main__":
    main()