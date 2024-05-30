import json
import numpy as np

def generate_dummy_ground_truth(num_samples, num_joints):
    ground_truth = []
    for _ in range(num_samples):
        gt_pose = np.random.rand(num_joints, 3).tolist()  # Assuming 3D coordinates
        ground_truth.append(gt_pose)
    return ground_truth

def main():
    num_samples = 100  # Number of samples to generate
    num_joints = 21  # Number of joints per hand pose
    gt_path = 'dummy_ground_truth.json'

    ground_truth = generate_dummy_ground_truth(num_samples, num_joints)

    with open(gt_path, 'w') as file:
        json.dump(ground_truth, file)

    print(f'Dummy ground truth saved to {gt_path}')

if __name__ == "__main__":
    main()