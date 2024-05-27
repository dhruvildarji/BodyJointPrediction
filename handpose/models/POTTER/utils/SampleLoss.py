import torch
import torch.nn as nn

class SimplePose3DLoss(nn.Module):
    def __init__(self):
        super(SimplePose3DLoss, self).__init__()

    def forward(self, pose_3d_pred, pose_3d_gt, vis_flag):
        # Ensure the predictions and ground truths have the same shape
        # print(pose_3d_pred.shape)
        # print(pose_3d_gt.shape)
        # print(pose_3d_pred)
        # print(pose_3d_gt)

        assert (
            pose_3d_pred.shape == pose_3d_gt.shape and len(pose_3d_pred.shape) == 3
        ), "Shape mismatch or incorrect dimensions"

        # Compute the squared difference
        pose_3d_diff = pose_3d_pred - pose_3d_gt
        pose_3d_loss = (pose_3d_diff ** 2).mean(dim=2) * vis_flag
        
        # Compute the average loss over all visible joints
        pose_3d_loss = pose_3d_loss.sum() / vis_flag.sum()

        return pose_3d_loss


