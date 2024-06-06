"""
RESNET implementation
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class ResNetDirectRegressionPose(nn.Module):
    def __init__(self, **kwargs):
        super(ResNetDirectRegressionPose, self).__init__()
        self.num_joints = kwargs["NUM_JOINTS"]
        self.img_H = kwargs["IMAGE_SIZE"][0]
        self.img_W = kwargs["IMAGE_SIZE"][1]
        # Depth dimension for 3D pose estimation
        self.depth_dim = kwargs["EXTRA"]["DEPTH_DIM"]

        # Load the pre-trained ResNet model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Number of features in the last layer of ResNet
        in_features = self.resnet.fc.in_features
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()

        self.pose_layer = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            # Predicting 3D coordinates directly
            nn.Linear(256, self.num_joints * 3)
        )

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)
        # print(f"Shape after ResNet: {x.shape}")

        # Forward pass through the pose layer
        out = self.pose_layer(x)  # [N, num_joints * depth_dim]
        out = out.view(-1, self.num_joints, 3)
        return out
    
class ResNet3DConvPose(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet3DConvPose, self).__init__()
        self.num_joints = kwargs["NUM_JOINTS"]
        self.img_H = kwargs["IMAGE_SIZE"][0]
        self.img_W = kwargs["IMAGE_SIZE"][1]
        # Depth dimension for 3D pose estimation
        self.depth_dim = kwargs["EXTRA"]["DEPTH_DIM"]

        # Load the pre-trained ResNet model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.conv_layer4 = self.resnet.layer4

        ######### 2D pose head #########
        # self.norm1 = GroupNorm(256)
        # self.up_sample = nn.Sequential(
        #     nn.Conv2d(self.embed_dims[0], 256, 1),
        #     nn.GELU(),
        # )
        self.final_layer = nn.Conv2d(
            512, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0
        )
        self.pose_layer = nn.Sequential(
            nn.Conv3d(self.num_joints, self.num_joints, 1),
            nn.GELU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            # nn.Dropout(p=0.2),
            nn.Conv3d(self.num_joints, self.num_joints, 1),
        )

        ######### 3D pose head #########
        self.pose_3d_head = nn.Sequential(
            nn.Linear(70, 512), # 70 = depth_dim + H_feat + W_feat
            nn.ReLU(),
            nn.GroupNorm(self.num_joints, self.num_joints),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        x_feature = self.features(x)
        # print("Features Output Shape: ", x_feature.shape) # feature_map: [N, 512, H_feat, W_feat]
        
        # out = self.up_sample(x_feature)
        # out = self.up_sample(x_feature)  # [N, 256, H_feat, W_feat]
        # # print("Upsampled Output Shape: ", out.shape)
        # out = self.norm1(out)
        # out = self.final_layer(out)  # [N, num_joints*emb_dim, H_feat, W_feat]
        # # print("Final Layer Output Shape: ", out.shape)

        out = self.final_layer(x_feature) # [N, num_joints*emb_dim, H_feat, W_feat]
        # print("Final Layer Output Shape: ", out.shape)

        out = self.pose_layer(
            out.reshape(
                out.shape[0],
                self.num_joints,
                self.depth_dim,
                out.shape[2],
                out.shape[3],
            )
        )  # (N, num_joints, emb_dim, H_feat, W_feat)
        # print("Pose Layer Output Shape: ", out.shape)

        # 3D pose head
        hm_x0 = out.sum((3, 4))
        hm_y0 = out.sum((2, 4))
        hm_z0 = out.sum((2, 3))
        pose_3d_pred = torch.cat((hm_x0, hm_y0, hm_z0), dim=2)
        # print("Before pose_3d_head: ", pose_3d_pred.shape)

        pose_3d_pred = self.pose_3d_head(pose_3d_pred)
        # print("3D Pose Output Shape: ", pose_3d_pred.shape)

        return pose_3d_pred


def load_pretrained_weights(model, checkpoint):
    import collections

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("backbone."):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(matched_layers)} pretrained parameters")
