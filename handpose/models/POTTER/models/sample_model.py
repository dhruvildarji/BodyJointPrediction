import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# class SimpleCNN(nn.Module):
#     def __init__(self, num_joints=21, num_coords=3):
#         super(SimpleCNN, self).__init__()
#         self.num_joints = num_joints
#         self.num_coords = num_coords
        
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)

#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)

#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)

                
# #         # Define the final layers for 2D to 3D pose estimation
#         # self.conv5 = nn.Conv2d(512, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(512 * 14 * 14, 1024)
#         self.fc2 = nn.Linear(1024, num_joints * num_coords)
        
#         self.relu = nn.ReLU()
        
#         self.features = {}  # Dictionary to store the features
        
#         self._register_hooks()  # Register the hooks
    
#     def _register_hooks(self):
#         def hook_fn(module, input, output, key):
#             self.features[key] = output.detach()
        
#         # Register hooks for each layer you want to visualize
#         self.conv1.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv1'))
#         self.conv2.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv2'))
#         self.conv3.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv3'))
#         self.conv4.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv4'))
    
#     def forward(self, x):
#         x = self.pool(self.relu(self.bn1(self.conv1(x))))
#         x = self.pool(self.relu(self.bn2(self.conv2(x))))
#         x = self.pool(self.relu(self.bn3(self.conv3(x))))
#         x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
#         x = x.view(-1, 512 * 14 * 14)  # Flattening the feature map
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         x = x.view(-1, self.num_joints, self.num_coords)  # Reshaping to [batch_size, num_joints, num_coords]
#         return x

#     def get_features(self):
#         return self.features




class SimpleCNN(nn.Module):
    def __init__(self, num_joints=21, num_coords=3):
        super(SimpleCNN, self).__init__()
        self.num_joints = num_joints
        self.num_coords = num_coords
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.res1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.res2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.res3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.res4 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, num_joints * num_coords)
        
        self.relu = nn.ReLU()
        
        self.features = {}  # Dictionary to store the features
        
        self._register_hooks()  # Register the hooks
    
    def _register_hooks(self):
        def hook_fn(module, input, output, key):
            self.features[key] = output.detach()
        
        # Register hooks for each layer you want to visualize
        self.conv1.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv1'))
        self.conv2.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv2'))
        self.conv3.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv3'))
        self.conv4.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv4'))
    
    def forward(self, x):



        x = self.relu(self.bn1(self.conv1(x)))
        residual = self.res1(x)
        x = x +  residual
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        residual = self.res2(x)
        x = x +  residual
        x = self.pool(x)        

        x = self.relu(self.bn3(self.conv3(x)))
        residual = self.res3(x)
        x = x +  residual
        x = self.pool(x)

        x = self.relu(self.bn4(self.conv4(x)))
        residual = self.res4(x)
        x = x + residual
        x = self.pool(x)

        x = x.view(-1, 512 * 14 * 14)  # Flattening the feature map
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.view(-1, self.num_joints, self.num_coords)  # Reshaping to [batch_size, num_joints, num_coords]
        return x

    def get_features(self):
        return self.features



# class SimpleCNN(nn.Module):
#     def __init__(self, num_joints=21, depth_dim=64, input_channels=3, num_coords=3):
#         super(SimpleCNN, self).__init__()
#         self.num_joints = num_joints
#         self.depth_dim = depth_dim
#         self.num_coords = num_coords
#         # Define the CNN layers
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(512)
        
#         # Define the final layers for 2D to 3D pose estimation
#         self.conv5 = nn.Conv2d(512, self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)

#         self.pose_layer = nn.Sequential(
#             nn.Conv3d(self.num_joints, self.num_joints, 1),
#             nn.GELU(),
#             nn.GroupNorm(self.num_joints, self.num_joints),
#             nn.Conv3d(self.num_joints, self.num_joints, 1),
#         )

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(512 * 14 * 14, 1024)
#         self.fc2 = nn.Linear(1024, num_joints * num_coords)

#         self.relu = nn.ReLU()

        
# #         self.relu = nn.ReLU()
#         # self.pose_3d_head = nn.Sequential(
#         #     nn.Linear(self.depth_dim * 3, 512),
#         #     nn.ReLU(),
#         #     nn.GroupNorm(self.num_joints, self.num_joints),
#         #     nn.Linear(512, 3),
#         # )

#         self.features = {}  # Dictionary to store the features
        
#         self._register_hooks()  # Register the hooks
    
#     def _register_hooks(self):
#         def hook_fn(module, input, output, key):
#             self.features[key] = output.detach()
        
#         # Register hooks for each layer you want to visualize
#         self.conv1.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv1'))
#         self.conv2.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv2'))
#         self.conv3.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv3'))
#         self.conv4.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv4'))


#     def forward(self, x):
#         # Forward pass through the CNN layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))

#         # Reshape and forward pass through the final layers
#         out = self.conv5(x)  # [N, num_joints*depth_dim, H_feat, W_feat]
#         out = out.view(out.shape[0], self.num_joints, self.depth_dim, out.shape[2], out.shape[3])
#         x = self.pose_layer(out)  # (N, num_joints, depth_dim, H_feat, W_feat)


#         x = x.view(-1, 512 * 14 * 14)  # Flattening the feature map
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         x = x.view(-1, self.num_joints, self.num_coords)  # Reshaping to [batch_size, num_joints, num_coords]
#         return x
    
#         # 3D pose head
#         # hm_x0 = out.sum((2, 3))
#         # hm_y0 = out.sum((2, 4))
#         # hm_z0 = out.sum((3, 4))
#         # pose_3d_pred = torch.cat((hm_x0, hm_y0, hm_z0), dim=2)
#         # pose_3d_pred = self.pose_3d_head(pose_3d_pred)



#         # return pose_3d_pred

#     def get_features(self):
#         return self.features
