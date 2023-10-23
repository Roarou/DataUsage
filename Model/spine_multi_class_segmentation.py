import torch.nn as nn
import torch.nn.functional as F
from Model.utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation, PointNetSetAbstraction
import torch


class SpineSegmentationNet(nn.Module):

    def __init__(self, num_classes=6, normal_channel=False):
        super(SpineSegmentationNet, self).__init__()
        extra_channels = 3 if normal_channel else 0
        self.include_normal_channels = normal_channel
        self.abstraction1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + extra_channels,
                                                      [[32, 32, 64], [64, 64, 128],
                                                       [64, 96, 128]])
        self.abstraction2 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=320 + 3,
                                                   mlp=[256, 512, 1024],
                                                   group_all=True)
        self.feature_propagation2 = PointNetFeaturePropagation(in_channel=1344,
                                                               mlp=[256, 128])
        self.feature_propagation1 = PointNetFeaturePropagation(in_channel=131 + extra_channels,
                                                               mlp=[128, 128])
        self.conv_layer1 = nn.Conv1d(128, 128, 1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv_layer2 = nn.Conv1d(128, num_classes, 1)

        # Rest of the class methods (e.g., forward) remains the same

    def forward(self, point_cloud_xyz):

        # Set Abstraction layers
        # print(f'Input xyz shape: {point_cloud_xyz.shape}')
        if self.include_normal_channels:  # If the normal_channel flag is True
            # Input layer
            input_points = point_cloud_xyz.permute(0, 2, 1)
            input_xyz_coordinates = point_cloud_xyz[:, :, :3].permute(0, 2, 1)
        else:
            # Input layer
            input_points = point_cloud_xyz.permute(0, 2, 1)
            input_xyz_coordinates = point_cloud_xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.abstraction1(input_xyz_coordinates,
                                              input_points)  # Applying first set abstraction layer
        l2_xyz, l2_points = self.abstraction2(l1_xyz, l1_points)  # Applying second set abstraction layer
        # Feature Propagation layers
        l1_points = self.feature_propagation2(l1_xyz, l2_xyz, l1_points,
                                              l2_points)  # Applying second feature propagation layer
        l0_points = self.feature_propagation1(input_xyz_coordinates, l1_xyz, input_points, l1_points)


        feat1 = self.conv_layer1(l0_points)
        feat1 = self.batch_norm1(feat1)
        # FC layers
        x = F.relu(feat1)
        x = self.conv_layer2(x)  # Ensure that conv2 has 1 output channel for binary classification

        x = F.log_softmax(x, dim=1)  # Applying sigmoid activation for binary classification

        return x, l2_points
