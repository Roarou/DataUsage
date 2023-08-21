import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation

class SpineSegmentationNet(nn.Module):
    def __init__(self):
        super(SpineSegmentationNet, self).__init__()

        # Set abstraction layers for hierarchical point cloud processing
        self.set_abstraction_layer1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.set_abstraction_layer2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.set_abstraction_layer3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.set_abstraction_layer4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])

        # Feature propagation layers
        self.feature_propagation_layer4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.feature_propagation_layer3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.feature_propagation_layer2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.feature_propagation_layer1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # Convolution layers
        self.conv_layer1 = nn.Conv1d(128, 128, 1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout_layer = nn.Dropout(0.5)
        self.conv_layer2 = nn.Conv1d(128, 1, 1) # Segmentation layer for binary classification per point
        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, point_cloud_xyz):
        # Input layer
        input_points = point_cloud_xyz
        input_xyz_coordinates = point_cloud_xyz[:,:3,:]

        # Set abstraction layers
        l1_xyz, l1_points = self.set_abstraction_layer1(input_xyz_coordinates, input_points)
        l2_xyz, l2_points = self.set_abstraction_layer2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.set_abstraction_layer3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.set_abstraction_layer4(l3_xyz, l3_points)

        # Feature propagation layers
        l3_points = self.feature_propagation_layer4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.feature_propagation_layer3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.feature_propagation_layer2(l1_xyz, l2_xyz, l1_points, l2_points)
        input_points = self.feature_propagation_layer1(input_xyz_coordinates, l1_xyz, None, l1_points)

        # Convolution layers
        x = self.dropout_layer(F.relu(self.batch_norm1(self.conv_layer1(input_points))))
        x = self.conv_layer2(x)
        segmentation_mask = self.sigmoid_activation(x)
        segmentation_mask = segmentation_mask.permute(0, 2, 1)

        return segmentation_mask, l4_points