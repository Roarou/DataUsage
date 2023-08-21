import torch
import torch.nn as nn
from torch_geometric.nn import knn


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, sample_points, group_points):
        super().__init__()
        self.sample_points = sample_points
        self.group_points = group_points
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Get batch size, number of points, and number of features
        batch_size, num_points, num_features = x.size()

        # List to collect the output for each point cloud in the batch
        features_list = []
        sampled_x_list = []

        # Process each point cloud in the batch
        for i in range(batch_size):
            # Inside the loop, make sure both point_cloud and sampled_x are 2D tensors
            print(i)
            point_cloud = x[i, :, :]

            idx = torch.randperm(num_points)[:self.sample_points]
            sampled_x = point_cloud[idx, :]

            # Ensure that both tensors are 2D before passing them to the KNN function
            neighbors = knn(point_cloud, sampled_x, self.group_points)
            neighbors = neighbors[1]

            # Group neighbors
            groups = point_cloud[neighbors, :]
            groups = groups.permute(1, 0)

            # Apply MLPs
            features = self.mlp(groups)

            # Max Pooling
            features, _ = torch.max(features, dim=1)

            # Append to the list
            features_list.append(features)
            sampled_x_list.append(sampled_x)


        # Stack results to get the batched output

        # Add the missing dimension back (assuming it should be added as the third dimension)
        features_batch = torch.cat(features_list, dim=0).view(batch_size, -1, self.sample_points)
        sampled_x_batch = torch.cat(sampled_x_list, dim=0).view(batch_size, -1, self.sample_points)

        print(features_batch.shape)
        print(sampled_x_batch.shape)
        return features_batch, sampled_x_batch

class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1)
        )

    def forward(self, x, sampled_x, features):
        # Interpolate features
        dists, idx = knn(sampled_x, x, 3)
        interpolated_features = torch.sum(features[:, idx, :], dim=2) / 3

        # Concatenate with input features
        x = torch.cat([x, interpolated_features], dim=-1)

        # Apply MLP
        x = self.mlp(x)

        return x


class SpineSegmentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = SetAbstraction(3, 64, 512, 32)
        self.sa2 = SetAbstraction(64, 128, 128, 32)
        self.fp1 = FeaturePropagation(128 + 64, 64)
        self.fp2 = FeaturePropagation(64 + 3, 32)
        self.fc = nn.Linear(32, 2)  # Binary classification

    def forward(self, x):
        x1, s1 = self.sa1(x)
        x2, s2 = self.sa2(x1)

        x2 = self.fp1(x1, s1, x2)
        x2 = self.fp2(x, s2, x2)

        x = self.fc(x2)

        return torch.sigmoid(x)
