import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, sample_points, group_points):
        super().__init__()
        self.sample_points = sample_points
        self.group_points = group_points
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Randomly sample points
        idx = torch.randperm(x.size(1))[:self.sample_points]
        sampled_x = x[:, idx, :]

        # Find K nearest neighbors
        neighbors = knn(x, sampled_x, self.group_points, include_self=True)
        neighbors = neighbors[1]

        # Group neighbors
        groups = x[:, neighbors, :]

        # Apply MLPs
        features = self.mlp(groups)

        # Max Pooling
        features, _ = torch.max(features, dim=2)

        return features, sampled_x


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
        dists, idx = knn(sampled_x, x, 3, include_self=False)
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

        return F.log_softmax(x, dim=1)
