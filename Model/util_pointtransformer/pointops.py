from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) # (m, nsample, 3)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


import torch


import numpy as np
import torch

def knnquery(m, nsample, xyz, new_xyz, offset, new_offset):
    """
    K-nearest neighbors query in Python.

    Args:
        m (int): Number of query points.
        nsample (int): Number of neighbors to find.
        xyz (torch.Tensor): Source point cloud coordinates (n, 3).
        new_xyz (torch.Tensor): Query point cloud coordinates (m, 3).
        offset (torch.Tensor): Batch offsets for source cloud.
        new_offset (torch.Tensor): Batch offsets for query cloud.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Indices and squared distances of the k-nearest neighbors.
    """
    idx = torch.zeros(m, nsample, dtype=torch.int32)
    dist2 = torch.zeros(m, nsample, dtype=torch.float32)

    for pt_idx in range(m):
        start = 0 if pt_idx == 0 else offset[pt_idx - 1]
        end = offset[pt_idx]

        new_point = new_xyz[pt_idx]
        best_dist = np.full(nsample, 1e10)
        best_idx = np.full(nsample, start)

        for i in range(start, end):
            point = xyz[i]
            d2 = torch.sum((new_point - point) ** 2)

            if d2 < best_dist[0]:
                best_dist[0] = d2
                best_idx[0] = i
                reheap(best_dist, best_idx, nsample)

        heap_sort(best_dist, best_idx, nsample)

        idx[pt_idx] = torch.tensor(best_idx)
        dist2[pt_idx] = torch.tensor(best_dist)

    return idx, torch.sqrt(dist2)

def reheap(dist, idx, k):
    """ Helper function to reheap the distances and indices. """
    root = 0
    while True:
        child = 2 * root + 1
        if child >= k:
            break

        if child + 1 < k and dist[child + 1] > dist[child]:
            child += 1
        if dist[root] > dist[child]:
            break

        dist[root], dist[child] = dist[child], dist[root]
        idx[root], idx[child] = idx[child], idx[root]
        root = child

def heap_sort(dist, idx, k):
    """ Helper function to perform heap sort. """
    for i in range(k - 1, 0, -1):
        dist[0], dist[i] = dist[i], dist[0]
        idx[0], idx[i] = idx[i], idx[0]
        reheap(dist, idx, i)

# Example usage
m = 10  # Number of query points
nsample = 5  # Number of neighbors to find
xyz = torch.rand(100, 3)  # Random source point cloud
new_xyz = torch.rand(m, 3)  # Random query points
offset = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # Example offsets
new_offset = offset  # For simplicity, using the same offsets for query points

idx, dist2 = knnquery(m, nsample, xyz, new_xyz, offset, new_offset)

