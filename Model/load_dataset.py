import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.specimen_folders = [f'Specimen_{i}' for i in range(1, 11)]
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for specimen_folder in self.specimen_folders:
            input_folder = os.path.join(self.base_dir, specimen_folder, 'Input')
            file_list.extend([os.path.join(input_folder, filename) for filename in os.listdir(input_folder)])
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load point cloud using Open3D
        pcd = o3d.io.read_point_cloud(file_path)

        # Convert points to numpy array
        points = np.array(pcd.points)

        # Load corresponding labeled point cloud
        specimen_folder = os.path.basename(os.path.dirname(file_path))
        groundtruth_folder = os.path.join(self.base_dir, specimen_folder, 'Groundtruth')
        labeled_file_path = os.path.join(groundtruth_folder, os.path.basename(file_path))
        labeled_pcd = o3d.io.read_point_cloud(labeled_file_path)

        # Convert colors to binary labels based on threshold
        binary_labels = (np.array(labeled_pcd.colors)[:, 0] >= 0.5).astype(np.float32)

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(points, dtype=torch.float32)
        label_tensor = torch.tensor(binary_labels, dtype=torch.float32)

        return {'input': input_tensor, 'label': label_tensor}