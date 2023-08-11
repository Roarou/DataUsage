import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class PointcloudDataset(Dataset):
    def __init__(self, base_path=r'G:\SpineDepth', split='train', test_size=0.1, val_size=0.1, random_state=42):
        """
        Custom dataset class for loading and managing point cloud data.

        Args:
            base_path (str): Path to the root directory of the dataset.
            split (str): Split mode: 'train', 'val', or 'test'.
            test_size (float): Proportion of data to be used for testing.
            val_size (float): Proportion of data to be used for validation.
            random_state (int): Random seed for reproducibility.
        """
        self.base_path = base_path
        self.specimen_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('Specimen')])

        # Split the dataset into train, validation, and test sets
        train_dirs, test_dirs = train_test_split(self.specimen_dirs, test_size=test_size, random_state=random_state)
        train_dirs, val_dirs = train_test_split(train_dirs, test_size=val_size, random_state=random_state)

        if split == 'train':
            self.specimen_dirs = train_dirs
        elif split == 'val':
            self.specimen_dirs = val_dirs
        elif split == 'test':
            self.specimen_dirs = test_dirs
        else:
            raise ValueError("Invalid split mode. Use 'train', 'val', or 'test'.")

    def __len__(self):
        """Returns the number of specimens in the dataset."""
        return len(self.specimen_dirs)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a specimen's data and labels.

        Args:
            idx (int): Index of the specimen to retrieve.

        Returns:
            input_data (np.ndarray): Normalized point cloud data.
            labels (np.ndarray): Binary labels based on color values.
        """
        specimen_dir = self.specimen_dirs[idx]
        recordings_dir = os.path.join(self.base_path, specimen_dir, 'Recordings')

        input_data = []
        labels = []

        for recording_num in range(1, 41):
            recording_dir = os.path.join(recordings_dir, f'Recording_{recording_num}')
            for video_num in range(2):  # Iterate over 'Video_0' and 'Video_1'
                video_dir = os.path.join(recording_dir, f'Video_{video_num}')
                groundtruth_dir = os.path.join(video_dir, 'Groundtruth')
                input_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.pcd')])

                for input_file in input_files:
                    input_pcd = o3d.io.read_point_cloud(os.path.join(video_dir, input_file))
                    groundtruth_pcd = o3d.io.read_point_cloud(
                        os.path.join(groundtruth_dir, input_file.replace('.pcd', '_GT.pcd')))

                    # Normalize input data
                    normalized_input = self.normalize_point_cloud(np.array(input_pcd.points))

                    # Extract labels (binary values based on color)
                    labeled_pcd = np.array(groundtruth_pcd.colors)[:, 0]
                    binary_labels = (labeled_pcd >= 0.5).astype(np.float32)

                    input_data.append(normalized_input)
                    labels.append(binary_labels)

        input_data = np.array(input_data)
        labels = np.array(labels)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return input_data, labels

    def normalize_point_cloud(self, points):
        """
        Normalizes point cloud data.

        Args:
            points (np.ndarray): Point cloud data.

        Returns:
            np.ndarray: Normalized point cloud data.
        """
        centroid = np.mean(points, axis=0)
        points -= centroid  # center
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance  # scale
        return points
