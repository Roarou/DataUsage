import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm


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
        for recording_num in tqdm(range(1, 40), desc="Recordings", leave=False):
            recording_dir = os.path.join(recordings_dir, f'Recording{recording_num}')
            recording_dir = os.path.join(recording_dir, 'pointcloud')

            for video_num in range(2):  # Iterate over 'Video_0' and 'Video_1'
                video_dir = os.path.join(recording_dir, f'Video_{video_num}')
                groundtruth_dir = os.path.join(video_dir, 'Groundtruth')
                input_files = sorted([f for f in os.listdir(groundtruth_dir) if f.endswith('.pcd')])

                for input_file in tqdm(input_files, desc="Files", leave=False):
                    file_path = os.path.join(groundtruth_dir, input_file)
                    input_pcd = o3d.io.read_point_cloud(file_path, remove_nan_points=True,
                                                        remove_infinite_points=True)

                    # Normalize input data
                    input = np.asarray(input_pcd.points)
                    normalized_input = self.normalize_point_cloud(input)
                    # Extract labels (binary values based on color)
                    labeled_pcd = np.asarray(input_pcd.colors)[:, 0]
                    binary_labels = (labeled_pcd >= 0.5).astype(np.float32)
                    input_data.append(normalized_input)
                    labels.append(binary_labels)

        input_data = np.array(input_data)
        labels = np.array(labels)
        print(labels)
        input_data = torch.tensor(input_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return input_data, labels

    def normalize_point_cloud(self, points):
        if np.any(np.isnan(points)):
            raise ValueError("Input data contains NaN values.")

        centroid = np.mean(points, axis=0)

        if np.any(np.isnan(centroid)):
            raise ValueError("Centroid calculation resulted in NaN values.")

        points -= centroid  # center
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))

        if furthest_distance == 0.0:
            raise ValueError("Furthest distance is zero, which can cause division by zero.")

        points /= furthest_distance  # scale
        return points
