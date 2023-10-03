import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm


def normalize_point_cloud(points, filepath=None):
    if np.any(np.isnan(points)):
        raise ValueError(f"Input data contains NaN values. Reading: {filepath}")

    centroid = np.mean(points, axis=0)

    if np.any(np.isnan(centroid)):
        raise ValueError(f"Centroid calculation resulted in NaN values. Reading: {filepath}")

    points -= centroid  # center
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))

    if furthest_distance == 0.0:
        raise ValueError("Furthest distance is zero, which can cause division by zero.")

    points /= (furthest_distance + 1e-7)  # scale
    return points


def map_color_to_label(color):
    mapping = {
        (1, 0, 0): 1,  # L1
        (0, 1, 0): 2,  # L2
        (0, 0, 1): 3,  # L3
        (1, 1, 0): 4,  # L4
        (1, 0, 1): 5,  # L5
        (0, 0, 0): 0  # Non-spine
    }
    return mapping.get(tuple(color), -1)  # -1 for any unexpected colors


class PointcloudDataset(Dataset):
    def __init__(self, base_path=r'L:\groundtruth_labeled', test_validation_path=r'L:\groundtruth',
                 split='train', test_size=0.2, val_size=0.2,
                 random_state=42, num_points=340000):
        """
        Custom dataset class for loading and managing point cloud data.

        Args:
            base_path (str): Path to the root directory of the dataset.
            split (str): Split mode: 'train', 'val', or 'test'.
            test_size (float): Proportion of data to be used for testing.
            val_size (float): Proportion of data to be used for validation.
            random_state (int): Random seed for reproducibility.
        """
        self.root_dir = base_path
        self.file_list = os.listdir(self.root_dir)
        self.file_test_val = os.listdir(test_validation_path)
        self.target_num_points = num_points

        # Split the dataset into train, validation, and test sets
        # train_dirs, test_dirs = train_test_split(self.file_list, test_size=test_size, random_state=random_state)
        # train_dirs, val_dirs = train_test_split(train_dirs, test_size=val_size, random_state=random_state)
        train_dirs = self.file_list
        test_dirs, val_dirs = train_test_split(self.file_test_val, test_size=val_size, random_state=random_state)
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
        filename = self.file_list[idx]
        file_path = os.path.join(self.root_dir, filename)

        input_pcd = o3d.io.read_point_cloud(file_path, remove_nan_points=True,
                                            remove_infinite_points=True)

        # Normalize input data
        input = np.asarray(input_pcd.points)
        colors = np.asarray(input_pcd.colors)
        labels = np.array([map_color_to_label(c) for c in colors])

        if len(input) > self.target_num_points:
            sampled_indices = np.random.choice(len(input), self.target_num_points, replace=False)
            input = input[sampled_indices]
            labels = labels[sampled_indices]

        normalized_input = normalize_point_cloud(input, file_path)

        input_data = torch.tensor(normalized_input, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return input_data, labels_tensor
