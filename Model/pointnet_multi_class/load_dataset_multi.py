import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from collections import Counter

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
        (1, 0, 0): 0,  # L1
        (0, 1, 0): 1,  # L2
        (0, 0, 1): 2,  # L3
        (0, 1, 1): 3,  # L4
        (1, 0, 1): 4,  # L5
        (0, 0, 0): 5  # Non-spine
    }
    res = mapping.get(tuple(color), -1)
    if res == -1:
        print(f'test {color}')
    return  res # -1 for any unexpected colors


class PointcloudDataset(Dataset):
    def __init__(self, base_path=r'D:\Ghazi\Pointcloud', validation_path=r'D:\Ghazi\PointcloudVal1', test_path=r'D:\Ghazi\PointcloudTest',
                 split='train', test_size=0.2, val_size=0.2,
                 random_state=42, num_points=20000, sample=True):
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
        self.file_val = os.listdir(validation_path)
        self.file_test = os.listdir(test_path)
        self.target_num_points = num_points
        self.sample = sample

        # Split the dataset into train, validation, and test sets
        # train_dirs, test_dirs = train_test_split(self.file_list, test_size=test_size, random_state=random_state)
        # train_dirs, val_dirs = train_test_split(train_dirs, test_size=val_size, random_state=random_state)
        train_dirs = self.file_list
        # test_dirs, val_dirs = train_test_split(self.file_test_val, test_size=val_size, random_state=random_state)
        val_dirs = self.file_val
        test_dirs = self.file_test
        if split == 'train':
            self.specimen_dirs = train_dirs
        elif split == 'val':
            self.specimen_dirs = val_dirs
        elif split == 'test':
            self.specimen_dirs = test_dirs
        else:
            raise ValueError("Invalid split mode. Use 'train', 'val', or 'test'.")
        """
        labelweights = np.zeros(6)
        for file in tqdm(self.file_list, total=len(self.file_list)):
            file = os.path.join(base_path,file)
            pcd = o3d.io.read_point_cloud(file)
            colors = np.asarray(pcd.colors)
            labels = np.array([map_color_to_label(c) for c in colors])
            L1 = np.sum(np.all(labels == 0, axis=0))
            L2 = np.sum(np.all(labels == 1, axis=0))
            L3 = np.sum(np.all(labels == 2, axis=0))
            L4 = np.sum(np.all(labels == 3, axis=0))
            L5 = np.sum(np.all(labels == 4, axis=0))
            NS = np.sum(np.all(labels == 5, axis=0))
            tmp = [L1, L2, L3, L4, L5, NS]
            labelweights+= tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        """

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
        # print(filename)
        # Normalize input data
        input = np.asarray(input_pcd.points)
        colors = np.asarray(input_pcd.colors)
        labels = colors
        if self.sample:
            if len(input) > self.target_num_points:
                sampled_indices = np.random.choice(len(input), self.target_num_points, replace=False)
                input = input[sampled_indices]
                labels = colors[sampled_indices]
        labels = np.array([map_color_to_label(c) for c in labels])
        L1 = np.sum(np.all(labels == 1))
        frequency = Counter(labels)
        # print(frequency[5])
        if L1 > 0:
            print(f"Vector appears {L1} times in a list of length {len(labels)}.")
            print(f"Warning: Unexpected color detected in file: {filename}")
        normalized_input = normalize_point_cloud(input, file_path)

        input_data = torch.tensor(normalized_input, dtype=torch.float16)
        labels_tensor = torch.tensor(labels, dtype=torch.float16)
        return input_data, labels_tensor
