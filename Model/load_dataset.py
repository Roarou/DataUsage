import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset


class PointcloudDataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        self.specimen_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('Specimen')])

    def __len__(self):
        return len(self.specimen_dirs)

    def __getitem__(self, idx):
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

                    # Extract input data (points)
                    input_data.append(np.array(input_pcd.points))

                    # Extract labels (binary values based on color)
                    labeled_pcd = np.array(groundtruth_pcd.colors)[:, 0]
                    binary_labels = (labeled_pcd >= 0.5).astype(np.float32)
                    labels.append(binary_labels)

        input_data = np.array(input_data)
        labels = np.array(labels)

        return input_data, labels
