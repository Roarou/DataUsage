import numpy as np
import open3d as o3d
from Camera.get_tf_cam import get_transformation
from DataProcessing.get_vertebras_displacement import visualize_displacements
import os

colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (0.5, 0.5, 0.5),  # Gray
]


class PoseTransformation:
    def __init__(self, frame, specimen):
        """
        Initializes the PoseTransformation class.

        Args:
        path1 (str): The path for the first transformation.
        path2 (str): The path for the second transformation.
        """
        self.pcd1 = None
        self.pcd2 = None
        self.combined_pcd = None
        self.threshold = 10
        self.current_transformation = np.identity(4)
        self.geometries = []
        self.specimen = specimen
        self.frame = frame
    def read_point_cloud(self, file1, file2):
        """
        Reads point cloud data from files and applies uniform downsampling.

        Args:
        file1 (str): The first point cloud data file.
        file2 (str): The second point cloud data file.
        downsampling_factor (int): The downsampling factor to apply to the point clouds.
        """
        self.pcd1 = o3d.io.read_point_cloud(file1)
        self.pcd2 = o3d.io.read_point_cloud(file2)

    def visualize_displacement(self, poses_0_file_path, poses_1_file_path):
        """
        Visualizes the displacement between two poses.

        Args:
        poses_0_file_path (str): The file path for the first set of poses.
        poses_1_file_path (str): The file path for the second set of poses.

        Returns:
        TF_1, TF_2: Transformation matrices for the first and second set of poses.
        """
        vertebrae, TF_1 = visualize_displacements(poses_0_file_path, self.frame, self.specimen)
        _, TF_2 = visualize_displacements(poses_1_file_path)
        TF_1 = TF_1[0]
        TF_2 = TF_2[0]
        for i, vertebra in enumerate(vertebrae):
            bounding_box = vertebra.get_oriented_bounding_box()
            bounding_box.color = colors[i]
            self.geometries.append(vertebra)
            self.geometries.append(bounding_box)
        return TF_1, TF_2

    def apply_transformation(self, TF_1, TF_2):
        """
        Applies a transformation to the point cloud data and visualizes the result.

        Args:
        TF_1, TF_2: Transformation matrices for the first and second set of poses.
        """
        self.pcd2 = self.pcd2.transform(np.linalg.inv(TF_2))
        self.pcd2 = self.pcd2.transform(TF_1)

        bounding_box2 = self.pcd1.get_oriented_bounding_box()
        bounding_box2.color = colors[6]

    def registration_icp(self):
        """
        Applies the Iterative Closest Point (ICP) registration to the point cloud data.

        Returns:
        reg_p2p: The ICP registration result.
        """
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.pcd2, self.pcd1, self.threshold, self.current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
        self.pcd2.transform(reg_p2p.transformation)
        return reg_p2p

    def visualize(self, filepath):
        """Visualizes the point cloud data."""
        self.combined_pcd = o3d.io.read_point_cloud(filepath)
        bounding_box = self.combined_pcd.get_oriented_bounding_box()
        bounding_box.color = colors[5]
        self.geometries.append(self.combined_pcd)
        self.geometries.append(bounding_box)
        o3d.visualization.draw_geometries(self.geometries)

    def save_point_clouds_together(self, file_path: str, point_cloud1, point_cloud2):
        """
        Combines two point clouds and saves the result into a file.

        Args:
        file_path (str): The output file name.
        point_cloud1 (o3d.geometry.PointCloud): The first point cloud data.
        point_cloud2 (o3d.geometry.PointCloud): The second point cloud data.
        """
        # Check if the file already exists
        if os.path.exists(file_path):
            # If so, remove it
            os.remove(file_path)
        self.combined_pcd = point_cloud1 + point_cloud2
        o3d.io.write_point_cloud(file_path, self.combined_pcd)

