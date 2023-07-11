import numpy as np
import open3d as o3d
from Camera.get_tf_cam import get_transformation
from Visualization.get_vertebras_displacement import visualize_displacement


class PoseTransformation:
    def __init__(self, path1, path2):
        """
        Initializes the PoseTransformation class.

        Args:
        path1 (str): The path for the first transformation.
        path2 (str): The path for the second transformation.
        """
        self.path1 = path1
        self.path2 = path2
        self.K1, self.E1 = get_transformation(path1)  # Get the transformation for the first path
        self.K2, self.E2 = get_transformation(path2)  # Get the transformation for the second path
        self.pcd1 = None
        self.pcd2 = None
        self.threshold = 3
        self.current_transformation = np.identity(4)
        self.geometries = []

    def read_point_cloud(self, file1, file2):
        """
        Reads point cloud data from files and applies uniform downsampling.

        Args:
        file1 (str): The first point cloud data file.
        file2 (str): The second point cloud data file.
        """
        self.pcd1 = o3d.io.read_point_cloud(file1).uniform_down_sample(5)
        self.pcd2 = o3d.io.read_point_cloud(file2).uniform_down_sample(5)

    def visualize_displacement(self, poses_0_file_path, poses_1_file_path):
        """
        Visualizes the displacement between two poses.

        Args:
        poses_0_file_path (str): The file path for the first set of poses.
        poses_1_file_path (str): The file path for the second set of poses.

        Returns:
        TF_1, TF_2: Transformation matrices for the first and second set of poses.
        """
        vertebrae, TF_1 = visualize_displacement(poses_0_file_path)
        _, TF_2 = visualize_displacement(poses_1_file_path)
        for vertebra in vertebrae:
            self.geometries.append(vertebra)
        return TF_1, TF_2

    def apply_transformation(self, TF_1, TF_2):
        """
        Applies a transformation to the point cloud data and visualizes the result.

        Args:
        TF_1, TF_2: Transformation matrices for the first and second set of poses.
        """
        self.pcd2 = self.pcd2.transform(np.linalg.inv(TF_2))
        self.pcd2 = self.pcd2.transform(TF_1)
        self.geometries.append(self.pcd1)
        self.geometries.append(self.pcd2)
        o3d.visualization.draw_geometries([self.pcd1, self.pcd2])

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

    def visualize(self):
        """Visualizes the point cloud data."""
        o3d.visualization.draw_geometries(self.geometries)


if __name__ == "__main__":
    # File and directory paths
    poses_0_file_path = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"
    poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"
    file1 = "test_0.pcd"
    file2 = "test_1.pcd"
    path1 = "E:/Ghazi/CamParams_0_31/SN10027879.conf"
    path2 = "E:/Ghazi/CamParams_0_31/SN10028650.conf"

    # Initialize the PoseTransformation object
    pose_transformer = PoseTransformation(path1, path2)

    # Visualize the displacement between poses
    TF_1, TF_2 = pose_transformer.visualize_displacement(poses_0_file_path, poses_1_file_path)

    # Read and downsample the point cloud data
    pose_transformer.read_point_cloud(file1, file2)

    # Apply the transformation to the point cloud data
    pose_transformer.apply_transformation(TF_1, TF_2)

    # Apply ICP registration
    reg_p2p = pose_transformer.registration_icp()

    # Print the transformation result
    print(reg_p2p.transformation)

    # Visualize the final result
    pose_transformer.visualize()


