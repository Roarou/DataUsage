import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


class PointCloudProcessor:
    """
    This class encapsulates methods for processing a point cloud.
    """

    def __init__(self, file_path):
        """
        Initialize the processor with the point cloud data from the file at file_path.
        Args:
        - file_path: str, path to the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")

        self.pcd = o3d.io.read_point_cloud(file_path)

    @staticmethod
    def load_point_cloud(file_path):
        """
        Load a Point Cloud from a file.
        Args:
        - file_path: str, path to the file.

        Returns:
        - open3d.geometry.PointCloud, loaded point cloud.
        """
        return o3d.io.read_point_cloud(file_path)

    def downsample_point_cloud(self, voxel_size=1):
        """
        Downsample the input point cloud.
        Args:
        - voxel_size: float, voxel size for the downsampling process.

        Modifies:
        - self.pcd: The point cloud data is replaced with its downsampled version.
        """
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def remove_statistical_outliers(self, nb_neighbors=20, std_ratio=1.0):
        """
        Remove statistical outliers from the point cloud.
        Args:
        - nb_neighbors: int, number of neighbors to analyze for each point.
        - std_ratio: float, standard deviation ratio.

        Modifies:
        - self.pcd: The point cloud data is replaced with its inlier version.
        """
        _, ind = self.pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        self.pcd = self.pcd.select_by_index(ind)

    def remove_radius_outliers(self, nb_points=16, radius=0.05):
        """
        Remove radius outliers from the point cloud.
        Args:
        - nb_points: int, number of points to analyze for each point.
        - radius: float, radius to consider for each point.

        Modifies:
        - self.pcd: The point cloud data is replaced with its inlier version.
        """
        _, ind = self.pcd.remove_radius_outlier(nb_points, radius)
        self.pcd = self.pcd.select_by_index(ind)

    def cluster_point_cloud(self, eps=0.02, min_points=10, print_clusters=False):
        """
        Cluster the point cloud using the DBSCAN algorithm.
        Args:
        - eps: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_points: int, the number of samples in a neighborhood for a point to be considered as a core point.
        - print_clusters: bool, flag to print the clusters.

        Returns:
        - labels: np.ndarray, array of labels.
        - largest_cluster_idx: np.ndarray, array of indices of the largest cluster.
        """
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        # Get unique labels and count of points in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Extract the label of the largest cluster
        largest_cluster_label = unique_labels[np.argmax(counts)]

        # Get indices of points in the largest cluster
        largest_cluster_idx = np.where(labels == largest_cluster_label)[0]

        if print_clusters:
            print(f"point cloud has {len(unique_labels)} clusters")
            for i in range(len(unique_labels)):
                print(f'Cluster {i} has {counts[i]} points')

        return labels, largest_cluster_idx

    def visualize_point_cloud(self):
        """
        Visualize the input point cloud.
        """
        o3d.visualization.draw_geometries([self.pcd])


def run(file_path, show_clusters=True):
    # Create an instance of the PointCloudProcessor
    pc_processor = PointCloudProcessor(file_path)

    # Apply various filters to the point cloud
    pc_processor.downsample_point_cloud(voxel_size=1)
    pc_processor.remove_statistical_outliers(nb_neighbors=20, std_ratio=1.0)
    pc_processor.remove_radius_outliers(nb_points=10, radius=5)

    # Perform clustering on the filtered point cloud data
    cluster_labels, idx_labels = pc_processor.cluster_point_cloud(eps=8, min_points=2, print_clusters=show_clusters)

    if show_clusters:
        # Assign a unique color to each cluster and visualize the result
        colors = plt.get_cmap("tab20")(cluster_labels / (max(cluster_labels) if max(cluster_labels) > 0 else 1))
        colors[cluster_labels < 0] = 0
        pcd_copy = copy.deepcopy(pc_processor.pcd)
        pcd_copy.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd_copy])

    # Keep only the points in the most common cluster
    pc_processor.pcd.points = o3d.utility.Vector3dVector(np.asarray(pc_processor.pcd.points)[idx_labels])
    pc_processor.pcd.colors = o3d.utility.Vector3dVector(np.asarray(pc_processor.pcd.colors)[idx_labels])

    # Visualize the final result
    pc_processor.visualize_point_cloud()


if __name__ == "__main__":
    # Path to your point cloud data file
    file_path = "test_1.pcd"
    run(file_path)
