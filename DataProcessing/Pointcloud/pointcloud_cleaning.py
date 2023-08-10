import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KDTree
from DataProcessing.Pointcloud.crop_pointcloud import process_point_cloud


class PointCloudProcessor:
    """
    This class encapsulates methods for processing a point cloud.
    """

    def __init__(self, file_path: str, idx):
        """
        Initialize the processor with the point cloud data from the file at file_path.
        Args:
        - file_path: str, path to the file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
        self.idx = idx
        self.file_path = file_path
        self.GT = o3d.io.read_point_cloud(file_path)
        self.pcd = self.GT.select_by_index(idx)
        self.indice = []
    @staticmethod
    def load_point_cloud(file_path: str, downsampling_factor=2):
        """
        Load a Point Cloud from a file.
        Args:
        - file_path: str, path to the file.

        Returns:
        - open3d.geometry.PointCloud, loaded point cloud.
        """
        return o3d.io.read_point_cloud(file_path).uniform_down_sample(downsampling_factor)

    def downsample_point_cloud(self, voxel_size=1):
        """
        Downsample the input point cloud.
        Args:
        - voxel_size: float, voxel size for the downsampling process.

        Modifies:
        - self.pcd: The point cloud data is replaced with its downsampled version.
        """
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def update_colors(self, indices, save=False):
        """
        Update the colors of the point cloud, setting the specified indices to their original colors and the rest to
        black.

        Args:
        - indices: list or array, indices of the points to keep their original colors.
        """
        black_color = np.zeros_like(np.asarray(self.pcd.colors))
        if save:
            white_color = np.ones_like(np.asarray(self.pcd.colors)) / 2
            black_color[indices] = white_color[indices]
            colors = np.asarray(self.GT.colors)
            colors[self.idx] = black_color
            self.GT.colors = o3d.utility.Vector3dVector(colors)
            self.pcd = self.GT
        else:
            self.indice = indices
            original_color = np.asarray(self.pcd.colors)
            black_color[indices] = original_color[indices]
            self.pcd.colors = o3d.utility.Vector3dVector(black_color)

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
        self.update_colors(ind)

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
        self.update_colors(ind)

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

        # Keep only the points in the most common cluster
        self.update_colors(indices=largest_cluster_idx)
        if print_clusters:
            print(f"point cloud has {len(unique_labels)} clusters")
            for i in range(len(unique_labels)):
                print(f'Cluster {i} has {counts[i]} points')

        return labels, largest_cluster_idx

    def visualize_point_cloud(self):
        """
        Visualize the input point cloud.
        """
        # self.save_point_cloud()
        o3d.visualization.draw_geometries([self.pcd])

    def save_point_cloud(self):
        """
        Save the processed point cloud data back to the original file.
        """
        # Check if the file already exists
        self.update_colors(indices=self.indice, save=True)  # Use self.indice
        if os.path.exists(self.file_path):
            # If so, remove it
            os.remove(self.file_path)
        o3d.io.write_point_cloud(self.file_path, self.pcd)
        print(f"Saved the processed point cloud to {self.file_path}")

    def filter_by_color(self, color_diff_threshold=0.2):
        """
        Filters the point cloud by color. Keeps only those points that have a color similar to the average color.

        Args:
        - color_diff_threshold: float, maximum allowed difference from the average color.
        """
        # Calculate the average color of the point cloud
        avg_color = np.asarray(self.pcd.colors).mean(axis=0)

        # Calculate the color difference for each point
        color_diff = np.linalg.norm(np.asarray(self.pcd.colors) - avg_color, axis=1)

        # Filter out the points that have a color difference greater than the threshold
        inliers = np.where(color_diff < color_diff_threshold)[0]

        self.update_colors(inliers)

    def color_filtering(self, n_passes=3, init_threshold=0.5, threshold_decay=0.8):
        """
        Apply color-based filtering in multiple passes with a decaying threshold.
        Args:
        - n_passes: int, number of passes.
        - init_threshold: float, initial threshold.
        - threshold_decay: float, decay factor for the threshold.
        """
        for _ in range(n_passes):
            self.filter_by_color(color_diff_threshold=init_threshold)
            init_threshold *= threshold_decay

    def remove_outliers_with_isolation_forest(self):
        """
        Remove outliers using the Isolation Forest algorithm.
        Isolation Forest is a machine learning algorithm for anomaly detection. It's based on the concept of
        isolating anomalies, instead of the most common data-mining practice of profiling regular instances.
        """
        # Convert the points and colors to a NumPy array
        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        # Fit the model
        clf = IsolationForest(contamination=0.05)
        preds = clf.fit_predict(points)

        # Get the inliers (labeled as 1)
        inlier_indices = np.where(preds == 1)[0]
        # inliers = points[preds == 1]
        # inlier_colors = colors[preds == 1]

        self.update_colors(inlier_indices)
        # Update the point cloud
        # self.pcd.points = o3d.utility.Vector3dVector(inliers)
        # self.pcd.colors = o3d.utility.Vector3dVector(inlier_colors)

    def upsample_point_cloud(self, n_neighbors=5):
        """
        Upsample the point cloud using a KDTree for nearest neighbors computation.
        Args:
        - n_neighbors: int, the number of neighbors to use for the upsampling.

        Modifies:
        - self.pcd: The point cloud data is replaced with its upsampled version.
        """
        # Create a KDTree from the point cloud data
        tree = KDTree(np.asarray(self.pcd.points), leaf_size=2)

        # Interpolate each point with its nearest neighbors
        interpolated_points = []
        interpolated_colors = []
        for point, color in zip(self.pcd.points, self.pcd.colors):
            _, indices = tree.query([point], k=n_neighbors)
            interpolated_points.extend(self.pcd.points[i] for i in indices[0])
            interpolated_colors.extend([color] * n_neighbors)  # Assign the same color to each new point

        # Update the point cloud data
        self.pcd.points = o3d.utility.Vector3dVector(np.array(interpolated_points))
        self.pcd.colors = o3d.utility.Vector3dVector(np.array(interpolated_colors))  # Update the colors


def clean(file_path, idx, show_clusters=False, factor=8, rad=5, reconstruction=False, noisy_data=False):
    # Create an instance of the PointCloudProcessor
    pc_processor = PointCloudProcessor(file_path, idx)
    if noisy_data:
        # Apply various filters to the point cloud
        # seems to be working much better, without it
        pc_processor.downsample_point_cloud(voxel_size=1)
        pc_processor.remove_statistical_outliers(nb_neighbors=20, std_ratio=1.0)
        pc_processor.remove_radius_outliers(nb_points=10, radius=rad)
    # Perform clustering on the filtered point cloud data
    cluster_labels, idx_labels = pc_processor.cluster_point_cloud(eps=factor, min_points=2,
                                                                  print_clusters=show_clusters)

    if show_clusters:
        # Assign a unique color to each cluster and visualize the result
        colors = plt.get_cmap("tab20")(cluster_labels / (max(cluster_labels) if max(cluster_labels) > 0 else 1))
        colors[cluster_labels < 0] = 0
        pcd_copy = copy.deepcopy(pc_processor.pcd)
        pcd_copy.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd_copy])

    # Apply multi-pass color filtering
    pc_processor.remove_outliers_with_isolation_forest()

    if reconstruction:
        pc_processor.upsample_point_cloud(n_neighbors=3)
        pc_processor.color_filtering(n_passes=3, init_threshold=0.9, threshold_decay=0.8)

    # Visualize the final result
    # pc_processor.visualize_point_cloud()

    # Save the processed point cloud back to the original file

    pc_processor.save_point_cloud()



if __name__ == "__main__":
    # Path to your point cloud data file
    input_path = "G:\SpineDepth\Specimen_1\Recordings\Recording0\pointcloud\Video_0\Pointcloud_0.pcd"
    pose_path = "G:\SpineDepth\Specimen_1\Recordings\Recording0"
    idx = process_point_cloud(input_path, pose_path)
    file_path = "G:\SpineDepth\Specimen_1\Recordings\Recording0\pointcloud\Video_0\Pointcloud_0_GT.pcd"
    clean(file_path, idx)
