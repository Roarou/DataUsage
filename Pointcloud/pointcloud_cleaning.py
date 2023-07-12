import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from scipy.spatial import KDTree

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
        self.file_path = file_path
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

    def save_point_cloud(self):
        """
        Save the processed point cloud data back to the original file.
        """
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

        self.pcd = self.pcd.select_by_index(inliers)

    def estimate_normals(self, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
        """
        Estimate the normals of the point cloud.
        Args:
        - search_param: o3d.geometry.KDTreeSearchParamHybrid, search parameters for KDTree.

        Modifies:
        - self.pcd: The point cloud data is updated with normals.
        """
        self.pcd.estimate_normals(search_param)
    def reconstruct_surface_and_colorize(self, depth=8):
        """
        Reconstruct the 3D surface from the point cloud and colorize it based on the nearest neighbors.
        Args:
        - depth: int, depth for the Poisson surface reconstruction.

        Modifies:
        - self.mesh: The reconstructed and colorized mesh.
        """
        # Estimate normals
        self.pcd.estimate_normals()

        # perform Poisson surface reconstruction
        self.mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=depth)

        # build a kdtree from the point cloud
        kdtree = KDTree(np.asarray(self.pcd.points))

        # for each vertex in the mesh, find its nearest neighbor in the point cloud
        distances, indices = kdtree.query(np.asarray(self.mesh.vertices))

        # assign the color of the nearest neighbor to the vertex
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(self.pcd.colors)[indices])

    def visualize_mesh(self):
        """
        Visualize the reconstructed and colorized mesh.
        """
        o3d.visualization.draw_geometries([self.mesh])


def clean(file_path, show_clusters=True, factor=8, rad=5, reconstruction=False):
    # Create an instance of the PointCloudProcessor
    pc_processor = PointCloudProcessor(file_path)

    # Apply various filters to the point cloud
    pc_processor.downsample_point_cloud(voxel_size=1)
    pc_processor.remove_statistical_outliers(nb_neighbors=20, std_ratio=1.0)
    pc_processor.remove_radius_outliers(nb_points=10, radius=rad)
    pc_processor.filter_by_color(color_diff_threshold=0.35)
    # Perform clustering on the filtered point cloud data
    cluster_labels, idx_labels = pc_processor.cluster_point_cloud(eps=factor, min_points=2, print_clusters=show_clusters)

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
    if reconstruction:
        # After clustering and visualizing the point cloud, reconstruct the surface and colorize it.
        pc_processor.reconstruct_surface_and_colorize(depth=10)

        # Visualize the reconstructed and colorized mesh.
        pc_processor.visualize_mesh()


    # Save the processed point cloud back to the original file
    #pc_processor.save_point_cloud()


if __name__ == "__main__":
    # Path to your point cloud data file
    file_path = "../test_1.pcd"
    clean(file_path)
