import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

def filter_data(file, voxel_size=1, std_ratio=1.0, nb_points=10, radius=5):
    """
    Filters the point cloud data by performing voxel grid downsampling, statistical outlier removal, and radius
    outlier removal.

    Parameters:
    - file: The path to the PCD file.
    - voxel_size: The voxel size for the voxel grid downsampling.
    - std_ratio: The standard deviation ratio for the statistical outlier removal.
    - nb_points: The number of points for the radius outlier removal.
    - radius: The radius for the radius outlier removal.

    Returns:
    - The filtered point cloud.
    """
    # Load the point cloud from the file
    pcd = o3d.io.read_point_cloud(file)

    # Perform voxel grid downsampling
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # Perform statistical outlier removal
    cl, ind = pcd_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    pcd_statistical_outlier_removed = pcd_downsampled.select_by_index(ind)

    # Perform radius outlier removal
    cl, ind = pcd_statistical_outlier_removed.remove_radius_outlier(nb_points=nb_points, radius=radius)
    pcd_radius_outlier_removed = pcd_statistical_outlier_removed.select_by_index(ind)

    return pcd_radius_outlier_removed


def cluster_point_cloud(pcd, eps=0.02, min_points=10, print_clusters=False):
    """
    Performs DBSCAN clustering on the point cloud data.

    Parameters:
    - pcd: The point cloud data.
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_points: The number of points in a neighborhood for a point to be considered as a core point.
    - print_clusters: A boolean indicating whether to print out information about each cluster.

    Returns:
    - The labels of the clusters.
    """
    # Perform DBSCAN clustering
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # Print number of clusters found
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # Find the cluster with the most points
    unique_labels, counts = np.unique(labels, return_counts=True)
    most_common_label = unique_labels[np.argmax(counts)]

    if print_clusters:
        # Print points for each cluster
        for i in range(max_label + 1):
            print(f'Cluster {i} has {np.sum(labels == i)} points')

    idx_label = labels == most_common_label
    # Return the different labels and the points from the most common cluster
    return labels, idx_label


# Path to your point cloud data file
file_path = "test_1.pcd"

# Filter the point cloud data
pcd_filtered = filter_data(file_path)
print(len(pcd_filtered.points))
# Perform clustering on the filtered point cloud data
show_clusters = True
cluster_labels, idx_labels = cluster_point_cloud(pcd_filtered, eps=8, min_points=2, print_clusters=show_clusters)
if show_clusters:
    # Assign a unique color to each cluster and visualize the result
    colors = plt.get_cmap("tab20")(cluster_labels / (max(cluster_labels) if max(cluster_labels) > 0 else 1))
    colors[cluster_labels < 0] = 0
    vis = copy.deepcopy(pcd_filtered)
    vis.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([vis])
# Keep only the points in the most common cluster
pcd_filtered.points = o3d.utility.Vector3dVector(np.asarray(pcd_filtered.points)[idx_labels])
pcd_filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd_filtered.colors)[idx_labels])

# Visualize the result
o3d.visualization.draw_geometries([pcd_filtered])

