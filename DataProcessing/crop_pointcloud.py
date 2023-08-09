import open3d as o3d
import numpy as np
import os
from DataProcessing.extract_tf_matrix_groundtruth import extract_transformation_matrices
from DataProcessing.get_vertebras_displacement import visualize_displacements


def process_point_cloud(input_path: str, path_pose):
    """
    This function reads a point cloud from a file and processes it by applying the following steps:
    1. Defines a bounding box within the point cloud.
    2. Identifies the indices of points inside the bounding box.
    3. Changes the color of all points outside the bounding box to black.
    4. Keeps the original color of points inside the bounding box.
    5. Saves the modified point cloud to a new file with "_GT" added to the original filename.

    Input:
        Path to the point cloud file is hardcoded within the function.

    Output:
        A new file is created with the modified point cloud, saved in the same directory as the original.
    """

    # Read point cloud from file
    point_cloud = o3d.io.read_point_cloud(input_path)

    # Convert point cloud to numpy array for mathematical operations
    points_array = np.asarray(point_cloud.points)

    # Extract maximum and minimum coordinates of the point cloud
    max_coordinates = np.max(points_array, axis=0)
    min_coordinates = np.min(points_array, axis=0)
    print("Maximum coordinates:", max_coordinates)
    print("Minimum coordinates:", min_coordinates)
    print('Center of pointcloud:', point_cloud.get_center())
    # Initialize arrays to store combined point data and colors
    combined_points = []
    combined_colors = []

    # Set the bounding box coordinates based on the video source
    if 'Video_0' in input_path:
        pose = 'Poses_0.txt'
        pose_path = os.path.join(path_pose, pose)
        specimen_number = int(input_path.split('Specimen_')[1].split('\\')[0])
    elif 'Video_1' in input_path:
        pose = 'Poses_1.txt'
        pose_path = os.path.join(path_pose, pose)
        specimen_number = int(input_path.split('Specimen_')[1].split('\\')[0])
    else:
        raise ValueError("Input path must contain either 'Video_0' or 'Video_1'")
    frame = int(input_path.split('_')[-1].split('.')[0])
    vertebrae, _ = visualize_displacements(pose_path, frame, specimen=specimen_number)

    # Concatenate point data and colors from all point clouds
    for pc in vertebrae:
        combined_points.extend(np.asarray(pc.points))

        if pc.has_colors():
            combined_colors.extend(np.asarray(pc.colors))

    # Create a new Open3D point cloud
    combined_point_cloud = o3d.geometry.PointCloud()
    combined_point_cloud.points = o3d.utility.Vector3dVector(np.array(combined_points))

    if combined_colors:
        combined_point_cloud.colors = o3d.utility.Vector3dVector(np.array(combined_colors))

    # Create an oriented bounding box
    oriented_bbox = combined_point_cloud.get_oriented_bounding_box()

    # Extend the extent of the bounding box by 40%
    extent = oriented_bbox.extent * 2
    oriented_bbox.extent = extent

    oriented_bbox.color = (1.0, 0.0, 0.0)  # Red
    # Get indices of points inside the bounding box
    inside_bbox_mask = oriented_bbox.get_point_indices_within_bounding_box(point_cloud.points)
    # Create a black color array for all points
    black_color = np.zeros_like(np.asarray(point_cloud.colors))

    # If the original colors are not defined, use the default color
    if point_cloud.colors:
        original_colors = np.asarray(point_cloud.colors)
    else:
        original_colors = np.ones_like(points_array)  # Default white color

    inside_bounding_box_indices = inside_bbox_mask
    # Set the color of points outside the bounding box to black
    black_color[inside_bounding_box_indices] = original_colors[inside_bounding_box_indices]
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
    # Update the color of the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(black_color)
    o3d.visualization.draw_geometries([point_cloud, coord_frame, oriented_bbox])
    # Extract the base filename and add '_GT' before the extension
    base_filename = os.path.basename(input_path)
    filename_without_extension, extension = os.path.splitext(base_filename)
    output_filename = filename_without_extension + "_GT" + extension

    # Combine with the original directory path
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the modified point cloud to the file
    o3d.io.write_point_cloud(output_path, point_cloud)

    print(f"Saved to {output_path}")
    return inside_bounding_box_indices


if __name__ == "__main__":
    # Define the input path
    input_path = "Pointcloud/Video_1/Pointcloud_0.pcd"
    poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"
    # Run the function
    idx = process_point_cloud(input_path=input_path)
