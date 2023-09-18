import open3d as o3d
import numpy as np
import os
from DataProcessing.get_vertebras_displacement import visualize_displacements
import random

colors = [[1, 0, 0],  # Red
          [0, 1, 0],  # Green
          [0, 0, 1],  # Blue
          [1, 1, 0],  # Yellow
          [1, 0, 1]]  # Magenta


def process_point_cloud(input_path: str, path_pose, gt_path=None):
    """
    Processes a point cloud from a specified input file, particularly focusing on identifying
    the bounding box of vertebral segments, and updating the color of the points accordingly.

    Steps:
    1. Reads the point cloud from the provided input path.
    2. Determines the video source (either 'Video_0' or 'Video_1') from the input path and retrieves the corresponding
    pose data.
    3. Identifies vertebral segments in the point cloud and computes their oriented bounding boxes.
    4. Determines indices of points inside each vertebral bounding box.
    5. Updates the color of all points outside the bounding boxes to black while retaining the original color for
    those inside.
    6. Optionally saves the modified point cloud to a new file with "_GT" added to the  original filename, in the same
    directory or a specified one.

    Args:
        input_path (str): Path to the input point cloud file.
        path_pose (str): Base path to retrieve the corresponding pose data.
        gt_path (str, optional): Destination directory to save the modified point cloud. If not provided, the original
        directory is used.

    Returns:
        dict: A dictionary mapping vertebral labels ("L1" to "L5") to the indices of points within their corresponding
        bounding boxes.

    Raises:
        ValueError: If the input path doesn't contain either 'Video_0' or 'Video_1'.
    """
    # Read point cloud from file
    point_cloud = o3d.io.read_point_cloud(input_path)
    final_pcd = o3d.geometry.PointCloud()
    # Convert point cloud to numpy array for mathematical operations
    points_array = np.asarray(point_cloud.points)

    # Initialize arrays to store combined point data and colors
    inside_bounding_box_indices = []
    vertebrae_indices = {
        "L1": None,
        "L2": None,
        "L3": None,
        "L4": None,
        "L5": None
    }
    combined_points = []
    combined_colors = []
    pcds = []
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
    for i, pc in enumerate(vertebrae):
        combined_points = np.asarray(pc.points)
        # Create a new Open3D point cloud
        combined_point_cloud = o3d.geometry.PointCloud()
        combined_point_cloud.points = o3d.utility.Vector3dVector(np.array(combined_points))

        # Create an oriented bounding box
        oriented_bbox = combined_point_cloud.get_oriented_bounding_box()
        oriented_bbox.color = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))  # Red
        idx = 'L' + str(i + 1)
        # Get indices of points inside the bounding box
        vertebrae_indices[idx] = oriented_bbox.get_point_indices_within_bounding_box(point_cloud.points)
    # Create a black color array for all points
    black_color = np.zeros_like(np.asarray(point_cloud.colors))
    # If the original colors are not defined, use the default color
    for id in pcds:
        final_pcd += id

    for vertebra in vertebrae_indices:
        inside_bounding_box_indices.append(vertebrae_indices[vertebra])
    # Set the color of points outside the bounding box to black
    i = 0
    print(len(inside_bounding_box_indices))
    for single_vertebra_indices in inside_bounding_box_indices:
        num_points_in_subcloud = len(single_vertebra_indices)
        black_color[single_vertebra_indices] = o3d.utility.Vector3dVector([colors[i]] * num_points_in_subcloud)
        i += 1
    # Update the color of the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(black_color)

    # Extract the base filename and add '_GT' before the extension
    base_filename = os.path.basename(input_path)
    filename_without_extension, extension = os.path.splitext(base_filename)
    output_filename = filename_without_extension + "_GT" + extension

    # Combine with the original directory path
    if gt_path:
        output_path = os.path.join(gt_path, output_filename)
    else:
        output_path = os.path.join(os.path.dirname(input_path), output_filename)
    # Save the modified point cloud to the file
    # SUCCESS = o3d.io.write_point_cloud(output_path, point_cloud)
    o3d.visualization.draw_geometries([point_cloud])
    print(f"Saved to {output_path}")
    return vertebrae_indices  #, SUCCESS


if __name__ == "__main__":
    # Define the input path
    input_path = r"G:\SpineDepth\Specimen_2\Recordings\Recording0\pointcloud\Video_0\Pointcloud_0.pcd"
    poses_1_file_path = r"G:\SpineDepth\Specimen_2\Recordings\Recording0"
    # Run the function
    idx = process_point_cloud(input_path=input_path, path_pose=poses_1_file_path)
