import open3d as o3d
import numpy as np
import os


def process_point_cloud(input_path: str):
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

    # Set the bounding box coordinates based on the video source
    if 'Video_0' in input_path:
        min_bounding_box_coords = [-200, -50, 600]
        max_bounding_box_coords = [250, 200, 800]
    elif 'Video_1' in input_path:
        min_bounding_box_coords = [-50, -100, 600]
        max_bounding_box_coords = [300, 200, 780]
    else:
        raise ValueError("Input path must contain either 'Video_0' or 'Video_1'")

    # Find the indices of points within the bounding box
    inside_bounding_box_indices = np.where(
        (points_array[:, 0] >= min_bounding_box_coords[0]) &
        (points_array[:, 1] >= min_bounding_box_coords[1]) &
        (points_array[:, 2] >= min_bounding_box_coords[2]) &
        (points_array[:, 0] <= max_bounding_box_coords[0]) &
        (points_array[:, 1] <= max_bounding_box_coords[1]) &
        (points_array[:, 2] <= max_bounding_box_coords[2])
    )[0]

    # Create a black color array for all points
    black_color = np.zeros_like(np.asarray(point_cloud.colors))

    # If the original colors are not defined, use the default color
    if point_cloud.colors:
        original_colors = np.asarray(point_cloud.colors)
    else:
        original_colors = np.ones_like(points_array)  # Default white color

    # Set the color of points outside the bounding box to black
    black_color[inside_bounding_box_indices] = original_colors[inside_bounding_box_indices]

    # Update the color of the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(black_color)

    # Extract the base filename and add '_GT' before the extension
    base_filename = os.path.basename(input_path)
    filename_without_extension, extension = os.path.splitext(base_filename)
    output_filename = filename_without_extension + "_GT" + extension

    # Combine with the original directory path
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    # Save the modified point cloud to the file
    o3d.io.write_point_cloud(output_path, point_cloud)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Define the input path
    input_path = "Pointcloud/Video_0/Pointcloud_0.pcd"
    # Run the function
    process_point_cloud(input_path=input_path)
