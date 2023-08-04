import open3d as o3d
import numpy as np
import os
from Visualization.extract_tf_matrix_groundtruth import extract_transformation_matrices
# Specimen 3 AND CHECK HERE HOW TO CROP BETTER
min_bounding_box_coords = [-100, -250, 500]
max_bounding_box_coords = [100, 220, 610]
# Specimen 4
min_bounding_box_coords = [-300, -200, 400]
max_bounding_box_coords = [100, 170, 700]

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
    x,y,z = point_cloud.get_center()
    print('Center of pointcloud:', point_cloud.get_center())
    # Set the bounding box coordinates based on the video source
    if 'Video_0' in input_path:
        pose = 'Poses_0.txt'
        # Specimen 5 maybe, It'll take much much longer than expected
        pose_path = os.path.join(path_pose, pose)
        tf = extract_transformation_matrices(pose_path)
        tf = tf[0]
        x,y,z = tf[:3, 3]
        """
        min_bounding_box_coords = [-150, -150, 500]
        max_bounding_box_coords = [250, 200, 700]
        # Specimen 1
        min_bounding_box_coords = [x-600, y+170, z-850]
        max_bounding_box_coords = [x-100, y+450, z-550]
        # Specimen 2
        # min_bounding_box_coords = [x, y+40, z-500]
        # max_bounding_box_coords = [x+450, y+290, z-300]
        # Specimen 3
        # min_bounding_box_coords = [-100, -130, 500]
        # max_bounding_box_coords = [250, 200, 650]
        # Specimen 4 Problem here
        # min_bounding_box_coords = [-100, -130, 500]
        # max_bounding_box_coords = [250, 200, 650]
"""
    elif 'Video_1' in input_path:
        pose = 'Poses_1.txt'
        pose_path = os.path.join(path_pose, pose)
        frame = int(input_path.split('_')[-1].split('.')[0])
        tf = extract_transformation_matrices(pose_path)
        print(pose_path)
        v_0 = tf[0+frame*5]
        v_2 = tf[2+frame*5]
        v_1 = tf[4+frame*5]
        x0, y0, z0 = v_0[:3, 3]
        x, y, z = v_1[:3, 3]
        points = [v_0[:3, 3], v_1[:3, 3], v_2[:3, 3]]
        min_x, max_x = min(x0, x), max(x0, x)
        min_y, max_y = min(y0, y), max(y0, y)
        min_z, max_z = min(z0, z), max(z0, z)
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        """
        min_bounding_box_coords = [-50, -100, 600]
        max_bounding_box_coords = [300, 200, 780]
        # Specimen 1
        min_bounding_box_coords = [x-75, y-400, z-200]
        max_bounding_box_coords = [x+225, y-20, z]
        """
        # Specimen 2
        min_bounding_box_coords = np.array([min_x-(300-abs(abs(min_x)-abs(max_x)))/2, min_y-(300-abs(abs(min_y)-abs(max_y)))/2, min_z-(0.3*abs(min_z))])
        max_bounding_box_coords = np.array([max_x+(300-abs(abs(min_x)-abs(max_x)))/2, max_y+(300-abs(abs(min_y)-abs(max_y)))/2, max_z])
        center = (min_bounding_box_coords + max_bounding_box_coords) / 2
        extent = max_bounding_box_coords - min_bounding_box_coords
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(normal)
        bbox = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
        bbox.color = (0, 0, 1)
        inside_bounding_box_mask = bbox.get_point_indices_within_bounding_box(point_cloud.points)
    else:
        raise ValueError("Input path must contain either 'Video_0' or 'Video_1'")

    # Find the indices of points within the bounding box
    """
    inside_bounding_box_indices = np.where(
        (points_array[:, 0] >= min_bounding_box_coords[0]) &
        (points_array[:, 1] >= min_bounding_box_coords[1]) &
        (points_array[:, 2] >= min_bounding_box_coords[2]) &
        (points_array[:, 0] <= max_bounding_box_coords[0]) &
        (points_array[:, 1] <= max_bounding_box_coords[1]) &
        (points_array[:, 2] <= max_bounding_box_coords[2])
    )[0]
"""
    # Create a black color array for all points
    black_color = np.zeros_like(np.asarray(point_cloud.colors))

    # If the original colors are not defined, use the default color
    if point_cloud.colors:
        original_colors = np.asarray(point_cloud.colors)
    else:
        original_colors = np.ones_like(points_array)  # Default white color


    inside_bounding_box_indices = inside_bounding_box_mask
    # Set the color of points outside the bounding box to black
    black_color[inside_bounding_box_indices] = original_colors[inside_bounding_box_indices]
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
    # Update the color of the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(black_color)
    # Create a new point with custom coordinates, size, and color
    new_point = np.array([[x, y, z]])  # Set the desired coordinates [x, y, z]
    new_point_color = [1.0, 0, 0.0]  # Set the desired color [R, G, B]
    new_point_radius = 10  # Set the desired size
    new_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=new_point_radius)
    new_point_sphere.paint_uniform_color(new_point_color)
    new_point_sphere.translate(new_point[0])
    # Add the new point to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate([point_cloud.points, new_point]))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([point_cloud.colors,np.tile(new_point_color, (1, 1))]))
    new_point = np.array([[x0, y0, z0]])  # Set the desired coordinates [x, y, z]
    new_point_color = [0.0, 1.0, 0.0]  # Set the desired color [R, G, B]
    new_point_radius = 10  # Set the desired size
    new_point_sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=new_point_radius)
    new_point_sphere2.paint_uniform_color(new_point_color)
    new_point_sphere2.translate(new_point[0])
    # Add the new point to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate([point_cloud.points, new_point]))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([point_cloud.colors, np.tile(new_point_color, (1, 1))]))
    o3d.visualization.draw_geometries([point_cloud, coord_frame, new_point_sphere, new_point_sphere2, bbox])
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
