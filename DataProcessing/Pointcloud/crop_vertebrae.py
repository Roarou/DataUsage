import open3d as o3d
import numpy as np
import os
from DataProcessing.get_vertebras_displacement import visualize_displacements
import random
import re
# Color definitions for vertebral segments
colors = [[1, 0, 0],  # Red
          [0, 1, 0],  # Green
          [0, 0, 1],  # Blue
          [1, 1, 0],  # Yellow
          [1, 0, 1]]  # Magenta


def process_point_cloud(input_path: str, path_pose, gt_path=None) -> dict:
    """
    Process and visualize a point cloud focusing on vertebral segments.

    The function identifies vertebral segments, computes their oriented bounding boxes, and colors the points accordingly.

    Args:
    - input_path (str): Path to the input point cloud file.
    - path_pose (str): Base path to retrieve the corresponding pose data.
    - gt_path (str, optional): Path to save the modified point cloud. Defaults to input directory.

    Returns:
    - dict: Map of vertebral labels ("L1" to "L5") to point indices within bounding boxes.
    """
    # Read the point cloud from the provided path
    try:
        point_cloud = o3d.io.read_point_cloud(input_path, remove_nan_points=True, remove_infinite_points=True)
    except Exception as e:
        print(f"An error occurred: {e}")
    o3d.visualization.draw_geometries([point_cloud])
    # Determine the video source and retrieve the corresponding pose data
    if 'Video_0' in input_path:
        pose = 'Poses_0.txt'
    elif 'Video_1' in input_path:
        pose = 'Poses_1.txt'
    else:
        raise ValueError("Input path must contain either 'Video_0' or 'Video_1'")

    pose_path = os.path.join(path_pose, pose)

    specimen_number = int(input_path.split('Specimen_')[1].split('\\')[0])
    frame = int(input_path.split('_')[-1].split('.')[0])
    # Compute vertebral segment oriented bounding boxes
    vertebrae, _ = visualize_displacements(pose_path, frame, specimen=specimen_number)
    vertebrae_indices = {f"L{i + 1}": None for i in range(5)}

    # Process each vertebral point cloud and compute bounding boxes
    for i, pc in enumerate(vertebrae):
        oriented_bbox = pc.get_oriented_bounding_box()
        oriented_bbox.color = random.choice(colors)

        vertebrae_indices[f"L{i + 1}"] = oriented_bbox.get_point_indices_within_bounding_box(point_cloud.points)
    # Color the points outside the bounding boxes black, and those inside with defined colors
    black_color = np.zeros_like(np.asarray(point_cloud.colors))
    for i, indices in enumerate(vertebrae_indices.values()):
        black_color[indices] = o3d.utility.Vector3dVector([colors[i]] * len(indices))

    point_cloud.colors = o3d.utility.Vector3dVector(black_color)

    # Save the modified point cloud (if needed)
    # base_filename = os.path.basename(input_path)
    # output_filename = os.path.splitext(base_filename)[0] + "_GT" + os.path.splitext(base_filename)[1]


    # Use regular expressions to extract numbers
    numbers = re.findall(r'\d+', input_path)

    # Extract values based on their positions in the list
    if len(numbers) >= 4:
        spec = numbers[0]  # 2
        rec = numbers[1]  # 3
        vid = numbers[2]  # 123
        pcd = numbers[3]  # 1
    else:
        # Handle the case where there are not enough numbers in the input path
        spec, rec, pcd, vid = "", "", "", ""
    output_filename = f'Poincloud_{pcd}_GT_Specimen_{spec}_Recording{rec}_Video_{vid}_aligned.pcd'
    for j,v in enumerate(vertebrae):
        point_cloud = v
        output_filename = f'L_{j+1}_Specimen_{spec}.pcd'
        output_path = os.path.join(gt_path or os.path.dirname(input_path), output_filename)
        SUCCESS = o3d.io.write_point_cloud(output_path, point_cloud)
    # Visualize and print save path
    #  o3d.visualization.draw_geometries([point_cloud, vertebrae[0],vertebrae[1],vertebrae[4],vertebrae[2],vertebrae[3]])
    #   print(f"Saved to {output_path}")
    #  SUCCESS = o3d.io.write_point_cloud(output_path, point_cloud)

    return vertebrae_indices#  , SUCCESS


if __name__ == "__main__":
    # Define the input path
    input_path = r"G:\SpineDepth\Specimen_2\Recordings\Recording10\pointcloud\Video_0\Pointcloud_0.pcd"
    poses_1_file_path = r"G:\SpineDepth\Specimen_2\Recordings\Recording10"
    # Run the function
    idx = process_point_cloud(input_path=input_path, path_pose=poses_1_file_path, gt_path=r"G:")
