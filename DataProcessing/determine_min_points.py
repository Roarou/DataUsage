import open3d as o3d
import os
from tqdm import tqdm
# Define the path to your dataset directory
dataset_dir = r'L:\groundtruth_labeled'

# Initialize a variable to store the minimum number of points
min_points = float('inf')
file_list = [filename for filename in os.listdir(dataset_dir) if filename.endswith('.pcd')]  # Adjust the file extension if needed
progress_bar = tqdm(file_list, desc="Calculating minimum points")

# Iterate through the files in the dataset directory
for filename in progress_bar:
    file_path = os.path.join(dataset_dir, filename)  # Adjust the file extension if needed
    pointcloud = o3d.io.read_point_cloud(file_path)
    num_points = len(pointcloud.points)
    min_points = min(min_points, num_points)

# Close the tqdm progress bar
progress_bar.close()
print("Minimum number of points in the dataset:", min_points)