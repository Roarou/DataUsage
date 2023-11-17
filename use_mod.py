import torch
from Model.pointnet_multi.spine_multi_class_segmentation import SpineSegmentationNet
import os
import numpy as np
import open3d as o3d
from Model.pointnet_multi.load_dataset_multi import normalize_point_cloud
import copy
import pandas as pd

colors = {
    (1, 0, 0),  # L1
    (0, 1, 0),  # L2
    (0, 0, 1),  # L3
    (0, 1, 1),  # L4
    (1, 0, 1)
}


def map_label_to_color(color):
    mapping = {
        0: [1, 0, 0],  # L1
        1: [0, 1, 0],  # L2
        2: [0, 0, 1],  # L3
        3: [0, 1, 1],  # L4
        4: [1, 0, 1],  # L5
        5: [0, 0, 0]  # Non-spine
    }
    res = mapping.get(color, -1)
    return res  # -1 for any unexpected colors


def create_transformation_matrix(df_row):
    """
    Creates a transformation matrix from a DataFrame row containing rotation and translation data.

    :param df_row: A Pandas Series with rotation matrix R and translation vector T components.
    :return: A 4x4 NumPy array representing the transformation matrix.
    """
    # Extract the rotation and translation values to construct the transformation matrix
    return np.array([
        [df_row["R00"], df_row["R01"], df_row["R02"], df_row["T0"]],
        [df_row["R10"], df_row["R11"], df_row["R12"], df_row["T1"]],
        [df_row["R20"], df_row["R21"], df_row["R22"], df_row["T2"]],
        [0, 0, 0, 1]  # The bottom row of a transformation matrix is always [0, 0, 0, 1]
    ])


def read_and_transform_vertebra(base_path, specimen, vertebra_name, transformation):
    """
    Reads a vertebra model from an STL file, computes its vertex normals, and applies a transformation.

    :param base_path: The base path to the specimen data.
    :param specimen: The specimen identifier.
    :param vertebra_name: The name of the vertebra to be transformed.
    :param transformation: The transformation matrix to be applied.
    :return: The transformed vertebra mesh.
    """
    # Read the STL file into an Open3D triangle mesh object
    vertebra = o3d.io.read_triangle_mesh(os.path.join(base_path, specimen, "STL", f"{vertebra_name}.stl"))
    # Compute the normals for the vertices of the mesh
    vertebra.compute_vertex_normals()
    # Apply the transformation to the vertebra mesh
    vertebra.transform(transformation)
    return vertebra


# Define the path to your model's .pth file
cur_frame = 0
model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\log\sem_seg\2023-11-15_02-34\checkpoints\best_model.pth'
model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\log\sem_seg\2023-11-15_16-47\checkpoints\best_model.pth'
specimen = "Specimen_2"
file = r'C:\Users\cheg\PycharmProjects\DataUsage\saved_pointclouds\pointcloud_7.pcd'
path = r'L:\Pointcloud'
SOURCE_DIR = r"G:\SpineDepth"
tracking_file = r'G:\SpineDepth\Specimen_2\Recordings\Recording0\Poses_0.txt'
df = pd.read_csv(tracking_file, sep=',', header=None,
                 names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2",
                        "R30",
                        "R31", "R33", "T3"])
path = os.path.join(path, file)
# Recreate the same mode l architecture as was used during training
model = SpineSegmentationNet()  # Replace with your actual model
# Read and transform vertebrae
vertebrae = []

for i in range(5):
    df_row = df.iloc[5 * cur_frame + i]
    transformation_matrix = create_transformation_matrix(df_row)
    vertebra = read_and_transform_vertebra(SOURCE_DIR, specimen, f"L{i + 1}", transformation_matrix)
    vertebrae.append(vertebra)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load the saved state dictionary
state_dict = torch.load(model_path)

# Apply the state dictionary to the mode
model.load_state_dict(state_dict['model_state_dict'])

# Set the model to evaluation mode
model.eval()

pcd = o3d.io.read_point_cloud(path)
input = np.asarray(pcd.points)
input_col = np.asarray(pcd.colors)
nb = len(input)

sampled_indices = np.random.choice(nb, 400000, replace=False)
point = input[sampled_indices]
down_col = input_col[sampled_indices]

downscale = copy.deepcopy(point)
downscale = normalize_point_cloud(downscale, path)
downscale = torch.tensor(downscale, dtype=torch.float32)
downscale = downscale.unsqueeze(0)
downscale = downscale.float()

pcd_f = o3d.geometry.PointCloud()
pcd_f.points = o3d.utility.Vector3dVector(point)
pcd_f.colors = o3d.utility.Vector3dVector(down_col)
o3d.visualization.draw_geometries([pcd_f])
# o3d.visualization.draw_geometries([pcd])
downscale = downscale.transpose(2, 1)
with torch.no_grad():
    predictions, _ = model(downscale)

predictions = predictions.cpu()[0]

predictions = torch.argmax(predictions, dim=-1).numpy()

# Convert the list to a NumPy array
my_array = np.array(predictions)

# Find indices where the value is 1

L1 = np.where(my_array == 0)[0]
L2 = np.where(my_array == 1)[0]
L3 = np.where(my_array == 2)[0]
L4 = np.where(my_array == 3)[0]
L5 = np.where(my_array == 4)[0]
idx = [L1, L2, L3, L4, L5]
pcd_f = o3d.geometry.PointCloud()

for i, color in enumerate(colors):
    L1_col = down_col[idx[i]]
    L1_point = point[idx[i]]
    pcd_or = o3d.geometry.PointCloud()
    pcd_or.points = o3d.utility.Vector3dVector(L1_point)
    pcd_or.colors = o3d.utility.Vector3dVector(L1_col)


    if i == 1:
        i = 4
    if i == 2:
        i = 1
    if i == 3:
        i = 2
    if i == 4:
        i = 3



    pcd_f += pcd_or

    # Translate the second point cloud
    # pcd_modified.translate((shift, 0, 0))
    # o3d.visualization.draw_geometries([pcd_modified])
    # curr_dir = os.getcwd()
    # filename = f'L{i + 1}_fused.pcd'
    # final_path = os.path.join(curr_dir, filename)
    # SUCCESS = o3d.io.write_point_cloud(final_path, pcd_f)
o3d.visualization.draw_geometries([pcd_f])
# pcd_modified = o3d.geometry.PointCloud()
# pcd_modified.points = o3d.utility.Vector3dVector(point)
# pcd_modified.colors = o3d.utility.Vector3dVector(predictions)
#
# o3d.visualization.draw_geometries([pcd_modified])
