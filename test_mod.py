import torch
from Model.spine_multi_class_segmentation import FastNet
import os
import numpy as np
import open3d as o3d
from Model.load_dataset_multi import normalize_point_cloud
import copy

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


# Define the path to your model's .pth file
model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\log\sem_seg\2023-11-15_02-34\checkpoints\best_model.pth'
file = 'Pointcloud_0_spec_Specimen_2_vid_0_Recording0.pcd'
path = r'L:\Pointcloud'

path = os.path.join(path, file)
# Recreate the same mode l architecture as was used during training
model = FastNet()  # Replace with your actual model

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

sampled_indices = np.random.choice(nb, 100000, replace=False)
point = input[sampled_indices]
down_col = input_col[sampled_indices]




downscale = copy.deepcopy(point)
downscale = normalize_point_cloud(downscale, path)
downscale = torch.tensor(downscale, dtype=torch.float32)
downscale = downscale.unsqueeze(0)
downscale = downscale.float()
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
predictions = np.array([map_label_to_color(c) for c in predictions])

for i, color in enumerate(colors):
    matches = np.all(down_col == color, axis=1)

    # Find indices where matches are True
    L1 = np.where(matches)[0]
    L1_col = down_col[L1]
    L1_point = point[L1]
    pcd_or = o3d.geometry.PointCloud()
    pcd_or.points = o3d.utility.Vector3dVector(L1_point)
    pcd_or.colors = o3d.utility.Vector3dVector(L1_col)
    print(len(pcd_or.points))
    # Compute the bounds of the first point cloud
    min_bound = pcd_or.get_min_bound()
    max_bound = pcd_or.get_max_bound()

    # Calculate the amount to translate
    extra_space = 10
    # Here, we shift the second point cloud to the right of the first
    # You might want to add some extra space between them
    shift = max_bound[0] - min_bound[0] + extra_space
    pcd_or.paint_uniform_color([0.5, 0.5, 0.5])


    if i== 1:
        i = 4
    if i == 2:
        i = 1
    if i== 3:
        i = 2
    if i== 4:
        i = 3
    L1_col = predictions[idx[i]]
    L1_point = point[idx[i]]
    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(L1_point)
    pcd_modified.colors = o3d.utility.Vector3dVector(L1_col)

    pcd_f = o3d.geometry.PointCloud()
    pcd_f  = pcd_modified + pcd_or

    # Translate the second point cloud
    # pcd_modified.translate((shift, 0, 0))
    o3d.visualization.draw_geometries([pcd_modified, pcd_or])
    curr_dir = os.getcwd()
    filename = f'L{i+1}_fused.pcd'
    final_path = os.path.join(curr_dir, filename)
    SUCCESS = o3d.io.write_point_cloud(final_path, pcd_f)

# pcd_modified = o3d.geometry.PointCloud()
# pcd_modified.points = o3d.utility.Vector3dVector(point)
# pcd_modified.colors = o3d.utility.Vector3dVector(predictions)
#
# o3d.visualization.draw_geometries([pcd_modified])