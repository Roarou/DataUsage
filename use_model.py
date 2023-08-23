import torch
from Model.spine_segmentation import SpineSegmentationNet
from Model.load_dataset import normalize_point_cloud
import open3d as o3d
import numpy as np
import time
pointcloud_path = r'L:\Specimen_6\Recordings\Recording0\pointcloud\Video_0\Pointcloud_0.pcd'
time_start = time.time()
pcd = o3d.io.read_point_cloud(pointcloud_path)

input = np.asarray(pcd.points)
input_col = np.asarray(pcd.colors)
sampled_indices = np.random.choice(len(input), 200000, replace=False)
point = input[sampled_indices]
down_col = input_col[sampled_indices]
downscale = normalize_point_cloud(point)
downscale = torch.tensor(downscale, dtype=torch.float32)
downscale = downscale.unsqueeze(0)
####o3d.visualization.draw_geometries([pcd])

model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\Model\model_epoch_2.pt'
dic = torch.load(model_path)
model = SpineSegmentationNet()
model.load_state_dict(dic)
model.eval()
start_time = time.time()
with torch.no_grad():

    predictions, _ = model(downscale)

end_time = time.time()

time_taken = end_time - start_time
print(f"Time taken for a single prediction: {time_taken} seconds")

binary_predictions = (predictions >= 0.5).float().numpy()[0]


spine_points = point[binary_predictions == 1]
spine_colors = down_col[binary_predictions == 1]

pcd_modified =  o3d.geometry.PointCloud()
pcd_modified.points =  o3d.utility.Vector3dVector(spine_points)
pcd_modified.colors =  o3d.utility.Vector3dVector(spine_colors)
aabbox = pcd_modified.get_axis_aligned_bounding_box()
time_end = time.time()
time_taken_script = time_end-time_start
print(f"Time taken for the entire script: {time_taken_script} seconds")
o3d.visualization.draw_geometries([pcd_modified, aabbox])

