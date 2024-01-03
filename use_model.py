import torch
from Model.pointnet_multi_class.spine_segmentation import SpineSegmentationNet
from Model.pointnet_binary.load_dataset import normalize_point_cloud
import open3d as o3d
import numpy as np
import time
import copy

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

pointcloud_path = r'G:\SpineDepth\Specimen_1\Recordings\Recording0\pointcloud\Video_0\Pointcloud_50.pcd'
pointcloud_path = r'C:\Users\cheg\PycharmProjects\DataUsage\saved_pointclouds\pointcloud_0.pcd'
time_start = time.time()
pcd = o3d.io.read_point_cloud(pointcloud_path)
input = np.asarray(pcd.points)
input_col = np.asarray(pcd.colors)
nb = len(input)
print(nb)
o3d.visualization.draw_geometries([pcd])
sampled_indices = np.random.choice(nb, 100000, replace=False)
point = input[sampled_indices]
down_col = input_col[sampled_indices]
downscale = copy.deepcopy(point)
downscale = normalize_point_cloud(downscale, pointcloud_path)
downscale = torch.tensor(downscale, dtype=torch.float32)
downscale = downscale.unsqueeze(0)
downscale = downscale.to(device)

pcdd =  o3d.geometry.PointCloud()
pcdd.points =  o3d.utility.Vector3dVector(point)
pcdd.colors =  o3d.utility.Vector3dVector(down_col)
o3d.visualization.draw_geometries([pcdd])
model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\Model\lr_0,0001_all_data\model_1_epoch_28.pth'
dic = torch.load(model_path,map_location=torch.device('cpu'))
model = SpineSegmentationNet()
model.load_state_dict(dic['model_state_dict'])
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

o3d.io.write_point_cloud('test.pcd',pcd_modified)
