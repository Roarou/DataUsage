import torch
from Model.spine_segmentation import SpineSegmentationNet
from Model.load_dataset import normalize_point_cloud
import open3d as o3d
import numpy as np
import time
import os
import copy

pointcloud_path = r'G:\SpineDepth\Specimen_6\Recordings\Recording5\pointcloud\Video_1\Pointcloud_120.pcd'
time_start = time.time()
pcd = o3d.io.read_point_cloud(pointcloud_path)
input = np.asarray(pcd.points)
input_col = np.asarray(pcd.colors)
nb = len(input)
print(nb)
check = False
sampled_indices = np.random.choice(nb, 100000, replace=False)
point = input[sampled_indices]
down_col = input_col[sampled_indices]
downscale = copy.deepcopy(point)
downscale = normalize_point_cloud(downscale)
downscale = torch.tensor(downscale, dtype=torch.float32)
downscale = downscale.unsqueeze(0)

o3d.visualization.draw_geometries([pcd])

model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\Model\lr0,01_4epochs_24000data\model_epoch_4.pt'
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
print(np.max(spine_points,axis=0))
spine_colors = down_col[binary_predictions == 1]
pcd_modified = o3d.geometry.PointCloud()
pcd_modified.points = o3d.utility.Vector3dVector(spine_points)
pcd_modified.colors = o3d.utility.Vector3dVector(spine_colors)
aabbox = pcd_modified.get_oriented_bounding_box()
aabbox.color = (0.0, 1.0, 0.0)  # Green
time_end = time.time()
time_taken_script = time_end - time_start
oriented_bbox = pcd_modified.get_oriented_bounding_box()

print(f"Time taken for the entire script: {time_taken_script} seconds")
o3d.visualization.draw_geometries([pcd_modified, aabbox])
final_path = os.path.join(os.path.dirname(pointcloud_path), 'Pointcloud_pred.pcd')
SUCCESS = o3d.io.write_point_cloud(final_path, pcd_modified)
if SUCCESS:
    print(f'Saved to: {final_path}')
if check:
    print(len(spine_points))
    arr1 = spine_points
    arr2 = point
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    # Use intersect1d on the views
    intersected = np.intersect1d(arr1_view, arr2_view)
    # Convert back to original shape and dtype
    common_rows = intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])
    print("Common rows are:", common_rows)
    print("Number of common rows:", len(common_rows))