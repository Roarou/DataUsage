import torch
from Model.pointnet_multi_class.spine_multi_class_segmentation import SpineSegmentationNet
from Model.pointnet_binary.load_dataset import normalize_point_cloud
import open3d as o3d
import numpy as np
import time
import os
import copy

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

directory_path = r'saved_pointclouds'

# Load the model once
model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\Model\segmentation_multi\model_dic_epoch_1.pt'
dic = torch.load(model_path)
model = SpineSegmentationNet()
model = model.to(device)
model.load_state_dict(dic)
model.eval()


def predict_on_pointcloud(pointcloud_path):
    time_start = time.time()

    pcd = o3d.io.read_point_cloud(pointcloud_path)
    input = np.asarray(pcd.points)
    input_col = np.asarray(pcd.colors)
    nb = len(input)

    sampled_indices = np.random.choice(nb, 100000, replace=False)
    point = input[sampled_indices]
    down_col = input_col[sampled_indices]
    downscale = copy.deepcopy(point)
    downscale = normalize_point_cloud(downscale, pointcloud_path)
    downscale = torch.tensor(downscale, dtype=torch.float32)
    downscale = downscale.unsqueeze(0)
    downscale = downscale.to(device)
    o3d.visualization.draw_geometries([pcd])
    with torch.no_grad():
        predictions, _ = model(downscale)
    np.set_printoptions(threshold=np.inf)
    predictions = predictions.cpu()

    binary_predictions = torch.argmax(predictions, dim=0).float().numpy()[0]

    print(binary_predictions)
    mask = (binary_predictions == 1) | (binary_predictions == 2) | (binary_predictions == 3) | (
                binary_predictions == 4) | (binary_predictions == 5)
    spine_points = point[mask]
    spine_colors = down_col[mask]

    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(spine_points)
    pcd_modified.colors = o3d.utility.Vector3dVector(spine_colors)

    aabbox = pcd_modified.get_oriented_bounding_box()
    aabbox.color = (0.0, 1.0, 0.0)  # Green

    time_end = time.time()
    time_taken_script = time_end - time_start

    print(f"Time taken for the entire script on {pointcloud_path}: {time_taken_script} seconds")
    o3d.visualization.draw_geometries([pcd_modified, aabbox])

    final_path = os.path.join(os.path.dirname(pointcloud_path), f"Pointcloud_pred_{os.path.basename(pointcloud_path)}")
    SUCCESS = o3d.io.write_point_cloud(final_path, pcd_modified)

    if SUCCESS:
        print(f'Saved to: {final_path}')


if __name__ == "__main__":
    for pointcloud_file in os.listdir(directory_path):
        if pointcloud_file.endswith('.pcd'):
            predict_on_pointcloud(os.path.join(directory_path, pointcloud_file))