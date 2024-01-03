import torch
from Model.pointnet_multi_class.spine_segmentation import SpineSegmentationNet
from Model.pointnet_binary.load_dataset import normalize_point_cloud
import open3d as o3d
import numpy as np
import time
import os
import copy
from tqdm import tqdm
from collections import Counter


def batch_inference(model, batch, device):
    batch = batch.to(device)
    with torch.no_grad():
        predictions, _ = model(batch)
    return predictions.cpu()


def is_valid_pointcloud(pcd):
    try:
        if len(pcd.points) == 0:
            return False
        return True
    except Exception as e:
        print(f"Error reading: {e}")
        return False


def load_and_preprocess_pointclouds(batch_paths):
    batch_data = []
    original_points = []
    original_colors = []

    for path in batch_paths:
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        sampled_indices = np.random.choice(len(points), 50000, replace=False)
        sampled_points = points[sampled_indices]
        sampled_colors = colors[sampled_indices]
        normalized_points = normalize_point_cloud(sampled_points, path)

        point_tensor = torch.tensor(normalized_points, dtype=torch.float32).unsqueeze(0)
        batch_data.append(point_tensor)

        original_points.append(sampled_points)
        original_colors.append(sampled_colors)

    batch_tensor = torch.cat(batch_data, dim=0)
    return batch_tensor, original_points, original_colors


def postprocess_and_save_batch(predictions, batch_paths, original_points, original_colors):
    for i, pred in enumerate(predictions):
        binary_predictions = (pred >= 0.5).float().numpy()
        spine_points = original_points[i][binary_predictions == 1]
        spine_colors = original_colors[i][binary_predictions == 1]

        pcd_modified = o3d.geometry.PointCloud()
        pcd_modified.points = o3d.utility.Vector3dVector(spine_points)
        pcd_modified.colors = o3d.utility.Vector3dVector(spine_colors)
        # Assuming pcd_modified.colors is a list of color tuples
        # color_counts = Counter(tuple(color) for color in pcd_modified.colors)
        #
        # # Display the counts for each color
        # for color, count in color_counts.items():
        #     print(f"Color {color}: {count} occurrences")
        # print(batch_paths[i])
        if is_valid_pointcloud(pcd_modified):
            aabbox = pcd_modified.get_oriented_bounding_box()
            aabbox.color = (0.0, 1.0, 0.0)

            # final_path = os.path.join(os.path.dirname(batch_paths[i]), f"Pointcloud_pred_{os.path.basename(batch_paths[i])}")
            final_path = os.path.join(r'C:\Users\cheg\PycharmProjects\DataUsage\saved_pointclouds', f"Pointcloud_pred_{os.path.basename(batch_paths[i])}")

            o3d.io.write_point_cloud(final_path, pcd_modified)


def process_pointcloud(pointcloud_path, model, device):
    time_start = time.time()

    pcd = o3d.io.read_point_cloud(pointcloud_path)
    input = np.asarray(pcd.points)
    input_col = np.asarray(pcd.colors)
    nb = len(input)

    sampled_indices = np.random.choice(nb, 200000, replace=False)
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

    predictions = predictions.cpu()
    binary_predictions = (predictions >= 0.5).float().numpy()[0]
    spine_points = point[binary_predictions == 1]
    spine_colors = down_col[binary_predictions == 1]

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

    return SUCCESS, final_path


def process_batch(batch, model, device):
    for pointcloud_path in batch:
        success, saved_path = process_pointcloud(pointcloud_path, model, device)
        if success:
            print(f'Successfully processed and saved: {saved_path}')
        else:
            print(f'Failed to process: {pointcloud_path}')


def main(directory_path, model, device, batch_size=10):
    pointcloud_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pcd')]

    for i in tqdm(range(0, len(pointcloud_files), batch_size), desc="Processing Point Clouds"):
        batch_paths = pointcloud_files[i:i + batch_size]
        batch_tensor, original_points, original_colors = load_and_preprocess_pointclouds(batch_paths)

        predictions = batch_inference(model, batch_tensor, device)
        # Check if predictions is a tuple and print the shape of each tensor
        # if isinstance(predictions, tuple):
        #     for j, tensor in enumerate(predictions):
        #         print(f"Shape of tensor {j} in predictions: {tensor.shape}")
        # else:
        #     print(predictions.shape)
        postprocess_and_save_batch(predictions, batch_paths, original_points, original_colors)


if __name__ == "__main__":
    directory_path = r"C:\Users\cheg\PycharmProjects\DataUsage\saved_pointclouds"  # Update this path if necessary
    batch_size = 10
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the model once
    model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\Model\lr_0,0001_all_data\model_dic_epoch_28.pt'
    dic = torch.load(model_path)
    model = SpineSegmentationNet()
    model = model.to(device)
    model.load_state_dict(dic)
    model.eval()

    main(directory_path, model, device, batch_size)
