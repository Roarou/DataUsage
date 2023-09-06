import copy
from Model.load_dataset import normalize_point_cloud
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys
import torch
from mod import model


def get_o3d_pcd(point_cloud):
    points = get_np_array(point_cloud.get_data())
    mask = ~np.isnan(points).any(axis=2)
    filtered_points = points[mask]
    xyz = filtered_points[:, :3].astype(np.float32)

    rgb = np.frombuffer(np.float32(filtered_points[:, 3]).tobytes(), np.uint8).reshape(-1, 4)[:, :3] / 255.0

    pcd = from_np_to_pcd(xyz, rgb)
    return pcd


def get_np_array(array):
    return np.asarray(array)


def get_prediction(prediction, pcd):
    points = get_np_array(pcd.points)
    colors = get_np_array(pcd.colors)
    binary_predictions = (prediction >= 0.5).float().numpy()[0]
    spine_points = points[binary_predictions == 1]
    spine_colors = colors[binary_predictions == 1]
    pred = from_np_to_pcd(spine_points, spine_colors)
    return pred


def return_sample_idx(len_input, len_output=100000):
    return np.random.choice(len_input, len_output, replace=False)


def from_np_to_pcd(points, colors):
    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(points)
    pcd_modified.colors = o3d.utility.Vector3dVector(colors)
    return pcd_modified


def normalize_pcd(len_input, pcd):
    sampled_indices = return_sample_idx(len_input)
    points = get_np_array(pcd.points)
    colors = get_np_array(pcd.colors)
    down_points = points[sampled_indices]
    down_col = colors[sampled_indices]
    pcd_downscaled = from_np_to_pcd(points=down_points, colors=down_col)
    pcd_norm = copy.deepcopy(points)
    pcd_norm = normalize_point_cloud(pcd_norm)
    pcd_norm = torch.tensor(pcd_norm, dtype=torch.float32)
    return pcd_norm, pcd_downscaled


def main():
    # Initialize ZED camera
    init_params = sl.InitParameters()
    cam = sl.Camera()
    point_clouds_normalized = []
    point_clouds_downscaled = []
    counter = 0

    # Open the camera
    if not cam.is_opened():
        print("Opening ZED Camera...")
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Failed to open the camera.")
            sys.exit(1)

    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", 1280, 720)

    # Set runtime parameters for live camera feed
    runtime_params = sl.RuntimeParameters()

    # Initialize point cloud matrix
    point_cloud_mat = sl.Mat()

    print("Starting point cloud extraction and visualization...")

    try:
        while True:
            # Grab a new frame from the live stream
            err = cam.grab(runtime_params)

            if err == sl.ERROR_CODE.SUCCESS:
                # Extract point cloud data
                cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

                # Convert ZED point cloud to Open3D format
                pcd = get_o3d_pcd(point_cloud_mat)
                # Append to list and update counter
                point_len = len(np.asarray(pcd.points))
                pcd_norm, pcd_downscaled = normalize_pcd(point_len, pcd)
                point_clouds_normalized.append(pcd_norm)
                point_clouds_downscaled.append(pcd_downscaled)  # Convert to PyTorch tensor and append
                counter += 1

                if counter == 24:
                    stacked_point_clouds = torch.stack(point_clouds_normalized, dim=0)
                    # Pass the stacked tensor to your model here
                    output, _ = model(stacked_point_clouds)
                    for i, pred in enumerate(output):
                        pcd = get_prediction(pred, point_clouds_downscaled[i])
                        vis.update_geometry(pcd)
                        vis.poll_events()
                        vis.update_renderer()
                    # Reset the list and counter
                    point_clouds_normalized = []
                    point_clouds_downscaled = []
                    counter = 0

            else:
                print("Failed to grab a frame.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        vis.destroy_window()
        cam.close()


if __name__ == "__main__":
    main()
