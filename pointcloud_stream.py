import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys
import os

def get_np_array(array):
    return np.asarray(array)


def from_np_to_pcd(points, colors):
    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(points)
    pcd_modified.colors = o3d.utility.Vector3dVector(colors)
    return pcd_modified


def get_o3d_pcd(point_cloud):
    points = get_np_array(point_cloud.get_data())
    mask = ~np.isnan(points).any(axis=2)
    filtered_points = points[mask]
    xyz = filtered_points[:, :3].astype(np.float32)

    rgb = np.frombuffer(np.float32(filtered_points[:, 3]).tobytes(), np.uint8).reshape(-1, 4)[:, :3] / 255.0

    pcd = from_np_to_pcd(xyz, rgb)
    return pcd

def save_pcd(pcd, idx):
    save_directory = "saved_pointclouds"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    o3d.io.write_point_cloud(f"{save_directory}/pointcloud_{idx}.pcd", pcd)

def main():
    # Initialize ZED camera
    init_params = sl.InitParameters()
    cam = sl.Camera()

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
        for idx in range(200):
            # Grab a new frame from the live stream
            err = cam.grab(runtime_params)

            if err == sl.ERROR_CODE.SUCCESS:
                # Extract point cloud data
                cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)

                # Convert ZED point cloud to Open3D format
                pcd = get_o3d_pcd(point_cloud_mat)

                # Save the point cloud
                save_pcd(pcd, idx)
                print(f"Saved point cloud #{idx + 1} successfully.")
            else:
                print(f"Failed to grab frame #{idx + 1}.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cam.close()

if __name__ == "__main__":
    main()
