import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys


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


def main():
    # Initialize parameters
    init_params1 = sl.InitParameters()
    init_params1.camera_resolution = sl.RESOLUTION.HD720
    init_params1.camera_fps = 30
    init_params1.camera_linux_id = 0  # Set the ID for the first camera

    init_params2 = sl.InitParameters()
    init_params2.camera_resolution = sl.RESOLUTION.HD720
    init_params2.camera_fps = 30
    init_params2.camera_linux_id = 1  # Set the ID for the second camera

    # Create Camera objects
    cam1 = sl.Camera()
    cam2 = sl.Camera()

    # Open the cameras
    status1 = cam1.open(init_params1)
    status2 = cam2.open(init_params2)

    if status1 != sl.ERROR_CODE.SUCCESS or status2 != sl.ERROR_CODE.SUCCESS:
        print("Error opening one or both cameras")
        exit(1)

    # Initialize Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", 1280, 720)

    # Set runtime parameters for live camera feed
    runtime_params = sl.RuntimeParameters()

    # Initialize point cloud matrix
    point_cloud_mat1 = sl.Mat()
    point_cloud_mat2 = sl.Mat()
    print("Starting point cloud extraction and visualization...")

    try:
        while True:
            # Grab a new frame from the live stream
            err1 = cam1.grab(runtime_params)
            err2 = cam2.grab(runtime_params)

            if err1 == sl.ERROR_CODE.SUCCESS and err2 == sl.ERROR_CODE.SUCCESS:
                # Extract point cloud data
                cam1.retrieve_measure(point_cloud_mat1, sl.MEASURE.XYZRGBA)
                cam2.retrieve_measure(point_cloud_mat2, sl.MEASURE.XYZRGBA)

                # Convert ZED point cloud to Open3D format
                pcd1 = get_o3d_pcd(point_cloud_mat1)
                pcd2 = get_o3d_pcd(point_cloud_mat2)

                # Update visualization
                vis.update_geometry(pcd1)
                vis.update_geometry(pcd2)
                vis.poll_events()
                vis.update_renderer()

            else:
                print("Failed to grab a frame.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        vis.destroy_window()
        cam1.close()
        cam2.close()


if __name__ == "__main__":
    main()
