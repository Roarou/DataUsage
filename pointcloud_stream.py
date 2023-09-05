import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys

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

    pcd = o3d.geometry.PointCloud()

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
                xyzrgba = point_cloud_mat.get_data()
                xyz = xyzrgba[:, :, :3].reshape((-1, 3))
                rgb = xyzrgba[:, :, 3]
                colors = ((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, (rgb) & 0xFF)
                colors = np.asarray(colors).T.reshape((-1, 3)) / 255.0

                # Update Open3D point cloud data
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # Update visualization
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            else:
                print("Failed to grab a frame.")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        vis.destroy_window()
        cam.close()


if __name__ == "__main__":
    main()
