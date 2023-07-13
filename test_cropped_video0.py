import open3d as o3d
import numpy as np
import inspect
import matplotlib.pyplot as plt
def run1():
    pcd = o3d.io.read_point_cloud("Pointcloud/Video_0/Pointcloud_0.pcd")
    #o3d.visualization.draw_geometries([pcd])
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Extract max and min coordinates
    max_coords = np.max(points, axis=0)
    min_coords = np.min(points, axis=0)
    print("Maximum coordinates:", max_coords)
    print("Minimum coordinates:", min_coords)

    # Create a new point with custom coordinates, size, and color
    new_point = np.array([[250.0, 5.0, 750.0]])  # Set the desired coordinates [x, y, z]
    new_point_size = 10000.0  # Set the desired point size
    new_point_color = [1.0, 0, 0.0]  # Set the desired color [R, G, B]

    # Add the new point to the point cloud
    #pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points, new_point]))
    #pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors, np.tile(new_point_color, (1, 1))]))

    # Define the bounding box
    min_bound = [-200, -50, 600]  # Minimum coordinates of the bounding box
    max_bound = [250, 200, 800]     # Maximum coordinates of the bounding box

    # Crop the point cloud
    cropped_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound))

    print(cropped_pcd)
    test = cropped_pcd.get_axis_aligned_bounding_box()
    # print(inspect.getmembers(test))

    test.color = (0, 1, 0)
    # Create a coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[250, 5, 750])
    output_path = "test_0.pcd"
    # Visualize the point cloud with the coordinate frame
    # o3d.visualization.draw_geometries([cropped_pcd, coord_frame, test])
    o3d.io.write_point_cloud(output_path, cropped_pcd)


run1()

