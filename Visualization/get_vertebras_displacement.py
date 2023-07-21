import open3d as o3d
from Visualization.extract_tf_matrix_groundtruth import extract_transformation_matrices

stl_files_local = [
    f"E:/Ghazi/STL/L{iteration}.stl" for iteration in range(1, 6)
]


def visualize_displacements(pose_file, frame, stl_files=stl_files_local):

    # Read the transformation matrices from the STR file
    transformation_data = extract_transformation_matrices(pose_file)
    meshes = []

    # Add each transformed mesh to the visualization
    for i in range(len(stl_files)):
        # Load the STL file for the corresponding vertebra
        mesh = o3d.io.read_triangle_mesh(stl_files[i])
        # List of colors for each point cloud
        colors = [[1, 0, 0],  # Red
                  [0, 1, 0],  # Green
                  [0, 0, 1],  # Blue
                  [1, 1, 0],  # Yellow
                  [1, 0, 1]]  # Magenta
        # Assign a unique color to the mesh
        color = colors[i % len(colors)]  # Repeat colors if more meshes than colors
        mesh.paint_uniform_color(color)
        # at some point the formula will have to transform into j*5 and j will  correspond to the frame
        mesh.transform(transformation_data[frame*5 + i])
        # mesh.transform(transformation_data[i])
        # Add the transformed mesh to the visualization
        mesh = mesh.sample_points_uniformly(number_of_points=100000)
        meshes.append(mesh)
    return meshes, transformation_data[0]




"""
    # Do not forget to change this ! !! ! ! ! ! ! !! ! ! ! !! ! ! ! ! !! !  ! ! ! ! !! ! ! !
    # pcd = o3d.io.read_point_cloud("../test_0.pcd")
    # meshes.append(pcd)
    # Uncomment this
    # pcd2 = o3d.io.read_point_cloud("../test_1.pcd")
    # meshes.append(pcd2)
    # Set visualization settings
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for mesh in meshes:
        print(1)
        vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.point_size = 3
    # Run the visualization loop
    vis.run()
    vis.destroy_window()
    """


def main():
    # File path for the STR file containing transformation matrices
    poses_0_file_path = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"
    # poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"
    # File paths for the STL files of the vertebrae

    # Visualize the displacement
    visualize_displacements(poses_0_file_path)


if __name__ == "__main__":
    main()
