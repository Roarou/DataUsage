import open3d as o3d

# Define the file paths for the STL files
stl_file_paths = ["E:/Ghazi/STL/L1.stl", "E:/Ghazi/STL/L2.stl"]

# Create a list to store the meshes
meshes = []

# Load the meshes from the STL files and assign colors
for i, stl_file_path in enumerate(stl_file_paths):
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh = mesh.sample_points_uniformly(number_of_points=100000)
    color = [0, 0, 0]  # Default color is black

    # Assign a unique color to each mesh
    if i == 0:
        color = [1, 0, 0]  # Red color for the first mesh
    elif i == 1:
        color = [0, 1, 0]  # Green color for the second mesh

    # Assign the color to the mesh
    mesh.paint_uniform_color(color)

    # Subdivide the mesh to increase resolution

    meshes.append(mesh)

# Create the visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the meshes to the visualization
for mesh in meshes:
    vis.add_geometry(mesh)

# Set the point size and run the visualization
render_option = vis.get_render_option()
render_option.point_size = 10
vis.run()
vis.destroy_window()
"""
mesh = mesh.sample_points_uniformly(number_of_points=100000)
mesh2 = mesh2.sample_points_uniformly(number_of_points=100000)
points1 = np.asarray(mesh.points)
points2 = np.asarray(mesh2.points)
translation = np.array([100.0, 0.0, 0.0])
points2 = points2+translation
merged_points = np.concatenate((points1, points2), axis=0)
merged_point_cloud = o3d.geometry.PointCloud()
merged_point_cloud.points = o3d.utility.Vector3dVector(
    merged_points
)
"""