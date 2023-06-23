import numpy as np
import open3d as o3d
from extract_tf_matrix import extract_transformation_matrices
from tqdm import tqdm

def visualize_displacement(transformation_data, stl_files):
    # Create an Open3D visualization window

    meshes=[]
    # Add each transformed mesh to the visualization
    for i in range(len(stl_files)):
        # Load the STL file for the corresponding vertebra
        mesh = o3d.io.read_triangle_mesh(stl_files[i])

        for j, (tr, rotation_matrix, translation_vector) in enumerate(transformation_data):
            # Apply the transformation to the mesh
            if j < 1:
                mesh.transform(tr)
        # Add the transformed mesh to the visualization
        mesh = mesh.sample_points_uniformly(number_of_points=100000)
        meshes.append(mesh)

    # Set visualization settings
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for mesh in meshes:
        print(1)
        vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.point_size = 10
    # Run the visualization loop
    vis.run()
    vis.destroy_window()


def main():

    # File path for the STR file containing transformation matrices
    poses_0_file_path = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"
    # poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"
    # File paths for the STL files of the vertebrae
    stl_files = [
        f"E:/Ghazi/STL/L{iteration}.stl" for iteration in range(1, 6)
    ]

    # Read the transformation matrices from the STR file
    transformation_data = extract_transformation_matrices(poses_0_file_path)

    # Visualize the displacement
    visualize_displacement(transformation_data, stl_files)


if __name__ == "__main__":
    main()
