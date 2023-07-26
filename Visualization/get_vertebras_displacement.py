import open3d as o3d
from Visualization.extract_tf_matrix_groundtruth import extract_transformation_matrices


def visualize_displacements(pose_file, frame, specimen):
    """
    Visualize the displacements of vertebrae based on transformation matrices.

    Parameters:
        pose_file (str): File path for the STR file containing transformation matrices.
        frame (int): Frame number for visualization.
        specimen (int): Specimen number for visualization.

    Returns:
        list: List of Open3D meshes representing the transformed vertebrae.
        ndarray: Transformation matrix of the first vertebra in the list.
    """
    stl_files = [
        f"G:/SpineDepth/Specimen_{specimen}/STL/L{iteration}.stl" for iteration in range(1, 6)
    ]
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


def main():
    # File path for the STR file containing transformation matrices
    poses_0_file_path = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"
    # poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"
    # File paths for the STL files of the vertebrae

    # Visualize the displacement
    visualize_displacements(poses_0_file_path)


if __name__ == "__main__":
    main()
