import numpy as np


def extract_transformation_matrices(file_path):
    """
    Extract transformation matrices from the specified file.

    Parameters:
        file_path (str): File path of the file containing transformation matrices.

    Returns:
        list: List of transformation matrices (4x4) for each vertebra in each frame.
        list: List of rotation matrices (3x3) for each vertebra in each frame.
        list: List of translation vectors (3x1) for each vertebra in each frame.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_frames = len(lines) // 5
    num_vertices = 5

    transformation_matrices = []
    rotation = []
    translation = []
    for frame_idx in range(num_frames):
        frame_start_idx = frame_idx * 5

        for vertebra_idx in range(num_vertices):
            vertebra_start_idx = frame_start_idx + vertebra_idx
            transformation_line = lines[vertebra_start_idx].strip()
            transformation_values = list(map(float, transformation_line.split(',')))
            transformation_matrix = np.array(transformation_values).reshape(4, 4)
            transformation_matrices.append(transformation_matrix)
            rotation.append(transformation_matrix[:3, :3])
            translation.append(transformation_matrix[:3, 3])
    #return zip(transformation_matrices, rotation, translation)
    return transformation_matrices


def main():
    poses_0_file_path = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"
    poses_1_file_path = "E:/Ghazi/Recordings/Recording0/Poses_1.txt"

    transformation_matrices_0, r1, t1 = extract_transformation_matrices(poses_0_file_path)
    transformation_matrices_1, r2, r2 = extract_transformation_matrices(poses_1_file_path)

    # Example usage: Print the transformation matrices for each vertebra in the first frame
    for vertebra_idx in range(5):
        matrix_0 = transformation_matrices_0[vertebra_idx]
        matrix_1 = transformation_matrices_1[vertebra_idx]
        print(f"Vertebra {vertebra_idx + 1} - Sensor 0:\n{matrix_0}\n")
        print(r1[vertebra_idx], t1[vertebra_idx])
        print(f"Vertebra {vertebra_idx + 1} - Sensor 1:\n{matrix_1}\n")


if __name__ == "__main__":
    main()
