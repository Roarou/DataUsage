import torch
from Model.pointnet_multi_class.spine_multi_class_segmentation import FastNet
import os
import numpy as np
import open3d as o3d
from Model.pointnet_multi_class.load_dataset_multi import normalize_point_cloud
import copy
import pandas as pd

colors = {
    (1, 0, 0),  # L1
    (0, 1, 0),  # L2
    (0, 0, 1),  # L3
    (0, 1, 1),  # L4
    (1, 0, 1)
}


def map_label_to_color(color):
    mapping = {
        0: [1, 0, 0],  # L1
        1: [0, 1, 0],  # L2
        2: [0, 0, 1],  # L3
        3: [0, 1, 1],  # L4
        4: [1, 0, 1],  # L5
        5: [0, 0, 0]  # Non-spine
    }
    res = mapping.get(color, -1)
    return res  # -1 for any unexpected colors


def create_transformation_matrix(df_row):
    """
    Creates a transformation matrix from a DataFrame row containing rotation and translation data.

    :param df_row: A Pandas Series with rotation matrix R and translation vector T components.
    :return: A 4x4 NumPy array representing the transformation matrix.
    """
    # Extract the rotation and translation values to construct the transformation matrix
    return np.array([
        [df_row["R00"], df_row["R01"], df_row["R02"], df_row["T0"]],
        [df_row["R10"], df_row["R11"], df_row["R12"], df_row["T1"]],
        [df_row["R20"], df_row["R21"], df_row["R22"], df_row["T2"]],
        [0, 0, 0, 1]  # The bottom row of a transformation matrix is always [0, 0, 0, 1]
    ])


def read_and_transform_vertebra(base_path, specimen, vertebra_name, transformation):
    """
    Reads a vertebra model from an STL file, computes its vertex normals, and applies a transformation.

    :param base_path: The base path to the specimen data.
    :param specimen: The specimen identifier.
    :param vertebra_name: The name of the vertebra to be transformed.
    :param transformation: The transformation matrix to be applied.
    :return: The transformed vertebra mesh.
    """
    # Read the STL file into an Open3D triangle mesh object
    vertebra = o3d.io.read_triangle_mesh(os.path.join(base_path, specimen, "STL", f"{vertebra_name}.stl"))
    # Compute the normals for the vertices of the mesh
    vertebra.compute_vertex_normals()
    # Apply the transformation to the vertebra mesh
    vertebra.transform(transformation)
    return vertebra


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


#


def load_model(model_path):
    """
    Load the model from the given path.
    """
    model = FastNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state_dict'])
    return model


def process_point_cloud(path, nb_sample=100e3):
    pcd = o3d.io.read_point_cloud(path)
    input_points = np.asarray(pcd.points)
    input_colors = np.asarray(pcd.colors)
    sampled_indices = np.random.choice(len(input_points), nb_sample, replace=False)
    point = input_points[sampled_indices]
    down_col = input_colors[sampled_indices]
    return point, down_col


def load_and_process_data(model_path, source_dir, tracking_file, pcl_path, specimen, cur_frame, nb_sample=100e3):
    """
    Load and process the data required for point cloud visualization.
    """
    # Load the transformation matrix
    df = pd.read_csv(tracking_file, sep=',', header=None,
                     names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2",
                            "R30", "R31", "R33", "T3"])
    vertebrae = [read_and_transform_vertebra(source_dir, specimen, f"L{i + 1}",
                                             create_transformation_matrix(df.iloc[5 * cur_frame + i]))
                 for i in range(5)]

    # Load model and point cloud
    model = load_model(model_path)
    model.eval()
    point, down_col = process_point_cloud(pcl_path, nb_sample=nb_sample)

    return model, vertebrae, point, down_col


def predict_and_visualize(model, point, down_col, vertebrae, display_stl=True):
    """
    Perform predictions and visualize point cloud data.
    """
    downscale = process_downscale(point)
    predictions = model_predict(model, downscale)
    pcd_gt, pcd_pred = generate_point_clouds(predictions, point, down_col, vertebrae)

    # Visualization
    if display_stl:
        visualize_point_clouds(pcd_gt, pcd_pred)
    return pcd_gt, pcd_pred


def create_individual_point_cloud(predictions, point, down_col, index):
    """
    Create individual point clouds for both original and predicted data.
    """
    # Create point cloud for original data
    pcd_or = o3d.geometry.PointCloud()
    pcd_or.points = o3d.utility.Vector3dVector(point)
    pcd_or.colors = o3d.utility.Vector3dVector(down_col)

    # Find indices for the current prediction label
    label_indices = np.where(predictions == index)[0]

    # Extract corresponding points and colors for predicted data
    labeled_points = point[label_indices]
    labeled_colors = down_col[label_indices]

    # Create point cloud for predicted data
    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(labeled_points)
    pcd_modified.colors = o3d.utility.Vector3dVector(labeled_colors)
    o3d.visualization.draw_geometries([pcd_modified])

    return pcd_or, pcd_modified


def generate_point_clouds(predictions, point, down_col, vertebrae):
    """
    Generate point clouds for ground truth and predictions.
    """
    # Generate point clouds
    pcd_gt = []
    pcd_pred = []
    for i in range(5):
        pcd_or, pcd_modified = create_individual_point_cloud(predictions, point, down_col, i)
        pcd_gt.extend([pcd_or, vertebrae[i]])
        pcd_pred.extend([pcd_modified, vertebrae[i]])
    return pcd_gt, pcd_pred


def process_downscale(point):
    """
    Process the point cloud data for downscaling.
    """
    # Processing steps
    downscale = copy.deepcopy(point)
    downscale = normalize_point_cloud(downscale)
    downscale = torch.tensor(downscale, dtype=torch.float32).unsqueeze(0).float().transpose(2, 1)
    return downscale


def visualize_point_clouds(pcd_gt, pcd_pred):
    """
    Visualize the point clouds.
    """
    # Visualizing the point clouds
    o3d.visualization.draw_geometries(pcd_gt)
    o3d.visualization.draw_geometries(pcd_pred)
    # pcd_modified = o3d.geometry.PointCloud()
    # pcd_modified.points = o3d.utility.Vector3dVector(point)
    # pcd_modified.colors = o3d.utility.Vector3dVector(predictions)
    #
    # o3d.visualization.draw_geometries([pcd_modified])


def model_predict(model, downscale):
    """
    Perform model prediction.
    """
    with torch.no_grad():
        predictions, _ = model(downscale)
    predictions = torch.argmax(predictions.cpu()[0], dim=-1).numpy()
    return predictions


# Main execution function
def main():
    """
    Main function to execute the process.
    """
    # Define file paths and model paths
    model_path = r'C:\Users\cheg\PycharmProjects\DataUsage\log\sem_seg\2023-11-15_02-34\checkpoints\best_model.pth'
    source_dir = r"G:\SpineDepth"
    specimen = "Specimen_2"
    cur_frame = 0
    pcd_path = r'L:\Pointcloud\Pointcloud_{}_spec_{}_vid_0_Recording0.pcd'.format(cur_frame, specimen)
    recording = 'Recording0'
    camera_num = 0
    tracking_file = os.path.join(source_dir, specimen, f"Recordings/{recording}/Poses_{camera_num}.txt")

    # Load and process data
    model, vertebrae, point, down_col = load_and_process_data(model_path, source_dir, tracking_file, pcd_path, specimen,
                                                              cur_frame, nb_sample=100000)
    # Perform prediction and visualization
    GT, pred = predict_and_visualize(model, point, down_col, vertebrae)

    try:

        # Convert the list to a NumPy array
        my_array = np.array(pred)

        # Find indices where the value is 1

        L1 = np.where(my_array == 0)[0]
        L2 = np.where(my_array == 1)[0]
        L3 = np.where(my_array == 2)[0]
        L4 = np.where(my_array == 3)[0]
        L5 = np.where(my_array == 4)[0]
        idx = [L1, L2, L3, L4, L5]
        predictions = np.array([map_label_to_color(c) for c in pred])
        pcd_gt = []
        pcd_pred = []
        for i, color in enumerate(colors):
            matches = np.all(down_col == color, axis=1)

            # Find indices where matches are True
            L1 = np.where(matches)[0]
            L1_col = down_col[L1]
            L1_point = point[L1]
            pcd_or = o3d.geometry.PointCloud()
            pcd_or.points = o3d.utility.Vector3dVector(L1_point)
            pcd_or.colors = o3d.utility.Vector3dVector(L1_col)
            print(len(pcd_or.points))
            # Compute the bounds of the first point cloud
            min_bound = pcd_or.get_min_bound()
            max_bound = pcd_or.get_max_bound()

            # Calculate the amount to translate
            extra_space = 10
            # Here, we shift the second point cloud to the right of the first
            # You might want to add some extra space between them
            shift = max_bound[0] - min_bound[0] + extra_space
            # pcd_or.paint_uniform_color([0.5, 0.5, 0.5])

            if i == 1:
                i = 4
            elif i == 2:
                i = 1
            elif i == 3:
                i = 2
            elif i == 4:
                i = 3

            L1_col = predictions[idx[i]]
            L1_point = point[idx[i]]
            pcd_modified = o3d.geometry.PointCloud()
            pcd_modified.points = o3d.utility.Vector3dVector(L1_point)
            pcd_modified.colors = o3d.utility.Vector3dVector(L1_col)

            pcd_f = o3d.geometry.PointCloud()
            pcd_f = pcd_modified + pcd_or
            pcd_gt.append(pcd_or)
            pcd_pred.append(pcd_modified)
            pcd_gt.append(vertebrae[i])
            pcd_pred.append(vertebrae[i])
            # Translate the second point cloud
            # pcd_modified.translate((shift, 0, 0))
            o3d.visualization.draw_geometries([pcd_modified, vertebrae[i]])
            curr_dir = os.getcwd()
            filename = f'L{i + 1}_fused.pcd'
            print(filename)
            final_path = os.path.join(curr_dir, filename)
            SUCCESS = o3d.io.write_point_cloud(final_path, pcd_f)



    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
