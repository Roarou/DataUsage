from Visualization.pose_transformation import PoseTransformation
import os
from Pointcloud.pointcloud_cleaning import clean

def main():
    # File and directory paths
    # Define some paths
    base_path = r'G:\SpineDepth'  # Base directory
    specimen_path = os.path.join(base_path, 'Specimen_1') # Specimen, Chnage number
    recordings_path = os.path.join(specimen_path, 'Recordings')

    # Check if recordings directory exists
    if not os.path.isdir(recordings_path):
        print(f"Recordings directory not found: {recordings_path}")
        return

    list_dir = os.listdir(recordings_path)
    list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))
    poses_0_file_path = "Poses_0.txt"
    poses_1_file_path = "Poses_1.txt"
    dirname_0 = 'Video_0'
    dirname_1 = 'Video_1'
    for dir_name in list_dir:
        subdirectory_path = os.path.join(recordings_path, dir_name)
        poses_0_file_path = os.path.join(recordings_path, poses_0_file_path)
        poses_1_file_path = os.path.join(recordings_path, poses_1_file_path)
        pointcloud_directory = os.path.join(subdirectory_path, 'pointcloud')

        # Check if pointcloud directory exists
        if not os.path.isdir(pointcloud_directory):
            print(f"Pointcloud directory not found: {pointcloud_directory}")
            continue
        file_path = os.path.join(pointcloud_directory, dirname_0)
        for filename in os.listdir(file_path):
            temp = os.path.join(file_path, filename)
            if os.path.isfile(temp) and filename.endswith('.pcd'):
                file1 = temp
                file2 = os.path.join(pointcloud_directory, dirname_1)
                file2 = os.path.join(file2, filename)
                print(file1, file2)
        print(subdirectory_path)
# create final path
    path1 = "E:/Ghazi/CamParams_0_31/SN10027879.conf"
    path2 = "E:/Ghazi/CamParams_0_31/SN10028650.conf"
    path_f = "final_combined_pcd.pcd"
    # Clean the coarse data
    clean(file1, show_clusters=False)
    clean(file2, show_clusters=False)

    # Initialize the PoseTransformation object
    pose_transformer = PoseTransformation(path1, path2)

    # Visualize the displacement between poses
    TF_1, TF_2 = pose_transformer.visualize_displacement(poses_0_file_path, poses_1_file_path)

    # Read and downsample the point cloud data
    pose_transformer.read_point_cloud(file1, file2)

    # Apply the transformation to the point cloud data
    pose_transformer.apply_transformation(TF_1, TF_2)

    # Apply ICP registration
    reg_p2p = pose_transformer.registration_icp()

    # Print the transformation result
    print(reg_p2p.transformation)

    # Save the transformed and registered point clouds together
    pose_transformer.save_point_clouds_together(path_f, pose_transformer.pcd1, pose_transformer.pcd2)
    clean(path_f, factor=5, rad=2, show_clusters=False, reconstruction=True)

    # Visualize the final result
    pose_transformer.visualize(path_f)


if __name__ == "__main__":
    main()
