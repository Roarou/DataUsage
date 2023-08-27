from DataProcessing.pose_transformation import PoseTransformation
import os
from DataProcessing.Pointcloud.pointcloud_cleaning import clean
from multiprocessing import Pool, cpu_count, TimeoutError


def process_point_cloud(args):
    """
    Process the point cloud data for a single frame.

    Parameters:
        args (tuple): A tuple containing file paths and frame information.
                     (file1, file2, poses_0_file_path, poses_1_file_path, i, frame)

    Returns:
        None
    """

    file1, file2, spec, frame, subdirectory_path = args

    # Process the point cloud data in the two files.
    # clean(file1, show_clusters=False)  # Clean the first file.
    # clean(file2, show_clusters=False)  # Clean the second file.

    # Process the point cloud data in the two files.
    pose_transformer = PoseTransformation(specimen=spec, frame=frame, file0=file2, file1=file1)

    # Visualize and get the transformation between the poses in the two pose files.
    TF_1, TF_2 = pose_transformer.visualize_displacement()

    # Load and downsample the point cloud data from the two files.
    pose_transformer.read_point_cloud()

    # Apply the transformation to the point cloud data.
    pose_transformer.apply_transformation(TF_1, TF_2)

    # Perform point-to-point Iterative Closest Point (ICP) registration.
    reg_p2p = pose_transformer.registration_icp()

    # Print the transformation result.
    print(reg_p2p.transformation)

    pointcloud_directory = os.path.join(subdirectory_path, 'merged_pointcloud')
    os.makedirs(pointcloud_directory, exist_ok=True)
    # Save the transformed and registered point clouds together into a file.
    path_f = os.path.join(pointcloud_directory, f"F_PCD_{frame}.pcd")  # Create a filename for the future combined
    # point cloud data file.
    pose_transformer.save_point_clouds_together(path_f, pose_transformer.pcd1, pose_transformer.pcd2)

    # Visualize the final combined and cleaned point cloud data.
    pose_transformer.visualize(path_f)


def main():
    """
    Main function to process point cloud data for all frames in multiple specimens.

    Parameters:
        None

    Returns:
        None
    """
    # Define the base directory path.
    base_path = r'G:\SpineDepth'

    # Loop through the numbers 1 to 10 to create paths for different specimens.
    for i in range(1, 11):
        specimen_directory = f'Specimen_{i}'  # Create the name of the specimen directory.
        specimen_path = os.path.join(base_path, specimen_directory)  # Create the path to the specimen directory.
        recordings_path = os.path.join(specimen_path,
                                       'Recordings')  # Create the path to the recordings directory within the specimen directory.

        # Check if the recordings directory exists.
        if not os.path.isdir(recordings_path):
            print(f"Recordings directory not found: {recordings_path}")
            return  # If the recordings directory doesn't exist, end the program.

        # Create a sorted list of all directories within the recordings directory.
        list_dir = os.listdir(recordings_path)
        list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))

        # Define file paths for two pose files and names for two directories.
        poses_0_file_path = os.path.join(recordings_path, "Poses_0.txt")
        poses_1_file_path = os.path.join(recordings_path, "Poses_1.txt")
        dirname_0 = 'Video_0'
        dirname_1 = 'Video_1'

        # For each directory in the list of directories.
        for dir_name in list_dir:
            subdirectory_path = os.path.join(recordings_path, dir_name)
            pointcloud_directory = os.path.join(subdirectory_path, '')

            # Check if pointcloud directory exists in the subdirectory.
            if not os.path.isdir(pointcloud_directory):
                print(f"Pointcloud directory not found: {pointcloud_directory}")
                continue  # If pointcloud directory doesn't exist, move to the next directory in the list.

            # Define the path to the first video directory.
            file_path = os.path.join(pointcloud_directory, dirname_0)

            # For each file in the first video directory.
            for filename in os.listdir(file_path):
                temp = os.path.join(file_path, filename)

                # Check if the current item is a .pcd file.
                if os.path.isfile(temp) and filename.endswith('.pcd'):
                    file1 = temp  # Store the path to the file.
                    frame = int(filename.split('_')[1].split('.')[0])  # Extract the frame number from the filename.
                    file2 = os.path.join(pointcloud_directory, dirname_1,
                                         filename)  # Create the path to the corresponding file in the second video
                    # directory.

                    print(f"Processing frame {frame}: {file1}, {file2}")  # Print the paths to the two files.
                    print(f"Processing specimen {i}, directory: {subdirectory_path}")
                    # Use multiprocessing to process point cloud data in parallel
                    # Create chunks of frames for each process
                    num_processes = cpu_count()
                    pool = Pool(processes=num_processes)
                    args = (file1, file2, poses_0_file_path, poses_1_file_path, i, frame, subdirectory_path)
                    result = pool.apply_async(process_point_cloud, args=(args,))

                    # Wait for the process to finish or timeout after 600 seconds.
                    try:
                        result.get(timeout=600)
                    except TimeoutError:
                        print("Process timed out. Restarting the pool.")
                        pool.terminate()
                        pool.join()
                        break  # Restart the pool for the current directory.

                    pool.close()
                    pool.join()


if __name__ == "__main__":
    main()
