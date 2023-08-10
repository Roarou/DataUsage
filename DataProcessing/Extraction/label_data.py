import os
from DataProcessing.Pointcloud.crop_pointcloud import process_point_cloud
from DataProcessing.Pointcloud.pointcloud_cleaning import clean
import concurrent.futures
import shutil
import time

base_path = r'G:\SpineDepth'  # Base directory
TIMEOUT = 600  # Timeout in seconds


def process_single_file(file_path, subdirectory_path, groundtruth_directory):
    """
    Process a single point cloud file, generate its groundtruth, and perform cleaning.

    Args:
    - file_path: Path to the input point cloud file.
    - subdirectory_path: Path to the subdirectory containing the point cloud.
    - groundtruth_directory: Path to the groundtruth directory.

    """
    filename = os.path.basename(file_path)
    if os.path.isfile(file_path) and filename.endswith('.pcd'):
        print(filename)
        idx = process_point_cloud(file_path, path_pose=subdirectory_path, gt_path=groundtruth_directory)
        filename_without_extension, extension = os.path.splitext(filename)
        groundtruth_filename = filename_without_extension + "_GT" + extension
        groundtruth_path = os.path.join(groundtruth_directory, groundtruth_filename)
        clean(groundtruth_path, idx)


def process_single_file_with_timeout(file_path, subdirectory_path, groundtruth_directory):
    """
    Process a single point cloud file with a timeout and restart mechanism.

    Args:
    - file_path: Path to the input point cloud file.
    - subdirectory_path: Path to the subdirectory containing the point cloud.
    - groundtruth_directory: Path to the groundtruth directory.

    """
    try:
        start_time = time.time()
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(process_single_file, file_path, subdirectory_path, groundtruth_directory)
                try:
                    result = future.result(timeout=TIMEOUT)
                    break
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    elapsed_time = time.time() - start_time
                    if elapsed_time > TIMEOUT:
                        print(f"Process for {file_path} timed out. Restarting...")
                        start_time = time.time()
    except Exception as e:
        print(f"An error occurred: {e}")


def process_video_directory(video_directory, subdirectory_path, groundtruth_directory):
    """
    Process all point cloud files in a video directory using threads.

    Args:
    - video_directory: Path to the video directory containing point cloud files.
    - subdirectory_path: Path to the subdirectory containing the point clouds.
    - groundtruth_directory: Path to the groundtruth directory.

    """
    list_pcd = os.listdir(video_directory)
    list_pcd = [filename for filename in list_pcd if filename.endswith('.pcd')]
    list_pcd = sorted(list_pcd, key=lambda x: int(x.split('_')[1].split('.')[0]))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_single_file_with_timeout,
                     [os.path.join(video_directory, filename) for filename in list_pcd],
                     [subdirectory_path] * len(list_pcd),
                     [groundtruth_directory] * len(list_pcd))


def launch_data():
    """
     Launch data processing for all specimens and recordings.

     """
    for i in range(1, 11):
        specimen_directory = f'Specimen_{i}'  # Create specimen directory name
        specimen_directory_path = os.path.join(base_path,
                                               specimen_directory)  # Create full path to the specimen directory
        # Create the path to the 'Recordings' directory inside the current specimen directory
        recordings_directory = 'Recordings'
        recordings_directory_path = os.path.join(specimen_directory_path, recordings_directory)
        # Get a list of all subdirectories in the 'Recordings' directory, sorted by recording number
        list_dir = os.listdir(recordings_directory_path)
        list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))

        # Process each subdirectory in the 'Recordings' directory
        for k, dir_name in enumerate(list_dir):
            subdirectory_path = os.path.join(recordings_directory_path, dir_name)
            pointcloud_directory = os.path.join(subdirectory_path, 'Pointcloud')

            for video_name in os.listdir(pointcloud_directory):
                video_directory = os.path.join(pointcloud_directory, video_name)
                groundtruth_directory = os.path.join(video_directory, 'Groundtruth')
                # Check if the groundtruth_directory exists
                if os.path.exists(groundtruth_directory):
                    # Remove the entire directory and its contents
                    shutil.rmtree(groundtruth_directory)
                os.makedirs(groundtruth_directory, exist_ok=True)
                process_video_directory(video_directory, subdirectory_path, groundtruth_directory)


if __name__ == "__main__":
    launch_data()
