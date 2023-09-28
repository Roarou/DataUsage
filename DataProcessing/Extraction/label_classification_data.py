import os
from DataProcessing.Pointcloud.crop_vertebrae import process_point_cloud
import multiprocessing
import shutil
import time
from tqdm import tqdm
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
        #        print(filename)
        _, SUCCESS = process_point_cloud(file_path, path_pose=subdirectory_path, gt_path=groundtruth_directory)
        #        print(SUCCESS)


def process_single_file_with_timeout(args):
    """
    Process a single point cloud file with a timeout and restart mechanism.

    Args:
    - args: Tuple containing (file_path, subdirectory_path, groundtruth_directory).
    """
    file_path, subdirectory_path, groundtruth_directory = args
    try:
        start_time = time.time()
        while True:
            try:
                process_single_file(file_path, subdirectory_path, groundtruth_directory)
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                elapsed_time = time.time() - start_time
                if elapsed_time > TIMEOUT:
                    print(f"Process for {file_path} timed out. Restarting...")
                    start_time = time.time()
    except Exception as e:
        print(f"An error occurred: {e}")


def process_video_directory(video_directory, subdirectory_path, groundtruth_directory):
    """
    Process all point cloud files in a video directory using multiprocessing.

    Args:
    - video_directory: Path to the video directory containing point cloud files.
    - subdirectory_path: Path to the subdirectory containing the point clouds.
    - groundtruth_directory: Path to the groundtruth directory.
    """
    list_pcd = os.listdir(video_directory)
    list_pcd = [filename for filename in list_pcd if filename.endswith('.pcd')]

    try:
        list_pcd = sorted(list_pcd, key=lambda x: int(x.split('_')[1].split('.')[0]))
    except Exception as e:
        print(f"An error occurred: {e}")
        print(list_pcd)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_single_file_with_timeout,
                 [(os.path.join(video_directory, filename), subdirectory_path, groundtruth_directory) for filename in
                  list_pcd])


def launch_data():
    """
    Launch data processing for all specimens and recordings.
    """
    groundtruth_directory = r'L:\groundtruth'
    for i in range(2, 11):
        specimen_directory = f'Specimen_{i}'  # Create specimen directory name
        specimen_directory_path = os.path.join(base_path, specimen_directory)
        recordings_directory = 'Recordings'
        recordings_directory_path = os.path.join(specimen_directory_path, recordings_directory)
        list_dir = os.listdir(recordings_directory_path)
        list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))

        for k, dir_name in enumerate(list_dir):
            subdirectory_path = os.path.join(recordings_directory_path, dir_name)
            pointcloud_directory = os.path.join(subdirectory_path, 'Pointcloud')

            for video_name in os.listdir(pointcloud_directory):
                video_directory = os.path.join(pointcloud_directory, video_name)

                process_video_directory(video_directory, subdirectory_path, groundtruth_directory)



if __name__ == "__main__":
    launch_data()
