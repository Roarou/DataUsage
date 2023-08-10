import os
from DataProcessing.Pointcloud.crop_pointcloud import process_point_cloud
from DataProcessing.Pointcloud.pointcloud_cleaning import clean
import multiprocessing

base_path = r'G:\SpineDepth'  # Base directory


def process_single_file(file_path, subdirectory_path, groundtruth_directory):
    filename = os.path.basename(file_path)
    if os.path.isfile(file_path) and filename.endswith('.pcd'):
        print(filename)
        idx = process_point_cloud(file_path, path_pose=subdirectory_path, gt_path=groundtruth_directory)
        filename_without_extension, extension = os.path.splitext(filename)
        groundtruth_filename = filename_without_extension + "_GT" + extension
        groundtruth_path = os.path.join(groundtruth_directory, groundtruth_filename)
        clean(groundtruth_path, idx)


def process_video_directory(video_directory, subdirectory_path, groundtruth_directory):
    list_pcd = os.listdir(video_directory)
    list_pcd = [filename for filename in list_pcd if filename.endswith('.pcd')]
    list_pcd = sorted(list_pcd, key=lambda x: int(x.split('_')[1].split('.')[0]))
    with multiprocessing.Pool() as pool:
        pool.starmap(process_single_file,
                     [(os.path.join(video_directory, filename), subdirectory_path, groundtruth_directory) for filename in
                      list_pcd])


def launch_data():
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
                os.makedirs(groundtruth_directory, exist_ok=True)
                process_video_directory(video_directory, subdirectory_path, groundtruth_directory)


if __name__ == "__main__":
    launch_data()
