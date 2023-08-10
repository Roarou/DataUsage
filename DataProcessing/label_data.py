import os
from DataProcessing.crop_pointcloud import process_point_cloud
from DataProcessing.Pointcloud.pointcloud_cleaning import clean
base_path = r'G:\SpineDepth'  # Base directory
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
            list_pcd = os.listdir(video_directory)
            list_pcd = [filename for filename in os.listdir(video_directory) if filename.endswith('.pcd')]
            list_pcd = sorted(list_pcd, key=lambda x: int(x.split('_')[1].split('.')[0]))
            print(list_pcd)
            groundtruth_directory = os.path.join(video_directory, 'Groundtruth')
            print(groundtruth_directory)
            os.makedirs(groundtruth_directory, exist_ok=True)
            for filename in list_pcd:
                file_path = os.path.join(video_directory, filename)
                if os.path.isfile(file_path) and filename.endswith('.pcd'):
                    file_path = os.path.join(video_directory, filename)
                    print(filename)
                    idx = process_point_cloud(file_path, subdirectory_path, gt_path=groundtruth_directory)
                    filename_without_extension, extension = os.path.splitext(filename)
                    groundtruth_filename = filename_without_extension + "_GT" + extension
                    groundtruth_path = os.path.join(groundtruth_directory, groundtruth_filename)
                    clean(groundtruth_path, idx)
