import os

base_path = r'G:\SpineDepth'  # Base directory
from crop_pointcloud import process_point_cloud

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
        pointcloud_directory = os.path.join(subdirectory_path, 'pointcloud')
        print(dir_name)
        video_directory = os.path.join(pointcloud_directory, 'Video_1')
        file_path = os.path.join(video_directory, 'Pointcloud_0.pcd')
        idx = process_point_cloud(file_path, subdirectory_path)