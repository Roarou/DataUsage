import os
import open3d as o3d
import shutil

spec = 'Specimen_4'
base_directory = f'G:\SpineDepth\{spec}\Recordings'

for i in range(40):
    recording_folder_name = f"Recording{i}"

    recording_folder_path = os.path.join(base_directory, recording_folder_name)
    pointcloud_folder_path = os.path.join(recording_folder_path, 'Pointcloud')
    print(recording_folder_name)
    if os.path.exists(pointcloud_folder_path):
        video_folders = ['Video_0', 'Video_1']

        for video_folder in video_folders:
            video_folder_path = os.path.join(pointcloud_folder_path, video_folder)
            groundtruth_path = os.path.join(video_folder_path, 'Groundtruth')
            for filename in os.listdir(groundtruth_path):
                file_path = os.path.join(groundtruth_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.pcd'):
                    filename_without_extension, extension = os.path.splitext(filename)
                    try:
                        # shutil.move(file_path, os.path.join(r'G:\SpineDepth\groundtruth_labeled',filename))
                        new_filename = f"{filename_without_extension}_{spec}_{recording_folder_name}_{video_folder}.pcd"
                        new_path = os.path.join(groundtruth_path, new_filename)
                        os.rename(file_path, new_path)
                        print(f"File '{file_path}' renamed into {new_path}")
                    except FileNotFoundError:
                        print(f"File '{file_path}' not found.")
                    except Exception as e:
                        print(f"An error occurred: {e}")


    else:
        print(f"Pointcloud folder in '{recording_folder_name}' not found")
