import os
import open3d as o3d

base_directory = r'G:\SpineDepth\Specimen_2\Recordings'

for i in range(1,40):
    recording_folder_name = f"Recording{i}"

    recording_folder_path = os.path.join(base_directory, recording_folder_name)
    pointcloud_folder_path = os.path.join(recording_folder_path, 'Pointcloud')
    print(recording_folder_name)
    if os.path.exists(pointcloud_folder_path):
        video_folders = ['Video_0', 'Video_1']

        for video_folder in video_folders:
            video_folder_path = os.path.join(pointcloud_folder_path, video_folder)
            for filename in os.listdir(video_folder_path):
                file_path = os.path.join(video_folder_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.pcd'):
                    groundtruth_path = os.path.join(video_folder_path, 'Groundtruth')
                    filename_without_extension, extension = os.path.splitext(filename)
                    groundtruth_filename = filename_without_extension + "_GT" + extension
                    gt_file_path = os.path.join(groundtruth_path, groundtruth_filename)
                    if not os.path.isfile(gt_file_path):
                        try:
                            os.remove(file_path)
                            print(f"File '{file_path}' has been deleted successfully.")
                        except FileNotFoundError:
                            print(f"File '{file_path}' not found.")
                        except Exception as e:
                            print(f"An error occurred: {e}")


    else:
        print(f"Pointcloud folder in '{recording_folder_name}' not found")
