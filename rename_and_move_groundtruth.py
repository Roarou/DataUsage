# Import necessary libraries
import os
import open3d as o3d
import shutil

# Define the specimen name and base directory for recordings
spec = 'Specimen_2'
base_directory = f'G:\SpineDepth\{spec}\Recordings'


def main():
    """This function organizes and labels point cloud data files related to spine depth measurements."""

    # Loop through 40 recordings
    for i in range(40):
        # Generate the recording folder name
        recording_folder_name = f"Recording{i}"

        # Construct paths for recording and point cloud folders
        recording_folder_path = os.path.join(base_directory, recording_folder_name)
        pointcloud_folder_path = os.path.join(recording_folder_path, 'Pointcloud')

        # Print the current recording folder being processed
        print(recording_folder_name)

        # Check if the point cloud folder exists
        if os.path.exists(pointcloud_folder_path):
            # List of video folders within the point cloud folder
            video_folders = ['Video_0', 'Video_1']

            # Loop through each video folder
            for video_folder in video_folders:
                # Construct path for the current video folder
                video_folder_path = os.path.join(pointcloud_folder_path, video_folder)

                # Construct path for the 'Groundtruth' folder within the video folder
                groundtruth_path = os.path.join(video_folder_path, 'Groundtruth')

                # Loop through files in the 'Groundtruth' folder
                for filename in os.listdir(groundtruth_path):
                    # Construct full file path
                    file_path = os.path.join(groundtruth_path, filename)

                    # Check if the file is a '.pcd' point cloud file
                    if os.path.isfile(file_path) and filename.endswith('.pcd'):
                        # Extract file name without extension
                        filename_without_extension, extension = os.path.splitext(filename)

                        try:
                            # Specify the new path for the file
                            new_path = os.path.join(r'G:\SpineDepth\groundtruth_labeled', filename)

                            # Move the file to the new path
                            shutil.move(file_path, new_path)

                            # Print message indicating successful file renaming and moving
                            print(f"File '{file_path}' moved to {new_path}")
                        except FileNotFoundError:
                            # Handle case where the file is not found
                            print(f"File '{file_path}' not found.")
                        except Exception as e:
                            # Handle any other exceptions that may occur
                            print(f"An error occurred: {e}")
        else:
            print(f"Pointcloud folder in '{recording_folder_name}' not found")


if __name__ == "__main":
    main()
