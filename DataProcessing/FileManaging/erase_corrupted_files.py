# Import necessary libraries
import os

# Define the base directory for recordings
base_directory = r'L:\Specimen_10\Recordings'


def main():
    """This function organizes and cleans up point cloud recordings and associated ground truth files."""

    # Loop through recordings from 1 to 39
    for i in range(1, 40):
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

                # Loop through files in the video folder
                for filename in os.listdir(video_folder_path):
                    # Construct full file path
                    file_path = os.path.join(video_folder_path, filename)

                    # Check if the file is a '.pcd' point cloud file
                    if os.path.isfile(file_path) and filename.endswith('.pcd'):
                        print(file_path)

                        # Construct path for the associated ground truth file
                        groundtruth_path = os.path.join(video_folder_path, 'Groundtruth')
                        filename_without_extension, extension = os.path.splitext(filename)
                        groundtruth_filename = filename_without_extension + "_GT" + extension
                        gt_file_path = os.path.join(groundtruth_path, groundtruth_filename)

                        # If ground truth file is missing, delete the point cloud file
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


if __name__ == "__main__":
    # Execute the main function when the script is run
    main()
