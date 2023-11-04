import os
import shutil

# Base paths for source and destination
source_base_path = 'J:'
destination_base_path = 'G:\SpineDepth'

# Loop through each Specimen_{I}
for i in range(2, 6):
    specimen_folder = f"Specimen_{i}"
    recordings_path = os.path.join(source_base_path, specimen_folder)

    # Check if the Recordings path exists (for safety)
    if not os.path.exists(recordings_path):
        print(f"Recordings path not found for Specimen_{i}")
        continue

    # Loop through each Recording{I}
    for j in range(40):
        recording_folder = f"recording{j}"
        rec_f = f"Recording{j}"
        source_recording_path = os.path.join(recordings_path, recording_folder)

        # Check if the Recording path exists (for safety)
        if not os.path.exists(source_recording_path):
            print(f"Recording path not found for Specimen_{i}, Recording{j}")
            continue

        # Define destination path
        destination_recording_path = os.path.join(destination_base_path, specimen_folder, "Recordings",
                                                  rec_f)
        path = os.path.join(destination_recording_path,'pointcloud')

        try:
            print('Deleting folder')
            shutil.rmtree(path)
            print(f"Folder at path {path} has been successfully deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {str(e)}")

        # Define file paths
        file1_path = os.path.join(source_recording_path, "Video_0.svo")
        file2_path = os.path.join(source_recording_path, "Video_1.svo")

        # Move files
        if os.path.exists(file1_path):
            print(f'Copying {file1_path}...')
            shutil.copy(file1_path, os.path.join(destination_recording_path, "Video_0.svo"))
        if os.path.exists(file2_path):
            print(f'Copying {file2_path}...')
            shutil.copy(file2_path, os.path.join(destination_recording_path, "Video_1.svo"))

print("Files moved successfully!")
