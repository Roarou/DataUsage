import os
import shutil

# Base paths for source and destination
source_base_path = 'D:\Ghazi'
destination_base_path = 'M:\\SpineDepth'

# Loop through each Specimen_{I}
for i in range(1,2):
    specimen_folder = f"Specimen_{i}"
    recordings_path = os.path.join(source_base_path, specimen_folder, "Recordings")

    # Check if the Recordings path exists (for safety)
    if not os.path.exists(recordings_path):
        print(f"Recordings path not found for Specimen_{i}")
        continue

    # Loop through each Recording{I}
    for j in range(40):
        recording_folder = f"Recording{j}"
        rec_f=f"recording{j}"
        source_recording_path = os.path.join(recordings_path, recording_folder)

        # Check if the Recording path exists (for safety)
        if not os.path.exists(source_recording_path):
            print(f"Recording path not found for Specimen_{i}, Recording{j}")
            continue

        # Define destination path
        destination_recording_path = os.path.join(destination_base_path, specimen_folder, "recordings",
                                                  rec_f)

        # Create destination folder if it doesn't exist
        os.makedirs(destination_recording_path, exist_ok=True)

        # Define file paths
        file1_path = os.path.join(source_recording_path, "Poses_0.txt")
        file2_path = os.path.join(source_recording_path, "Poses_1.txt")

        # Move files
        if os.path.exists(file1_path):
            shutil.copy(file1_path, os.path.join(destination_recording_path, "Poses_0.txt"))
        if os.path.exists(file2_path):
            shutil.copy(file2_path, os.path.join(destination_recording_path, "Poses_1.txt"))

print("Files moved successfully!")
