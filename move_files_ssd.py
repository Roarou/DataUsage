
import os
import shutil

# Define the source and destination directories
source_dir = 'L:\\groundtruth_labeled'
destination_dir = 'L:\\groundtruth'

# List all files in the source directory
files = os.listdir(source_dir)
for file in files:
    if 'Specimen_9' in file:
        # Create the full path for the source and destination
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)
        print(f"Moved {file} to {destination_path}")


