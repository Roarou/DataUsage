
import os
import shutil

# Define the source and destination directories
source_dir = 'L:\\groundtruth'
destination_dir = 'L:\\groundtruth_labeled'

# List all files in the source directory
specs = ['Specimen_1', 'Specimen_3', 'Specimen_5', 'Specimen_6', 'Specimen_7', 'Specimen_9']
for spec in specs:
    files = os.listdir(source_dir)
    for file in files:
        if spec in file:
            # Create the full path for the source and destination
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)
            print(f"Moved {file} to {destination_path}")


