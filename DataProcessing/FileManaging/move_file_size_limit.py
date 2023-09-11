import os
import shutil
from tqdm import tqdm

# Define the source and destination directories
source_dir = 'G:\\SpineDepth\\groundtruth_labeled'
destination_dir = 'L:\\groundtruth'

# Define the available space in GB
available_space_gb = 340
available_space_bytes = available_space_gb * 1024 ** 3

# Get the total space used in the destination directory
total_space_used = 0  # sum(os.path.getsize(os.path.join(destination_dir, f)) for f in os.listdir(destination_dir))

# List all files in the source directory
files = os.listdir(source_dir)
progress_bar = tqdm(files, desc="Moving files")
# Iterate through the files and move them if there is available space
for file in progress_bar:
    source_path = os.path.join(source_dir, file)
    file_size = os.path.getsize(source_path)

    # Check if there is enough space left in the destination directory
    if total_space_used + file_size <= available_space_bytes:
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)
        # print(f"Moved {file} to {destination_path}")
        total_space_used += file_size
    else:
        print(f"Reached the space limit, unable to move {file}")
        break
