import os
import shutil
import numpy as np
from tqdm import tqdm

source_folder = 'L:\\groundtruth_labeled'
destination_folder = 'L:\\backup_files'

# List all files in the source folder
all_files = os.listdir(source_folder)
total_files = len(all_files)
pbar = tqdm(total=100000)

# If there are fewer than 100,000 files, move them all
if total_files <= 100000:
    for filename in all_files:
        shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))
        pbar.update(1)

else:
    # Otherwise, uniformly select and move 100,000 files
    selected_indices = np.linspace(0, total_files - 1, 100000, dtype=int)

    for i in selected_indices:
        filename = all_files[i]
        shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))
        pbar.update(1)

pbar.close()