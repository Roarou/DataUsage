import os
import shutil
import numpy as np
from tqdm import tqdm

source_folder = 'L:\\groundtruth'
destination_folder = 'L:\\backup_files'

nb_files_to_move = 80000
# List all files in the source folder
all_files = os.listdir(source_folder)
total_files = len(all_files)
pbar = tqdm(total=nb_files_to_move)

# If there are fewer than 100,000 files, move them all
if total_files <= nb_files_to_move:
    for filename in all_files:
        shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))
        pbar.update(1)

else:
    # Otherwise, uniformly select and move 100,000 files
    selected_indices = np.linspace(0, total_files - 1, nb_files_to_move, dtype=int)

    for i in selected_indices:
        filename = all_files[i]
        shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, filename))
        pbar.update(1)

pbar.close()