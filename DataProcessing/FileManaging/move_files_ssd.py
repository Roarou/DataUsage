import os
import shutil
from tqdm import tqdm

# Define the source and destination directories
source_dir = 'L:\\groundtruth'
destination_dir = 'L:\\groundtruth_labeled'


def move_file(args):
    spec, filename = args
    if spec in filename:
        # Create the full path for the source and destination
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        shutil.move(source_path, destination_path)
        # print(f"Moved {filename} to {destination_path}")


if __name__ == "__main__":

    # List all files in the source directory
    files = os.listdir(source_dir)
    specs = ['Specimen_1', 'Specimen_3', 'Specimen_5', 'Specimen_6', 'Specimen_7', 'Specimen_9']

    # Prepare the list of arguments to pass to the move_file function
    args_list = [(spec, file) for spec in specs for file in files]

    # Use tqdm for the progress bar
    for args in tqdm(args_list, total=len(args_list)):
        move_file(args)
