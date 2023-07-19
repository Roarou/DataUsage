# Import necessary libraries
from Pointcloud.pointcloud_extraction import process_svo_file  # Import a function to process SVO files
import os  # Operating system functions

# Function to check if there's only one instance of 'CamParam' in the filenames in the given directory
def check_single_camparam(strings_list):
    camparam_count = 0  # Initialize the count of 'CamParam' occurrences
    b = None  # Initialize a variable
    for j, string in enumerate(strings_list):  # For each string in the list
        if string.startswith('CamParam'):  # If the string starts with 'CamParam'
            camparam_count += 1  # Increment the 'CamParam' count
            if camparam_count == 1:  # If this is the first 'CamParam'
                idx = j  # Save the index
        if camparam_count > 1:  # If there are more than one 'CamParam'
            _, _, b = strings_list[idx].split('_')  # Split the first 'CamParam' string on underscores and store the last part

    return camparam_count == 1, b  # Return whether there was only one 'CamParam' and the last part of the first 'CamParam' string

# Define some paths
base_path = r'G:\SpineDepth'  # Base directory
calibration_path = 'Calib'  # Calibration directory

# For each specimen directory
for i in range(1, 11):  # Range from 1 to 10 (inclusive)
    specimen_directory = f'Specimen_{i}'  # Create specimen directory name
    specimen_directory_path = os.path.join(base_path, specimen_directory)  # Create full path to the specimen directory

    # If the directory doesn't exist, exit the script
    if not os.path.isdir(specimen_directory_path):
        exit(1)

    print(f"Processing specimen directory: {specimen_directory_path}")  # Notify user that the specimen directory is being processed

    # Check for single 'CamParam' in the directory
    flag, b = check_single_camparam(os.listdir(specimen_directory_path))
    calibration_path = os.path.join(specimen_directory_path, calibration_path)

    # Create the path to the 'Recordings' directory inside the current specimen directory
    recordings_directory = 'Recordings'
    recordings_directory_path = os.path.join(specimen_directory_path, recordings_directory)

    # If the 'Recordings' directory doesn't exist, exit the script
    if not os.path.isdir(recordings_directory_path):
        exit(1)

    print(f"Processing recordings directory: {recordings_directory_path}")  # Notify user that the recordings directory is being processed

    pointcloud_count = 0
    camparams_folders = []

    # Get a list of all subdirectories in the 'Recordings' directory, sorted by recording number
    list_dir = os.listdir(recordings_directory_path)
    list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))

    # Process each subdirectory in the 'Recordings' directory
    for k, dir_name in enumerate(list_dir):
        subdirectory_path = os.path.join(recordings_directory_path, dir_name)
        print(subdirectory_path)
        conf_path = calibration_path  # Set the configuration file path
        # If there's more than one 'CamParam' in the directory
        if not flag:
            conf_path = calibration_path + '_a'  # Add '_a' to the configuration path
            if k > b:  # If this subdirectory's index is greater than the last part of the first 'CamParam'
                conf_path = calibration_path + '_b'  # Add '_b' to the configuration path
        # Create a 'pointcloud' directory inside the subdirectory
        pointcloud_directory = os.path.join(subdirectory_path, 'pointcloud')
        os.makedirs(pointcloud_directory, exist_ok=True)

        # Process each SVO file in the subdirectory
        folder_path = subdirectory_path
        i = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.svo'):
                # Determine the configuration file based on the SVO file name
                if filename == 'Video_0.svo':
                    conf_path = os.path.join(conf_path, 'SN10027879.conf')
                elif filename == 'Video_1.svo':
                    conf_path = os.path.join(conf_path, 'SN10028650.conf')

                # If the configuration file exists, process the SVO file
                if os.path.isfile(conf_path):
                    process_svo_file(file_path, conf_path, i, pointcloud_directory)
                    i = i + 1  # Increment the SVO file count

