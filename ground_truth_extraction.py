from Pointcloud.pointcloud_extraction import process_svo_file

import os

base_path = r'G:\SpineDepth'
for i in range(1, 11):  # Range from 1 to 10 (inclusive)
    specimen_directory = f'specimen{i}'
    specimen_directory_path = os.path.join(base_path, specimen_directory)

    if not os.path.isdir(specimen_directory_path):
        continue  # Skip if the directory doesn't exist

    print(f"Processing specimen directory: {specimen_directory_path}")

    recordings_directory = 'Recordings'
    recordings_directory_path = os.path.join(specimen_directory_path, recordings_directory)

    if not os.path.isdir(recordings_directory_path):
        continue  # Skip if the 'Recordings' directory doesn't exist

    print(f"Processing recordings directory: {recordings_directory_path}")

    pointcloud_count = 0
    camparams_folders = []

    for root, dirs, files in os.walk(recordings_directory_path):
        for dir_name in dirs:
            subdirectory_path = os.path.join(root, dir_name)
            print(subdirectory_path)

            # Create 'pointcloud' directory inside subdirectory_path
            pointcloud_directory = os.path.join(subdirectory_path, 'pointcloud')
            os.makedirs(pointcloud_directory, exist_ok=True)

            folder_path = subdirectory_path
            i = 0
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and filename.endswith('.svo'):
                    if filename == 'Video_0.svo':
                        conf_path = r"E:\Ghazi\CamParams_0_31\SN10027879.conf"
                    elif filename == 'Video_1.svo':
                        conf_path = r"E:\Ghazi\CamParams_0_31\SN10028650.conf"
                    if os.path.isfile(conf_path):
                        process_svo_file(file_path, conf_path, i, pointcloud_directory)
                        i = i + 1

            if dir_name.startswith('CamParams'):
                camparams_folders.append(dir_name)
                pointcloud_count += 1

    # Check the number of 'CamParams' folders and their names
    if pointcloud_count == 2 and len(camparams_folders) == 2:
        camparams_a, camparams_b = camparams_folders
        if camparams_a != camparams_b:
            # Extract values of a and b from the folder names
            _, a, b = camparams_a.split('_')
            _, _, b = camparams_b.split('_')

            # Create the 'Calib_a' and 'Calib_b' folder paths
            calib_a_folder = os.path.join(recordings_directory_path, f"Calib_{a}")
            calib_b_folder = os.path.join(recordings_directory_path, f"Calib_{b}")

            # Assign the path to 'conf_path' based on the subdirectory number
            for root, dirs, files in os.walk(recordings_directory_path):
                for dir_name in dirs:
                    if dir_name.startswith(camparams_a):
                        conf_path = calib_a_folder
                    elif dir_name.startswith(camparams_b):
                        conf_path = calib_b_folder
                    else:
                        continue

                    # Perform additional actions with 'conf_path'
                    print(conf_path)

    else:
        print("Either one or both 'CamParams' folders are missing, or the naming convention is not followed.")
