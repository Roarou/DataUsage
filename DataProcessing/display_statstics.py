from DataProcessing.Extraction.extract_tf_matrix_groundtruth import extract_transformation_matrices
import numpy as np
import matplotlib.pyplot as plt
import os

file_paths = []  # List of file paths
plt.close('all')

for j in range(2, 6):
    # Define the specimen name and base directory for recordings
    spec = f'Specimen_{j}'
    base_directory = f'G:\SpineDepth\{spec}\Recordings'
    for i in range(40):
        # Generate the recording folder name
        recording_folder_name = f"Recording{i}"

        # Construct paths for recording and point cloud folders
        recording_folder_path = os.path.join(base_directory, recording_folder_name)

        # Check if the point cloud folder exists
        if os.path.exists(recording_folder_path):
            # List of video folders within the point cloud folder
            pose_file = 'Poses_0.txt'
            pose_path = os.path.join(recording_folder_path, pose_file)
            print(pose_path)
            file_paths.append(pose_path)

files_data = []
vertebra_stats = []

# Step 1: Load Transformation Data
for file_path in file_paths:
    transformation_matrices = extract_transformation_matrices(file_path)
    file_data = []
    for tf_matrix in transformation_matrices:
        rotation_values = tf_matrix[:3, :3].flatten()
        translation_values = tf_matrix[:3, 3]
        transformation_values = np.concatenate((rotation_values, translation_values))
        file_data.append(transformation_values)
    print(f'Processed {file_path}')
    files_data.append(file_data)

# Step 2: Calculate Statistics
for file_data in files_data:
    file_stats = [{'rotation_mean': 0, 'rotation_std': 0, 'translation_mean': 0, 'translation_std': 0} for _ in range(5)]

    for frame_index in range(0, len(file_data), 5):
        frame = file_data[frame_index:frame_index + 5]

        for vertebra_index, transformation in enumerate(frame):
            rotation_values = transformation[:9]
            translation_values = transformation[9:]
            translation_norm = np.linalg.norm(translation_values)
            file_stats[vertebra_index]['translation_mean'] += translation_norm
            file_stats[vertebra_index]['rotation_mean'] += np.mean(rotation_values)
            file_stats[vertebra_index]['rotation_std'] += np.std(rotation_values)
            file_stats[vertebra_index]['translation_std'] += np.std(translation_values)

    total_frames_in_file = len(file_data) // 5
    for vertebra_stat in file_stats:
        vertebra_stat['rotation_mean'] /= total_frames_in_file
        vertebra_stat['rotation_std'] /= total_frames_in_file
        vertebra_stat['translation_mean'] /= total_frames_in_file
        vertebra_stat['translation_std'] /= total_frames_in_file

    vertebra_stats.append(file_stats)

# Step 2.5: Group the statistics by specimen
specimen_stats = {}
for j, file_stats in enumerate(vertebra_stats):
    spec_num = (j // 40) + 2  # Assuming 40 recordings per specimen and specimens numbered from 2 to 5
    if spec_num not in specimen_stats:
        specimen_stats[spec_num] = [file_stat.copy() for file_stat in file_stats]
    else:
        for vertebra_stat, accumulated_stat in zip(file_stats, specimen_stats[spec_num]):
            accumulated_stat['rotation_mean'] += vertebra_stat['rotation_mean']
            accumulated_stat['translation_mean'] += vertebra_stat['translation_mean']

# Divide by the total number of files for each specimen to get the mean
total_files_per_specimen = 40  # Assuming 40 recordings per specimen
for spec_stats in specimen_stats.values():
    for vertebra_stat in spec_stats:
        vertebra_stat['rotation_mean'] /= total_files_per_specimen
        vertebra_stat['translation_mean'] /= total_files_per_specimen
# Step 3: Plot the Statistics (Box Plots and Scatter Plots)
def plot_statistics(stats, title, y_label, stat_type):
    plt.figure()  # Create a new figure
    # Iterate through each vertebra and plot its statistics
    for vertebra_index in range(5):
        vertebra_stats = [file_stats[vertebra_index][stat_type] for file_stats in stats]
        plt.boxplot(vertebra_stats, positions=[vertebra_index], widths=0.6)  # Plotting at specific positions

    plt.title(title)  # Main title for the entire figure
    plt.ylabel(y_label)
    plt.xlabel('Vertebra Index')
    plt.xticks(range(5), [f'Vertebra {i}' for i in range(5)])  # Set x-tick labels to vertebra names
    plt.show()

    for vertebra_index in range(5):
        vertebra_stats_scatter = [file_stats[vertebra_index][stat_type] for file_stats in stats]
        # Scatter Plot for individual vertebra
        plt.scatter(range(len(vertebra_stats_scatter)), vertebra_stats_scatter)
        plt.title(f'{title} for Vertebra {vertebra_index} (Scatter Plot)')
        plt.xlabel('File Index')
        plt.ylabel(y_label)
        plt.show()

rotation_means = [[stats['rotation_mean'] for stats in file_stats] for file_stats in vertebra_stats]
translation_means = [[stats['translation_mean'] for stats in file_stats] for file_stats in vertebra_stats]

plot_statistics(vertebra_stats, 'Mean Rotation by Vertebra', 'Rotation Value', 'rotation_mean')
plot_statistics(vertebra_stats, 'Mean Translation by Vertebra', 'Translation Value', 'translation_mean')
# Step 4: Plot the 'Number of specimen' Box Plot for each Specimen on the same figure
def plot_specimen_statistics(stats, title, y_label, stat_type_mean, stat_type_std):
    specimens = sorted(list(stats.keys()))
    data_means = []
    data_stds = []
    for specimen in specimens:
        specimen_data_mean = [vertebra_stats[stat_type_mean] for vertebra_stats in stats[specimen]]
        specimen_data_std = [vertebra_stats[stat_type_std] for vertebra_stats in stats[specimen]]
        data_means.append(specimen_data_mean) # Taking means across all vertebrae
        data_stds.append(specimen_data_std) # Taking standard deviations across all vertebrae

    plt.boxplot(data_means) # Box plot for the means
    for i, (mean, std) in enumerate(zip(data_means, data_stds)):
        plt.errorbar(i + 1, np.mean(mean), yerr=np.mean(std), fmt='o', color='red') # Plot mean of standard deviations with error bars

    plt.title(f'{title} across Specimens')
    plt.ylabel(y_label)
    plt.xlabel('Specimen Number')
    plt.xticks(range(1, len(specimens) + 1), [f'Specimen {spec}' for spec in specimens])
    plt.show()

plot_specimen_statistics(specimen_stats, 'Mean Rotation', 'Rotation Value', 'rotation_mean', 'rotation_std')
plot_specimen_statistics(specimen_stats, 'Mean Translation', 'Translation Value', 'translation_mean', 'translation_std')
