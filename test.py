# from DataProcessing.Pointcloud.merge_point_cloud import process_point_cloud
# video_0 =r'G:\SpineDepth\Specimen_6\Recordings\Recording5\pointcloud\Video_0\Pointcloud_pred.pcd'
# video_1 = r'G:\SpineDepth\Specimen_6\Recordings\Recording5\pointcloud\Video_1\Pointcloud_pred.pcd'
# process_point_cloud([video_0, video_1, 6, 120, r'G:\SpineDepth\Specimen_6\Recordings\Recording5'])
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_file(file):
    base_path = r'L:\groundtruth'  # Replace with your actual path
    corrupted_path = []

    file_path = os.path.join(base_path, file)
    input_pcd = o3d.io.read_point_cloud(file_path, remove_nan_points=True, remove_infinite_points=True)
    points = np.asarray(input_pcd.points)

    if np.any(np.isnan(points)):
        raise ValueError(f"Input data contains NaN values. Reading: {file}")
        corrupted_path.append(file)

    centroid = np.mean(points, axis=0)
    if np.any(np.isnan(centroid)):
        corrupted_path.append(file)

    return corrupted_path


if __name__ == '__main__':
    base_path = r'L:\groundtruth'
    file_list = os.listdir(base_path)

    num_cpus = cpu_count()

    # Utilize multiprocessing with Pool
    with Pool(processes=num_cpus) as pool:
        # wrap the pool.map with tqdm to show the progress bar
        results = list(tqdm(pool.imap(process_file, file_list), total=len(file_list), desc='Processing files'))

    # Flatten the list of corrupted_path and remove duplicates
    corrupted_path = list(set([item for sublist in results for item in sublist]))

    print("Corrupted Files:", corrupted_path)
    # Writing the corrupted files to a text file
    with open('corrupted_files_gt.txt', 'w') as f:
        for file_name in corrupted_path:
            f.write(f"{file_name}\n")

    print("Corrupted files have been written to 'corrupted_files.txt'")
