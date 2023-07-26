import os
import shutil
import pyzed.sl as sl
import open3d as o3d
import numpy as np
from multiprocessing import Pool, cpu_count, TimeoutError, Manager


def process_config_files(config_file_path):
    path_b = r'C:\\ProgramData\\Stereolabs\\settings'  # Replace with your path

    # Check if the config file exists
    if not os.path.isfile(config_file_path):
        print("Config file doesn't exist.")
        return False

    # Delete all files in path_b
    for filename in os.listdir(path_b):
        file_path = os.path.join(path_b, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f'Deleted: {file_path}.')
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # Copy the config file to path_b
    try:
        shutil.copy2(config_file_path, path_b)
        print(f'Copied {config_file_path} ')
        return True
    except PermissionError:
        print(f'PermissionError: Could not copy {config_file_path} due to insufficient permissions.')
        return False


def process_frame(zed, frame_index, video_folder, dir_path):
    """
    Process a single frame: grab point cloud, save it, and save the pose information.
    """

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        points = np.asarray(point_cloud.get_data())
        mask = ~np.isnan(points).any(axis=2)
        filtered_points = points[mask]
        xyz = filtered_points[:, :3].astype(np.float32)

        rgb = np.frombuffer(np.float32(filtered_points[:, 3]).tobytes(), np.uint8).reshape(-1, 4)[:, :3] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        downsampled_pcd = pcd

        output_path = os.path.join(video_folder, f'Pointcloud_{frame_index}.pcd')

        success = o3d.io.write_point_cloud(output_path, downsampled_pcd)

        if success:
            print(f'Point cloud saved: {output_path}')
        else:
            print(f'Failed to save point cloud: {output_path}')

        return success


def process_frames(queue, file_path, frame_indices, video_folder, dir_path):
    """
    Process multiple frames from the same SVO file: open the camera, and process each frame.
    """
    input_type = sl.InputType()
    input_type.set_from_svo_file(file_path)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    zed = sl.Camera()
    status = zed.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(repr(status))

    for frame_index in frame_indices:
        success = process_frame(zed, frame_index, video_folder, dir_path)
        queue.put(success)


def process_pool(args, conf_path):

    if not process_config_files(config_file_path=conf_path):
        print('Failed to process config files.')
        return False

    with Manager() as manager:
        queue = manager.Queue()  # Queue to store results
        pool = Pool()
        results = []
        for arg in args:
            # Add the queue as the first argument to process_frames
            results.append(pool.apply_async(process_frames, (queue, *arg)))

        for result in results:
            try:
                result.get(timeout=600)  # Set timeout to 10 minutes
            except TimeoutError:
                print("Timeout reached. Terminating the pool.")
                pool.terminate()
                pool.join()
                return False
        pool.close()
        pool.join()

        # Here you could handle the results stored in the queue.
        # For example, you could check if all tasks were successful:
        while not queue.empty():
            success = queue.get()
            if not success:
                print("At least one task failed.")
                return False  # Or handle the failure in some other way

        return True

def process_svo_file(file_path, conf_path, iteration, pointcloud_directory):
    """
    Process an SVO file: open it, iterate through frames, and process each frame.
    """
    input_type = sl.InputType()
    input_type.set_from_svo_file(file_path)
    process_config_files(config_file_path=conf_path)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    zed = sl.Camera()
    status = zed.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(repr(status))

    nb_frames = zed.get_svo_number_of_frames()
    dir_path = pointcloud_directory
    if not iteration:
        print("Clearing old output")

        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s : %s" % (dir_path, e.strerror))

    video_folder = f'Video_{iteration}'
    video_folder = os.path.join(pointcloud_directory, video_folder)
    os.makedirs(video_folder, exist_ok=True)

    # Create chunks of frames for each process
    num_processes = cpu_count()

    frame_chunks = np.array_split(range(nb_frames), num_processes)

    args = [(file_path, frame_chunk, video_folder, dir_path) for frame_chunk in frame_chunks]

    while not process_pool(args, conf_path):
        print("Restarting the pool.")

if __name__ == "__main__":
    folder_path = "E:/Ghazi/Recordings/Recording0"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.svo'):
            if filename == 'Video_0.svo':
                conf_path = "E:/Ghazi/CamParams_0_31/SN10027879.conf"
            elif filename == 'Video_1.svo':
                conf_path = "E:/Ghazi/CamParams_0_31/SN10028650.conf"
            if os.path.isfile(conf_path):
                process_svo_file(file_path, conf_path, i)
                i = i + 1
