import os
import json
import shutil
from tqdm import tqdm
import pyzed.sl as sl
import open3d as o3d
import numpy as np

def get_pose(zed, zed_pose, zed_sensors):
    """
    Get the pose of the left eye of the camera with reference to the world frame.
    """
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)

    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)

    print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz,
                                                                           zed_pose.timestamp.get_milliseconds()))

    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)

    print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

    pose_dict = {'Translation': {'Tx': tx, 'Ty': ty, 'Tz': tz},
                 'Orientation': {'Ox': ox, 'Oy': oy, 'Oz': oz, 'Ow': ow}}

    return pose_dict


def process_config_files(config_file_path):
    path_b = r'C:\\ProgramData\\Stereolabs\\settings'  # Replace with your path

    # Check if the config file exists
    if not os.path.isfile(config_file_path):
        print("Config file doesn't exist.")
        exit(1)

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
    shutil.copy2(config_file_path, path_b)
    print(f'Copied {config_file_path} ')


def process_frame(zed, frame_index, video_folder, dir_path):
    """
    Process a single frame: grab point cloud, save it, and save the pose information.
    """
    # pose_dir = os.path.join(dir_path, "frame_{}/pose".format(frame_index))
    # os.makedirs(pose_dir, exist_ok=True)

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        # Convert to Open3D format
        points = np.asarray(point_cloud.get_data())
        # Mask to remove 'nan' values
        mask = ~np.isnan(points).any(axis=2)
        filtered_points = points[mask]

        xyz = filtered_points[:, :3].astype(np.float32)  # Convert to float32
        # Decode the fourth channel to retrieve colors
        rgba = filtered_points[:, 3]
        rgba_uint8 = np.zeros((rgba.shape[0], 4), dtype=np.uint8)
        for i in range(4):
            rgba_uint8[:, i] = np.right_shift(
                np.bitwise_and(np.array(rgba, dtype=np.uint32), np.uint32(255 << (i * 8))), i * 8)

        rgb = rgba_uint8[:, :3] / 255  # normalize RGB values

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        downsampled_pcd = pcd.uniform_down_sample(2)

        output_path = os.path.join(video_folder, f'Pointcloud_{frame_index}.pcd')

        # Save the downsampled point cloud
        err = o3d.io.write_point_cloud(output_path, downsampled_pcd)

        if err:
            print(f'Point cloud saved: {output_path}')
            """
            zed_pose = sl.Pose()
            zed_sensors = sl.SensorsData()
            pose_dict = get_pose(zed, zed_pose, zed_sensors)
            pose_filepath = os.path.join(pose_dir, 'pose.json')

            with open(pose_filepath, 'w') as outfile:
                json.dump(pose_dict, outfile)
                """
        else:
            print(f'Failed to save point cloud: {output_path}')


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

    print("Clearing old output")
    dir_path = pointcloud_directory
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

    video_folder = f'Video_{iteration}'
    video_folder = os.path.join(pointcloud_directory, video_folder)
    os.makedirs(video_folder, exist_ok=True)

    with tqdm(total=nb_frames, desc=f'Processing {file_path}', unit='frame') as pbar:
        for frame_index in range(nb_frames):
            process_frame(zed, frame_index, video_folder, dir_path)
            pbar.update(1)


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
