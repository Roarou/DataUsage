import open3d as o3d
import numpy as np
import cv2
import pyzed.sl as sl
import os
import ogl_viewer.viewer as gl
from tqdm import tqdm
import sys
import json
import shutil

res = sl.Resolution()
res.width = 1920
res.height = 1080

def get_pose(zed, zed_pose, zed_sensors):
    # Get the pose of the left eye of the camera with reference to the world frame
    zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    # zed_imu = zed_sensors.get_imu_data()

    # Display the translation and timestamp
    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz,
                                                                           zed_pose.timestamp.get_milliseconds()))

    # Display the orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

    pose_dict = {'Translation': {'Tx': tx, 'Ty': ty, 'Tz': tz},
                 'Orientation': {'Ox': ox, 'Oy': oy, 'Oz': oz, 'Ow': ow}}

    return pose_dict

def iter(folder_path: str):
    i = 0
    for filename in os.listdir(folder_path):
        input_type = sl.InputType()
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.svo'):
            input_type.set_from_svo_file(file_path)
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
            zed = sl.Camera()
            status = zed.open(init)
            zed_pose = sl.Pose()
            zed_sensors = sl.SensorsData()
            camera_model = zed.get_camera_information().camera_model
            point_cloud = sl.Mat()
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
            nb_frames = zed.get_svo_number_of_frames()
            nb_frames = 5
            # set up output directory and delete old output
            print("clear old output")
            dir_path = "output"
            try:
                shutil.rmtree(dir_path)
            except OSError as e:
                print("Error: %s : %s" % (dir_path, e.strerror))
            video_folder = f'Video_{i}'

            i = i + 1
            os.makedirs(video_folder, exist_ok=True)  # Create the video folder if it doesn't exist

            with tqdm(total=nb_frames, desc=f'Processing {filename}', unit='frame') as pbar:
                for j in range(nb_frames):
                    pose_dir = os.path.join(dir_path, "frame_{}/pose".format(j))
                    os.makedirs(pose_dir, exist_ok=True)
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                        output_path = os.path.join(video_folder, f'Pointcloud_{j}.pcd')
                        err = point_cloud.write(output_path)
                        if err == sl.ERROR_CODE.SUCCESS:
                            print(f'Point cloud saved: {output_path}')
                            pose_dict = get_pose(zed, zed_pose, zed_sensors)
                            pose_filepath = os.path.join(pose_dir, 'pose.json')
                            with open(pose_filepath, 'w') as outfile:
                                json.dump(pose_dict, outfile)
                        else:
                            print(f'Failed to save point cloud: {output_path}')
                    pbar.update(1)


if __name__ == "__main__":
    folder_path = "E:/Ghazi/Recordings/Recording0"
    iter(folder_path)
