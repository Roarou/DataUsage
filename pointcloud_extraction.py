import open3d as o3d
import numpy as np
import cv2
import pyzed.sl as sl
import os
import ogl_viewer.viewer as gl
from tqdm import tqdm
import sys

res = sl.Resolution()
res.width = 720
res.height = 404


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
            camera_model = zed.get_camera_information().camera_model
            point_cloud = sl.Mat()
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()
            nb_frames = zed.get_svo_number_of_frames()
            nb_frames = 5
            video_folder = f'Video_{i}'
            i = i + 1
            os.makedirs(video_folder, exist_ok=True)  # Create the video folder if it doesn't exist

            with tqdm(total=nb_frames, desc=f'Processing {filename}', unit='frame') as pbar:
                for j in range(nb_frames):
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                        output_path = os.path.join(video_folder, f'Pointcloud_{j}.pcd')
                        point_cloud_to_save = sl.Mat()
                        err = point_cloud.write(output_path)
                        if err == sl.ERROR_CODE.SUCCESS:
                            print(f'Point cloud saved: {output_path}')
                        else:
                            print(f'Failed to save point cloud: {output_path}')
                    pbar.update(1)


if __name__ == "__main__":
    folder_path = "E:/Ghazi/Recordings/Recording0"
    iter(folder_path)
