import pyrender
import trimesh
import os
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import pandas as pd
import shutil
import time


def initialize_zed_camera(video_dir):
    input_type = sl.InputType()
    input_type.set_from_svo_file(video_dir)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    return zed


def retrieve_data_from_zed(zed):
    image = sl.Mat()
    depth = sl.Mat()
    pcd_data = [sl.Mat() for _ in range(5)]

    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    for pcd in pcd_data:
        zed.retrieve_measure(pcd, sl.MEASURE.XYZRGBA)

    return image, depth, pcd_data


def read_and_transform_vertebrae(dir, specimen, df, cur_frame):
    vertebrae_files = [f"L{i}.stl" for i in range(1, 6)]
    vertebrae = [o3d.io.read_triangle_mesh(os.path.join(dir, specimen, "STL", f)) for f in vertebrae_files]
    for i, vertebra in enumerate(vertebrae):
        vertebra.compute_vertex_normals()
        pose_index = 5 * cur_frame + i
        vertebra_pose = np.array([
            [df.iloc[pose_index]["R00"], df.iloc[pose_index]["R01"], df.iloc[pose_index]["R02"],
             df.iloc[pose_index]["T0"]],
            [df.iloc[pose_index]["R10"], df.iloc[pose_index]["R11"], df.iloc[pose_index]["R12"],
             df.iloc[pose_index]["T1"]],
            [df.iloc[pose_index]["R20"], df.iloc[pose_index]["R21"], df.iloc[pose_index]["R22"],
             df.iloc[pose_index]["T2"]],
            [0, 0, 0, 1]
        ])
        vertebra.transform(vertebra_pose)
    return vertebrae


def process_recordings(dir, specimens, camera_nums):
    calib_dest_dir = "/usr/local/zed/settings/"

    for specimen in specimens:
        recordings = len(os.listdir(os.path.join(dir, specimen, "recordings")))

        for recording in range(17, recordings):
            for camera_num in camera_nums:
                print(f"processing specimen: {specimen} recording {recording} video: {camera_num}")

                video_dir = os.path.join(dir, specimen, f"recordings/recording{recording}/Video_{camera_num}.svo")
                calib_src_dir = os.path.join(dir, specimen, "Calib")
                shutil.copytree(calib_src_dir, calib_dest_dir, dirs_exist_ok=True)

                tracking_file = os.path.join(dir, specimen, f"recordings/recording{recording}/Poses_{camera_num}.txt")
                df = pd.read_csv(tracking_file, sep=',', header=None,
                                 names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2",
                                        "R30", "R31", "R33", "T3"])

                zed = initialize_zed_camera(video_dir)

                runtime_parameters = sl.RuntimeParameters()
                nb_frames = 1  # As per original code, consider only 1 frame
                cur_frame = 0

                while cur_frame < nb_frames:
                    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                        image, depth, pcd_data = retrieve_data_from_zed(zed)

                        save_data_dir = os.path.join("/media/aidana/US study/segmented_spinedepth", specimen,
                                                     f"recording{recording}",
                                                     f"pointcloud_cam_{camera_num}_recording_{cur_frame}")
                        print(f"saving files to: {save_data_dir}")
                        if not os.path.exists(save_data_dir):
                            os.makedirs(save_data_dir)

                        image.write(os.path.join(save_data_dir, 'image.png'))
                        depth.write(os.path.join(save_data_dir, 'depth.png'))

                        vertebrae = read_and_transform_vertebrae(dir, specimen, df, cur_frame)

                        # Continue with the rest of the operations like rendering, masking, point cloud processing, etc.
                        # This would include the rest of the steps you had in your original script for each recording and camera number.
                        # ...

                        cur_frame += 1


# Main execution
if __name__ == "__main__":
    start_time = time.time()
    dir = "/media/aidana/US study/SpineDepth"
    specimens = ["Specimen_3"]
    camera_nums = [0, 1]

    process_recordings(dir, specimens, camera_nums)

    print("time taken: ", time.time() - start_time)
