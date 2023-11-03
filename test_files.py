import os
import shutil
import time
from itertools import product
import pandas as pd
import pyrender
import trimesh
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d

# Constants
START_TIME = time.time()
SOURCE_DIR = "/media/aidana/US study/SpineDepth"
CALIB_DEST_DIR = "/usr/local/zed/settings/"


def create_scene_with_camera(camera, rotation_matrix):
    scene = pyrender.Scene()
    scene.add(camera, pose=rotation_matrix)
    return scene


def process_and_save_pointcloud(pcd, mask_image, save_data_dir):
    numpy_array = pcd.get_data()
    mask_all_zero = np.all(mask_image == 0, axis=2)
    indices = np.argwhere(mask_all_zero)

    for j, k in indices:
        rgba = numpy_array[j][k][:]
        if not np.isnan(rgba.all()):
            rgba[:] = 0
            numpy_array[j][k][:] = rgba
    return pcd
def create_rotation_matrix():
    rotate1 = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0),
        direction=[0, 1, 0],
        point=[0, 0, 0])
    rotate2 = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0),
        direction=[0, 0, 1],
        point=[0, 0, 0])
    return rotate1 @ rotate2

def render_scene_and_save_depth(renderer, scene, save_path, mask_name):
    _, depth = renderer.render(scene)
    cv2.imwrite(os.path.join(save_path, f"{mask_name}.png"), depth)
# Function to read tracking file into a DataFrame
def read_tracking_file(file_path):
    column_names = ["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1",
                    "R20", "R21", "R22", "T2", "R30", "R31", "R33", "T3"]
    return pd.read_csv(file_path, sep=',', header=None, names=column_names)

def create_and_save_directory(base_path, specimen, recording, camera_num, cur_frame):
    save_data_dir = os.path.join(base_path, specimen, f"recording{recording}", f"pointcloud_cam_{camera_num}_recording_{cur_frame}")
    print(f"Saving files to: {save_data_dir}")
    os.makedirs(save_data_dir, exist_ok=True)
    return save_data_dir

def read_and_transform_vertebra(base_path, specimen, vertebra_name, transformation):
    vertebra = o3d.io.read_triangle_mesh(os.path.join(base_path, specimen, "STL", f"{vertebra_name}.stl"))
    vertebra.compute_vertex_normals()
    vertebra.transform(transformation)
    return vertebra

def create_transformation_matrix(df_row):
    return np.array([
        [df_row["R00"], df_row["R01"], df_row["R02"], df_row["T0"]],
        [df_row["R10"], df_row["R11"], df_row["R12"], df_row["T1"]],
        [df_row["R20"], df_row["R21"], df_row["R22"], df_row["T2"]],
        [0, 0, 0, 1]
    ])

# Main code
specimens = ["Specimen_3", "Specimen_5", "Specimen_6", "Specimen_7", "Specimen_9", "Specimen_10"]
camera_nums = [0, 1]
runtime_parameters = sl.RuntimeParameters()
start_time = time.time()
for specimen, camera_num in product(specimens, camera_nums):
    print(f"Processing specimen: {specimen}, camera: {camera_num}")

    recordings = len(os.listdir(os.path.join(SOURCE_DIR, specimen, "recordings")))
    for recording in range(recordings):
        video_dir = os.path.join(SOURCE_DIR, specimen, f"recordings/recording{recording}/Video_{camera_num}.svo")
        calib_src_dir = os.path.join(SOURCE_DIR, specimen, "Calib")
        shutil.copytree(calib_src_dir, CALIB_DEST_DIR, dirs_exist_ok=True)

        tracking_file = os.path.join(SOURCE_DIR, specimen, f"recordings/recording{recording}/Poses_{camera_num}.txt")
        df = read_tracking_file(tracking_file)

        input_type = sl.InputType()
        input_type.set_from_svo_file(video_dir)

        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

        with sl.Camera() as zed:
            status = zed.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()

        zed_pose = sl.Pose()
        zed_sensors = sl.SensorsData()
        calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        nb_frames = zed.get_svo_number_of_frames()
        nb_frames = 100
        cur_frame = 0
        while cur_frame<nb_frames:

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                image, depth = sl.Mat(), sl.Mat()
                pc_mats = [sl.Mat() for _ in range(5)]

                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                pcd = sl.Mat()
                for pc_mat in pc_mats:
                    zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)

                save_data_dir = os.path.join("/media/aidana/US study/segmented_spinedepth", specimen,"recording{}".format(recording),
                                                 "pointcloud_cam_{}_recording_{}".format(camera_num, cur_frame))
                print("saving files to:{}".format(save_data_dir))
                if not os.path.exists(save_data_dir):
                    os.makedirs(save_data_dir)


                image.write(os.path.join(save_data_dir, 'image.png'))
                depth.write(os.path.join(save_data_dir, 'depth.png'))

                # Read and transform vertebrae
                vertebrae = []
                for i in range(5):
                    df_row = df.iloc[5 * cur_frame + i]
                    transformation_matrix = create_transformation_matrix(df_row)
                    vertebra = read_and_transform_vertebra(SOURCE_DIR, specimen, f"L{i + 1}", transformation_matrix)
                    vertebrae.append(vertebra)
                # Combine all vertebrae
                spine = sum(vertebrae)

                R = zed_pose.get_rotation_matrix(sl.Rotation()).r.T
                t = zed_pose.get_translation(sl.Translation()).get()

                # Create the 4x4 extrinsics transformation matrix
                extrinsics_matrix = np.identity(4)
                extrinsics_matrix[:3, :3] = R
                extrinsics_matrix[:3, 3] = t

                camera_pose = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ])

                left_camera_intrinsic = calibration_params.left_cam
                camera = pyrender.camera.IntrinsicsCamera(left_camera_intrinsic.fx, left_camera_intrinsic.fy,
                                                          left_camera_intrinsic.cx, left_camera_intrinsic.cy, znear=250,
                                                          zfar=2000)
                mesh = []
                for i, vert in enumerate(vertebrae):
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, f"transformed_vertebra{i}.stl"), vert)
                    mask_stl = trimesh.load(os.path.join(save_data_dir, f"transformed_vertebra{i}.stl"))
                    mesh.append(pyrender.Mesh.from_trimesh(mask_stl))

                # Create rotation matrix
                rotation_matrix = create_rotation_matrix()

                # Create renderers and scenes
                renderers = [pyrender.OffscreenRenderer(1920, 1080) for _ in range(5)]
                scenes = [create_scene_with_camera(camera, rotation_matrix) for _ in range(5)]

                # Render scenes and save depth images
                for i, (renderer, scene) in enumerate(zip(renderers, scenes)):
                    render_scene_and_save_depth(renderer, scene, save_data_dir, f"mask{i + 1}")

                mask_image1 = cv2.imread(os.path.join(save_data_dir, "mask1.png"))
                mask_image2 = cv2.imread(os.path.join(save_data_dir, "mask2.png"))
                mask_image3 = cv2.imread(os.path.join(save_data_dir, "mask3.png"))
                mask_image4 = cv2.imread(os.path.join(save_data_dir, "mask4.png"))
                mask_image5 = cv2.imread(os.path.join(save_data_dir, "mask5.png"))

                mask_images = [mask_image1, mask_image2, mask_image3, mask_image4, mask_image5]

                for i, pcd in enumerate(pc_mats):
                    pcd += process_and_save_pointcloud(pcd, mask_images[i], save_data_dir)

                pcd_verts = [o3d.io.read_point_cloud(os.path.join(save_data_dir, f'pointcloud_vert{i + 1}.ply')) for i
                             in range(5)]

                # Coloring point clouds
                colors = [[1, 0.706, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
                for i in range(5):
                    pcd_verts[i].paint_uniform_color(colors[i])