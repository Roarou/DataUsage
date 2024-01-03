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

start_time = time.time()
# Define the source directory containing the depth data of spine scans.
SOURCE_DIR = r"G:\SpineDepth"

# Define the destination directory for calibration settings required by the ZED camera SDK.
CALIB_DEST_DIR = r'C:\\ProgramData\\Stereolabs\\settings'

# specimens = os.listdir(dir)
specimens = ["Specimen_3"]
camera_nums = [0, 1]
for specimen in specimens:
    recordings = len(os.listdir(os.path.join(SOURCE_DIR, specimen, "Recordings")))
    for recording in range(recordings):
        for camera_num in camera_nums:
            print("processing specimen: {} recording {} video: {}".format(specimen, recording, camera_num))
            video_dir = os.path.join(SOURCE_DIR, specimen, "Recordings/Recording{}/Video_{}.svo".format(recording, camera_num))
            calib_src_dir = os.path.join(SOURCE_DIR, specimen, "Calib")
            shutil.copytree(calib_src_dir, CALIB_DEST_DIR, dirs_exist_ok=True)
            tracking_file = os.path.join(SOURCE_DIR, specimen,
                                         "Recordings/Recording{}/Poses_{}.txt".format(recording, camera_num))
            df = pd.read_csv(tracking_file, sep=',', header=None,
                             names=["R00", "R01", "R02", "T0", "R10", "R11", "R12", "T1", "R20", "R21", "R22", "T2",
                                    "R30",
                                    "R31", "R33", "T3"])

            input_type = sl.InputType()
            input_type.set_from_svo_file(video_dir)
            init_params = sl.InitParameters()
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

            zed = sl.Camera()
            status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()

            runtime_parameters = sl.RuntimeParameters()

            zed_pose = sl.Pose()
            zed_sensors = sl.SensorsData()
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            nb_frames = zed.get_svo_number_of_frames()
            nb_frames = 1
            cur_frame = 0
            while cur_frame < nb_frames:

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    image = sl.Mat()
                    depth = sl.Mat()
                    pcd1 = sl.Mat()
                    pcd2 = sl.Mat()
                    pcd3 = sl.Mat()
                    pcd4 = sl.Mat()
                    pcd5 = sl.Mat()

                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    zed.retrieve_measure(pcd1, sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(pcd2, sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(pcd3, sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(pcd4, sl.MEASURE.XYZRGBA)
                    zed.retrieve_measure(pcd5, sl.MEASURE.XYZRGBA)

                    save_data_dir = os.path.join("L:")
                    print("saving files to:{}".format(save_data_dir))
                    if not os.path.exists(save_data_dir):
                        os.makedirs(save_data_dir)

                    image.write(os.path.join(save_data_dir, 'image.png'))
                    depth.write(os.path.join(save_data_dir, 'depth.png'))

                    # reading corresponding stl files
                    vertebra1 = o3d.io.read_triangle_mesh(os.path.join(SOURCE_DIR, specimen, "STL/L1.stl"))
                    vertebra2 = o3d.io.read_triangle_mesh(os.path.join(SOURCE_DIR, specimen, "STL/L2.stl"))
                    vertebra3 = o3d.io.read_triangle_mesh(os.path.join(SOURCE_DIR, specimen, "STL/L3.stl"))
                    vertebra4 = o3d.io.read_triangle_mesh(os.path.join(SOURCE_DIR, specimen, "STL/L4.stl"))
                    vertebra5 = o3d.io.read_triangle_mesh(os.path.join(SOURCE_DIR, specimen, "STL/L5.stl"))

                    vertebra1 = vertebra1.compute_vertex_normals()
                    vertebra2 = vertebra2.compute_vertex_normals()
                    vertebra3 = vertebra3.compute_vertex_normals()
                    vertebra4 = vertebra4.compute_vertex_normals()
                    vertebra5 = vertebra5.compute_vertex_normals()
                    i = 5 * cur_frame

                    vertebra1_pose = np.array(
                        [[df.iloc[i]["R00"], df.iloc[i]["R01"], df.iloc[i]["R02"], df.iloc[i]["T0"]],
                         [df.iloc[i]["R10"], df.iloc[i]["R11"], df.iloc[i]["R12"], df.iloc[i]["T1"]],
                         [df.iloc[i]["R20"], df.iloc[i]["R21"], df.iloc[i]["R22"], df.iloc[i]["T2"]],
                         [0, 0, 0, 1]])

                    vertebra2_pose = np.array(
                        [[df.iloc[i + 1]["R00"], df.iloc[i + 1]["R01"], df.iloc[i + 1]["R02"], df.iloc[i + 1]["T0"]],
                         [df.iloc[i + 1]["R10"], df.iloc[i + 1]["R11"], df.iloc[i + 1]["R12"], df.iloc[i + 1]["T1"]],
                         [df.iloc[i + 1]["R20"], df.iloc[i + 1]["R21"], df.iloc[i + 1]["R22"], df.iloc[i + 1]["T2"]],
                         [0, 0, 0, 1]])

                    vertebra3_pose = np.array(
                        [[df.iloc[i + 2]["R00"], df.iloc[i + 2]["R01"], df.iloc[i + 2]["R02"], df.iloc[i + 2]["T0"]],
                         [df.iloc[i + 2]["R10"], df.iloc[i + 2]["R11"], df.iloc[i + 2]["R12"], df.iloc[i + 2]["T1"]],
                         [df.iloc[i + 2]["R20"], df.iloc[i + 2]["R21"], df.iloc[i + 2]["R22"], df.iloc[i + 2]["T2"]],
                         [0, 0, 0, 1]])

                    vertebra4_pose = np.array(
                        [[df.iloc[i + 3]["R00"], df.iloc[i + 3]["R01"], df.iloc[i + 3]["R02"], df.iloc[i + 3]["T0"]],
                         [df.iloc[i + 3]["R10"], df.iloc[i + 3]["R11"], df.iloc[i + 3]["R12"], df.iloc[i + 3]["T1"]],
                         [df.iloc[i + 3]["R20"], df.iloc[i + 3]["R21"], df.iloc[i + 3]["R22"], df.iloc[i + 3]["T2"]],
                         [0, 0, 0, 1]])

                    vertebra5_pose = np.array(
                        [[df.iloc[i + 4]["R00"], df.iloc[i + 4]["R01"], df.iloc[i + 4]["R02"], df.iloc[i + 4]["T0"]],
                         [df.iloc[i + 4]["R10"], df.iloc[i + 4]["R11"], df.iloc[i + 4]["R12"], df.iloc[i + 4]["T1"]],
                         [df.iloc[i + 4]["R20"], df.iloc[i + 4]["R21"], df.iloc[i + 4]["R22"], df.iloc[i + 4]["T2"]],
                         [0, 0, 0, 1]])

                    vertebra1.transform(vertebra1_pose)
                    vertebra2.transform(vertebra2_pose)
                    vertebra3.transform(vertebra3_pose)
                    vertebra4.transform(vertebra4_pose)
                    vertebra5.transform(vertebra5_pose)

                    spine = vertebra1 + vertebra2 + vertebra3 + vertebra4 + vertebra5

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
                                                              left_camera_intrinsic.cx, left_camera_intrinsic.cy,
                                                              znear=250, zfar=2000)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, "transformed_vertebra1.stl"), vertebra1)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, "transformed_vertebra2.stl"), vertebra2)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, "transformed_vertebra3.stl"), vertebra3)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, "transformed_vertebra4.stl"), vertebra4)
                    o3d.io.write_triangle_mesh(os.path.join(save_data_dir, "transformed_vertebra5.stl"), vertebra5)

                    mask1_stl = trimesh.load(os.path.join(save_data_dir, "transformed_vertebra1.stl"))
                    mask2_stl = trimesh.load(os.path.join(save_data_dir, "transformed_vertebra2.stl"))
                    mask3_stl = trimesh.load(os.path.join(save_data_dir, "transformed_vertebra3.stl"))
                    mask4_stl = trimesh.load(os.path.join(save_data_dir, "transformed_vertebra4.stl"))
                    mask5_stl = trimesh.load(os.path.join(save_data_dir, "transformed_vertebra5.stl"))

                    mesh1 = pyrender.Mesh.from_trimesh(mask1_stl)
                    mesh2 = pyrender.Mesh.from_trimesh(mask2_stl)
                    mesh3 = pyrender.Mesh.from_trimesh(mask3_stl)
                    mesh4 = pyrender.Mesh.from_trimesh(mask4_stl)
                    mesh5 = pyrender.Mesh.from_trimesh(mask5_stl)

                    scene1 = pyrender.Scene()
                    scene2 = pyrender.Scene()
                    scene3 = pyrender.Scene()
                    scene4 = pyrender.Scene()
                    scene5 = pyrender.Scene()
                    print(extrinsics_matrix)
                    scene1.add(mesh1, pose=extrinsics_matrix)
                    scene2.add(mesh2, pose=extrinsics_matrix)
                    scene3.add(mesh3, pose=extrinsics_matrix)
                    scene4.add(mesh4, pose=extrinsics_matrix)
                    scene5.add(mesh5, pose=extrinsics_matrix)

                    rotate = trimesh.transformations.rotation_matrix(
                        angle=np.radians(180.0),
                        direction=[0, 1, 0],
                        point=[0, 0, 0])
                    rotate2 = trimesh.transformations.rotation_matrix(
                        angle=np.radians(180.0),
                        direction=[0, 0, 1],
                        point=[0, 0, 0])
                    scene1.add(camera, pose=rotate * rotate2)
                    scene2.add(camera, pose=rotate * rotate2)
                    scene3.add(camera, pose=rotate * rotate2)
                    scene4.add(camera, pose=rotate * rotate2)
                    scene5.add(camera, pose=rotate * rotate2)
                    r1 = pyrender.OffscreenRenderer(1920, 1080)
                    r2 = pyrender.OffscreenRenderer(1920, 1080)
                    r3 = pyrender.OffscreenRenderer(1920, 1080)
                    r4 = pyrender.OffscreenRenderer(1920, 1080)
                    r5 = pyrender.OffscreenRenderer(1920, 1080)

                    _, depth1 = r1.render(scene1)
                    _, depth2 = r1.render(scene2)
                    _, depth3 = r1.render(scene3)
                    _, depth4 = r1.render(scene4)
                    _, depth5 = r1.render(scene5)

                    cv2.imwrite(os.path.join(save_data_dir, "mask1.png"), depth1)
                    cv2.imwrite(os.path.join(save_data_dir, "mask2.png"), depth2)
                    cv2.imwrite(os.path.join(save_data_dir, "mask3.png"), depth3)
                    cv2.imwrite(os.path.join(save_data_dir, "mask4.png"), depth4)
                    cv2.imwrite(os.path.join(save_data_dir, "mask5.png"), depth5)

                    mask_image1 = cv2.imread(os.path.join(save_data_dir, "mask1.png"))
                    mask_image2 = cv2.imread(os.path.join(save_data_dir, "mask2.png"))
                    mask_image3 = cv2.imread(os.path.join(save_data_dir, "mask3.png"))
                    mask_image4 = cv2.imread(os.path.join(save_data_dir, "mask4.png"))
                    mask_image5 = cv2.imread(os.path.join(save_data_dir, "mask5.png"))

                    numpy_array1 = pcd1.get_data()
                    numpy_array2 = pcd2.get_data()
                    numpy_array3 = pcd3.get_data()
                    numpy_array4 = pcd4.get_data()
                    numpy_array5 = pcd5.get_data()
                    print(numpy_array2)
                    mask_all_zero_1 = np.all(mask_image1 == 0, axis=2)
                    mask_all_zero_2 = np.all(mask_image2 == 0, axis=2)
                    mask_all_zero_3 = np.all(mask_image3 == 0, axis=2)
                    mask_all_zero_4 = np.all(mask_image4 == 0, axis=2)
                    mask_all_zero_5 = np.all(mask_image5 == 0, axis=2)

                    inverted_mask_all_zero_1 = ~mask_all_zero_1
                    inverted_mask_all_zero_2 = ~mask_all_zero_2
                    inverted_mask_all_zero_3 = ~mask_all_zero_3
                    inverted_mask_all_zero_4 = ~mask_all_zero_4
                    inverted_mask_all_zero_5 = ~mask_all_zero_5

                    # Get indices where mask is True
                    indices1 = np.argwhere(mask_all_zero_1)
                    indices2 = np.argwhere(mask_all_zero_2)
                    indices3 = np.argwhere(mask_all_zero_3)
                    indices4 = np.argwhere(mask_all_zero_4)
                    indices5 = np.argwhere(mask_all_zero_5)
                    for j, k in indices1:
                        rgba = numpy_array1[j][k][:]

                        rgba[0] = 1
                        rgba[1] = 0
                        rgba[2] = 0
                        numpy_array1[j][k][:] = rgba

                    # Loop over the indices
                    for j, k in indices1:
                        rgba = numpy_array1[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = np.nan
                            rgba[1] = np.nan
                            rgba[2] = np.nan
                            rgba[3] = np.nan
                            numpy_array1[j][k][:] = rgba

                    for j, k in indices2:
                        rgba = numpy_array2[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = np.nan
                            rgba[1] = np.nan
                            rgba[2] = np.nan
                            rgba[3] = np.nan
                            numpy_array2[j][k][:] = rgba

                    for j, k in indices3:
                        rgba = numpy_array3[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = np.nan
                            rgba[1] = np.nan
                            rgba[2] = np.nan
                            rgba[3] = np.nan
                            numpy_array3[j][k][:] = rgba

                    for j, k in indices4:
                        rgba = numpy_array4[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = np.nan
                            rgba[1] = np.nan
                            rgba[2] = np.nan
                            rgba[3] = np.nan
                            numpy_array4[j][k][:] = rgba

                    for j, k in indices5:
                        rgba = numpy_array5[j][k][:]
                        if np.isnan(rgba.all()) == False:
                            rgba[0] = np.nan
                            rgba[1] = np.nan
                            rgba[2] = np.nan
                            rgba[3] = np.nan
                            numpy_array5[j][k][:] = rgba

                    pcd1.write(os.path.join(save_data_dir, 'pointcloud_vert1.ply'))
                    pcd2.write(os.path.join(save_data_dir, 'pointcloud_vert2.ply'))
                    pcd3.write(os.path.join(save_data_dir, 'pointcloud_vert3.ply'))
                    pcd4.write(os.path.join(save_data_dir, 'pointcloud_vert4.ply'))
                    pcd5.write(os.path.join(save_data_dir, 'pointcloud_vert5.ply'))

                    pcd_vert1 = o3d.io.read_point_cloud(os.path.join(save_data_dir, 'pointcloud_vert1.ply'))
                    pcd_vert2 = o3d.io.read_point_cloud(os.path.join(save_data_dir, 'pointcloud_vert2.ply'))
                    pcd_vert3 = o3d.io.read_point_cloud(os.path.join(save_data_dir, 'pointcloud_vert3.ply'))
                    pcd_vert4 = o3d.io.read_point_cloud(os.path.join(save_data_dir, 'pointcloud_vert4.ply'))
                    pcd_vert5 = o3d.io.read_point_cloud(os.path.join(save_data_dir, 'pointcloud_vert5.ply'))

                    pcd_vert1.paint_uniform_color([1, 0.706, 0])
                    pcd_vert2.paint_uniform_color([1, 0, 0])
                    pcd_vert3.paint_uniform_color([0, 1, 0])
                    pcd_vert4.paint_uniform_color([0, 0, 1])
                    pcd_vert5.paint_uniform_color([0, 1, 1])

                    o3d.visualization.draw_geometries([pcd_vert1, pcd_vert2, pcd_vert3, pcd_vert4, pcd_vert5,
                                                       vertebra1, vertebra2, vertebra3, vertebra4, vertebra5])

                    cur_frame += 1

print("time taken: ", time.time() - start_time)
