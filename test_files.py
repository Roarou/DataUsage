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
import multiprocessing
# Set the start time of the script execution for performance tracking.
START_TIME = time.time()

# Define the source directory containing the depth data of spine scans.
SOURCE_DIR = r"G:\SpineDepth"

# Define the destination directory for calibration settings required by the ZED camera SDK.
CALIB_DEST_DIR = r'C:\\ProgramData\\Stereolabs\\settings'
def check_single_camparam(strings_list):
    """
    Check if there's only one instance of 'CamParam' in the filenames in the given directory.

    Parameters:
        strings_list (list): A list of strings representing filenames in the directory.

    Returns:
        tuple: A tuple containing a boolean indicating whether there was only one 'CamParam'
               and the last part of the first 'CamParam' string (if more than one).
    """
    camparam_count = 0  # Initialize the count of 'CamParam' occurrences
    b = None  # Initialize a variable
    for j, string in enumerate(strings_list):  # For each string in the list
        if string.startswith('CamParam'):  # If the string starts with 'CamParam'
            camparam_count += 1  # Increment the 'CamParam' count
            if camparam_count == 1:  # If this is the first 'CamParam'
                idx = j  # Save the index
        if camparam_count > 1:  # If there are more than one 'CamParam'
            _, _, b = strings_list[idx].split(
                '_')  # Split the first 'CamParam' string on underscores and store the last part

    return camparam_count == 1, b  # Return whether there was only one 'CamParam' and the last part of the first 'CamParam' string


def process_config_files(config_file_path):
    """
    Process configuration files by copying them to the specified path.

    Parameters:
        config_file_path (str): Path of the configuration file to be copied.

    Returns:
        bool: True if the configuration file is successfully copied, False otherwise.
    """

    if not os.path.isfile(config_file_path):
        print("Config file doesn't exist.")
        return False
    # Delete all files in path_b
    for filename in os.listdir(CALIB_DEST_DIR):
        file_path = os.path.join(CALIB_DEST_DIR, filename)
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
        shutil.copy2(config_file_path, CALIB_DEST_DIR)
        print(f'Copied {config_file_path} ')
        return True
    except PermissionError:
        print(f'PermissionError: Could not copy {config_file_path} due to insufficient permissions.')
        return False

def create_scene_with_camera(camera, rotation_matrix, mesh, extrinsics_matrix):
    """
    Initialize a rendering scene and add a camera with a given pose.

    Parameters:
    - camera: The camera object to be added to the scene.
    - rotation_matrix: A 4x4 transformation matrix representing the pose of the camera.

    Returns:
    - scene: The created scene with the camera added.
    """
    scene = pyrender.Scene()
    scene.add(mesh, pose= extrinsics_matrix)
    scene.add(camera, pose=rotation_matrix)
    return scene


def pack_rgb_to_float(color):
    # Ensure the arguments are within the 8-bit range [0, 255]
    r = color[0]
    g = color[1]
    b = color[2]
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Shift the bits of each color into the correct position and combine them
    packed = r << 16 | g << 8 | b

    # Use a view to reinterpret the integer bits as a float
    return np.asarray(packed, dtype=np.uint32).view(np.float32)


def process_and_save_pointcloud(pcd, mask_image, colors, filename):
    """
    Apply color masks to a point cloud based on segmentation masks and save the colored point cloud.

    Parameters:
    - pcd: The point cloud data as a 3D matrix.
    - mask_image: A list of binary images where non-zero pixels represent areas to color.
    - colors: A list of RGB color values to apply to the point cloud.

    Returns:
    - pcd: The point cloud with applied color.
    """
    numpy_array = pcd.get_data()
    # Iterate over each mask and apply the corresponding color.
    for i in range(5):
        mask_all_zero = np.all(mask_image[i] == 0, axis=2)
        f_indices = np.argwhere(mask_all_zero)
        for j, k in f_indices:
            rgba = numpy_array[j][k][:]
            rgba[3] = 0 # Assign new RGB values while preserving the alpha channel.
            numpy_array[j][k][:] = rgba
    for i in range(5):
        color = pack_rgb_to_float(colors[i])
        mask_all_zero = np.all(mask_image[i] == 0, axis=2)
        mask_all_zero = ~mask_all_zero
        indices = np.argwhere(mask_all_zero)
        for j, k in indices:
            rgba = numpy_array[j][k][:]
            rgba[3] = color # Assign new RGB values while preserving the alpha channel.
            numpy_array[j][k][:] = rgba

    mask = ~np.isnan(numpy_array).any(axis=2)
    filtered_points = numpy_array[mask]
    xyz = filtered_points[:, :3].astype(np.float32)
    rgb = np.frombuffer(np.float32(filtered_points[:, 3]).tobytes(), np.uint8).reshape(-1, 4)[:, :3] / 255.0
    pcd_downs= o3d.geometry.PointCloud()
    pcd_downs.points = o3d.utility.Vector3dVector(xyz)
    pcd_downs.colors = o3d.utility.Vector3dVector(rgb)
    success = o3d.io.write_point_cloud(filename, pcd_downs)
    return success


def create_rotation_matrix():
    """
    Create a composite rotation matrix that performs a 180-degree rotation about the Y and Z axes.

    Returns:
    - A 4x4 numpy array representing the composite rotation matrix.
    """
    # Define individual rotation matrices about the Y and Z axes, respectively.
    rotate1 = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0), direction=[0, 1, 0], point=[0, 0, 0])
    rotate2 = trimesh.transformations.rotation_matrix(
        angle=np.radians(180.0), direction=[0, 0, 1], point=[0, 0, 0])
    # Return the combined rotation matrix.
    return rotate1 @ rotate2


def render_scene_and_save_depth(renderer, scene, save_path, mask_name):
    """
        Render a scene to obtain a depth image and save it to the specified path.

        Parameters:
        - renderer: The offscreen renderer used for rendering the scene.
        - scene: The scene containing the objects to render.
        - save_path: Directory path where the depth image will be saved.
        - mask_name: The filename prefix for the saved depth image.
        """
    # Render the scene to get the color and depth buffer.
    _, depth = renderer.render(scene)
    # Save the depth buffer as a PNG image in the specified path.
    cv2.imwrite(os.path.join(save_path, f"{mask_name}.png"), depth)


# Define custom functions for processing
def read_tracking_file(file_path):
    """
    Reads a CSV file containing tracking information into a DataFrame.

    :param file_path: Path to the CSV file with the tracking data.
    :return: A DataFrame with the tracking data.
    """
    # Define the column names as per the file's structure
    column_names = [
        "R00", "R01", "R02", "T0",
        "R10", "R11", "R12", "T1",
        "R20", "R21", "R22", "T2",
        "R30", "R31", "R33", "T3"
    ]
    # Read the CSV file into a DataFrame with the specified column names
    return pd.read_csv(file_path, sep=',', header=None, names=column_names)


def create_and_save_directory(base_path, specimen, recording, camera_num, cur_frame):
    """
    Creates a directory path for saving data and ensures the directory exists.

    :param base_path: The base path where directories will be created.
    :param specimen: The specimen identifier.
    :param recording: The recording session number.
    :param camera_num: The camera number being processed.
    :param cur_frame: The current frame number.
    :return: The directory path where data will be saved.
    """
    # Build the directory path
    save_data_dir = os.path.join(base_path, specimen, recording,
                                 f"pointcloud_cam_{camera_num}_recording_{cur_frame}")
    print(f"Saving files to: {save_data_dir}")
    # Create the directory if it doesn't exist, with the option to not raise an error if it already exists
    os.makedirs(save_data_dir, exist_ok=True)
    return save_data_dir


def read_and_transform_vertebra(base_path, specimen, vertebra_name, transformation):
    """
    Reads a vertebra model from an STL file, computes its vertex normals, and applies a transformation.

    :param base_path: The base path to the specimen data.
    :param specimen: The specimen identifier.
    :param vertebra_name: The name of the vertebra to be transformed.
    :param transformation: The transformation matrix to be applied.
    :return: The transformed vertebra mesh.
    """
    # Read the STL file into an Open3D triangle mesh object
    vertebra = o3d.io.read_triangle_mesh(os.path.join(base_path, specimen, "STL", f"{vertebra_name}.stl"))
    # Compute the normals for the vertices of the mesh
    vertebra.compute_vertex_normals()
    # Apply the transformation to the vertebra mesh
    vertebra.transform(transformation)
    return vertebra


def create_transformation_matrix(df_row):
    """
    Creates a transformation matrix from a DataFrame row containing rotation and translation data.

    :param df_row: A Pandas Series with rotation matrix R and translation vector T components.
    :return: A 4x4 NumPy array representing the transformation matrix.
    """
    # Extract the rotation and translation values to construct the transformation matrix
    return np.array([
        [df_row["R00"], df_row["R01"], df_row["R02"], df_row["T0"]],
        [df_row["R10"], df_row["R11"], df_row["R12"], df_row["T1"]],
        [df_row["R20"], df_row["R21"], df_row["R22"], df_row["T2"]],
        [0, 0, 0, 1]  # The bottom row of a transformation matrix is always [0, 0, 0, 1]
    ])

def main():
    # Main code [ "Specimen_2","Specimen_3","Specimen_4", "Specimen_5", "Specimen_6", "Specimen_7","Specimen_8","Specimen_9",
    specimens = [ "Specimen_1",  "Specimen_10"]
    camera_nums = [0, 1]
    for specimen, camera_num in product(specimens, camera_nums):
        specimen_directory_path = os.path.join(SOURCE_DIR, specimen)
        print(f"Processing specimen: {specimen}, camera: {camera_num}")
        flag, b = check_single_camparam(os.listdir(specimen_directory_path))
        list_dir = os.listdir(os.path.join(specimen_directory_path, 'Recordings'))
        list_dir = sorted(list_dir, key=lambda record: int(record.split('Recording')[1]))
        # Process each subdirectory in the 'Recordings' directory
        for k, recording in enumerate(list_dir):
            video_dir = os.path.join(SOURCE_DIR, specimen, f"Recordings/{recording}/Video_{camera_num}.svo")
            print(video_dir)
            conf_path = os.path.join(specimen_directory_path, 'Calib')

            if not flag:
                if k > int(b):  # If this subdirectory's index is greater than the last part of the first 'CamParam'
                    conf_path = conf_path + '_b'  # Add '_b' to the configuration path
                else:
                    conf_path = conf_path + '_a'  # Add '_a' to the configuration path
            print(conf_path)

            if camera_num == 0:
                calib_src_dir = os.path.join(conf_path, 'SN10027879.conf')
            elif camera_num == 1:
                calib_src_dir = os.path.join(conf_path, 'SN10028650.conf')
            if not process_config_files(config_file_path=calib_src_dir):
                print('Failed to process config files.')
                return False


            tracking_file = os.path.join(SOURCE_DIR, specimen, f"Recordings/{recording}/Poses_{camera_num}.txt")
            df = read_tracking_file(tracking_file)

            input_type = sl.InputType()
            input_type.set_from_svo_file(video_dir)
            init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False, optional_settings_path=SOURCE_DIR)

            zed = sl.Camera()
            status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit()

            runtime_parameters = sl.RuntimeParameters()

            zed_pose = sl.Pose()
            calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
            nb_frames = zed.get_svo_number_of_frames()
            nb_frames = 20
            cur_frame = 0
            # Prepare arguments for each frame

            while cur_frame < nb_frames:

                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    image, depth = sl.Mat(), sl.Mat()

                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    pcd = sl.Mat()
                    zed.retrieve_measure(pcd, sl.MEASURE.XYZRGBA)

                    save_data_dir = os.path.join(r"C:\Users\cheg\PycharmProjects\DataUsage", specimen,
                                                 recording,
                                                 "pointcloud_cam_{}_recording_{}".format(camera_num, cur_frame))
                    filename = f'L:\Pointcloud\Pointcloud_{cur_frame}_spec_{specimen}_vid_{camera_num}_{recording}.pcd'
                    print("saving files to:{}".format(save_data_dir))
                    if not os.path.exists(save_data_dir):
                        os.makedirs(save_data_dir)

                    image.write(os.path.join(save_data_dir, f'image_frame{cur_frame}.png'))
                    depth.write(os.path.join(save_data_dir, f'depth_frame{cur_frame}.png'))

                    # Read and transform vertebrae
                    vertebrae = []

                    for i in range(5):
                        df_row = df.iloc[5 * cur_frame + i]
                        transformation_matrix = create_transformation_matrix(df_row)
                        vertebra = read_and_transform_vertebra(SOURCE_DIR, specimen, f"L{i + 1}", transformation_matrix)
                        vertebrae.append(vertebra)

                    # Combine all vertebrae
                    # spine = vertebra[0] + vertebra[1] + vertebra[2] + vertebra[3] + vertebra[4]
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
                        o3d.io.write_triangle_mesh(os.path.join(save_data_dir, f"transformed_vertebra{i}_frame{cur_frame}.stl"), vert)
                        mask_stl = trimesh.load(os.path.join(save_data_dir, f"transformed_vertebra{i}_frame{cur_frame}.stl"))
                        mesh.append(pyrender.Mesh.from_trimesh(mask_stl))

                    # Create rotation matrix
                    rotation_matrix = create_rotation_matrix()

                    # Create renderers and scenes
                    renderers = [pyrender.OffscreenRenderer(1920, 1080) for _ in range(5)]
                    scenes = [create_scene_with_camera(camera, rotation_matrix, mesh[i], extrinsics_matrix) for i in range(5)]

                    # Render scenes and save depth images
                    for i, (renderer, scene) in enumerate(zip(renderers, scenes)):
                        render_scene_and_save_depth(renderers[0], scene, save_data_dir, f"mask{i + 1}_frame{cur_frame}")

                    mask_image1 = cv2.imread(os.path.join(save_data_dir, f"mask1_frame{cur_frame}.png"))
                    mask_image2 = cv2.imread(os.path.join(save_data_dir, f"mask2_frame{cur_frame}.png"))
                    mask_image3 = cv2.imread(os.path.join(save_data_dir, f"mask3_frame{cur_frame}.png"))
                    mask_image4 = cv2.imread(os.path.join(save_data_dir, f"mask4_frame{cur_frame}.png"))
                    mask_image5 = cv2.imread(os.path.join(save_data_dir, f"mask5_frame{cur_frame}.png"))

                    mask_images = [mask_image1, mask_image2, mask_image3, mask_image4, mask_image5]
                    # Coloring point clouds
                    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]])
                    assert process_and_save_pointcloud(pcd, mask_images, colors, filename)
                    #pcd = o3d.io.read_point_cloud(filename=filename)
                    #o3d.visualization.draw_geometries([pcd])
                    cur_frame += 1
if __name__ == '__main__':
    main()
