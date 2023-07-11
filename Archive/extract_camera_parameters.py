def parse_camera_parameters(conf_file_path):
    camera_parameters = {}
    current_camera = None
    with open(conf_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_camera = line[1:-1]
                camera_parameters[current_camera] = {}
            elif '=' in line and current_camera is not None:
                key, value = line.split('=')
                camera_parameters[current_camera][key] = float(value)
    return camera_parameters


if __name__ == "__main__":
    conf_file_path = "E:/Ghazi/CamParams_0_31/SN10027879.conf"
    # Parse the camera parameters from the conf file
    camera_parameters = parse_camera_parameters(conf_file_path)
    # Extract the parameters for each camera
    left_cam_2k_params = camera_parameters['LEFT_CAM_2K']
    right_cam_2k_params = camera_parameters['RIGHT_CAM_2K']
    stereo_params = camera_parameters['STEREO']
    left_cam_fhd_params = camera_parameters['LEFT_CAM_FHD']
    right_cam_fhd_params = camera_parameters['RIGHT_CAM_FHD']
    left_cam_hd_params = camera_parameters['LEFT_CAM_HD']
    right_cam_hd_params = camera_parameters['RIGHT_CAM_HD']
    left_cam_vga_params = camera_parameters['LEFT_CAM_VGA']
    right_cam_vga_params = camera_parameters['RIGHT_CAM_VGA']

    print(left_cam_vga_params)
