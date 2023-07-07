import numpy as np
from configparser import ConfigParser
from test2 import combine_rotation_translation
# Load the conf files
def load_conf_file(filepath):
    parser = ConfigParser()
    parser.read(filepath)

    params = {}
    for section in parser.sections():
        params[section] = {k: float(v) for k, v in parser.items(section)}
    return params

# Compute the transformation matrix from the parameters
def get_transformation(path, resolution = '2K'):
    params = load_conf_file(path)
    # Load focal length
    fx = params[f'LEFT_CAM_{resolution}']['fx']
    fy = params[f'LEFT_CAM_{resolution}']['fy']
    # Load optical center
    cx = params[f'LEFT_CAM_{resolution}']['cx']
    cy = params[f'LEFT_CAM_{resolution}']['cy']
    # Load baseline
    baseline = params['STEREO']['baseline']
    # Load rotation angles
    RX = params['STEREO'][f'rx_{resolution.lower()}']
    RZ = params['STEREO'][f'rx_{resolution.lower()}']

    # Assemble the intrinsic and extrinsic matrices
    # Assemble intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # Rotation matrix about x
    R = np.array([[1, 0, 0], [0, np.cos(RX), -np.sin(RX)], [0, np.sin(RX), np.cos(RX)]])
    # Rotation matrix about z
    R = np.dot(R, np.array([[np.cos(RZ), -np.sin(RZ), 0], [np.sin(RZ), np.cos(RZ), 0], [0, 0, 1]]))
    # Translation matrix
    T = np.array([-baseline / 2, 0, 0])
    # Extrinsic matrix
    E = combine_rotation_translation(R, T)
    # Compute the transformation matrix
    transformation = K.dot(E)
    transformation = np.vstack((transformation, [0, 0, 0, 1]))
    return transformation

