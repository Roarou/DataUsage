import numpy as np
from configparser import ConfigParser

# Load the conf files
def load_conf_file(filepath):
    parser = ConfigParser()
    parser.read(filepath)

    params = {}
    for section in parser.sections():
        params[section] = {k: float(v) for k, v in parser.items(section)}
    return params

# Compute the transformation matrix from the parameters
def get_transformation(path, resolution ='2K'):
    params = load_conf_file(path)
    fx = params[f'LEFT_CAM_{resolution}']['fx']
    fy = params[f'LEFT_CAM_{resolution}']['fy']
    cx = params[f'LEFT_CAM_{resolution}']['cx']
    cy = params[f'LEFT_CAM_{resolution}']['cy']

    baseline = params['STEREO']['BaseLine']
    RX = params['STEREO'][f'RX_{resolution}']
    RZ = params['STEREO'][f'RZ_{resolution}']

    # Assemble the intrinsic and extrinsic matrices
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, np.cos(RX), -np.sin(RX)], [0, np.sin(RX), np.cos(RX)]])
    R = np.dot(R, np.array([[np.cos(RZ), -np.sin(RZ), 0], [np.sin(RZ), np.cos(RZ), 0], [0, 0, 1]]))
    T = np.array([[1, 0, 0, -baseline / 2], [0, 1, 0, 0], [0, 0, 1, 0]])

    # Compute the transformation matrix
    transformation = np.linalg.inv(K).dot(R).dot(T)
    return transformation

