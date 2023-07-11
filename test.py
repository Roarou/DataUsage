import copy

import open3d as o3d
from Camera.get_tf_cam import get_transformation
import numpy as np
pcd1 = o3d.io.read_point_cloud("test_0.pcd")
pcd2 = o3d.io.read_point_cloud("test_1.pcd")
path1 = "E:/Ghazi/CamParams_0_31/SN10027879.conf"
path2 = "E:/Ghazi/CamParams_0_31/SN10028650.conf"
K1, E1 = get_transformation(path1)
K2, E2 = get_transformation(path2)
pcd1 = pcd1.uniform_down_sample(10)
pcd2 = pcd2.uniform_down_sample(10)
pcd1 = pcd1.transform(E1)

#pcd1.rotate(np.linalg.inv(K1))
pcd2 = pcd2.transform(E2)

# Visualize the initial alignment
#o3d.visualization.draw_geometries([pcd1, pcd2])

source = copy.deepcopy()
# Run ICP
transformation_icp = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, max_correspondence_distance=1000,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000)
).transformation

pcd2 = pcd2.transform(transformation_icp)

# Visualize the final alignment
o3d.visualization.draw_geometries([pcd1, pcd2])

