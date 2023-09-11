import open3d as o3d
import numpy as np
import cv2
def get_init_transformation():
    # Perform individual camera calibration and obtain camera matrices, distortion coefficients etc.
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
    ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)

    # Perform stereo calibration
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        mtx1, dist1, mtx2, dist2, gray1.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=0
    )

    # Form the transformation matrix
    trans_init = np.eye(4)
    trans_init[:3, :3] = cv2.Rodrigues(R)[0]  # Convert rotation vector to matrix
    trans_init[:3, 3] = T.T[0]

    print("Initial Transformation Matrix:")
    print(trans_init)

# Load your point clouds (Replace with your own loading code if necessary)
source = o3d.io.read_point_cloud("Pointcloud_pred1.pcd")
target = o3d.io.read_point_cloud("Pointcloud_pred0.pcd")
print("Number of points in source: ", len(source.points))
print("Number of points in target: ", len(target.points))

# Estimate normals for point clouds
source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Set ICP parameters
threshold = 10  # Distance threshold
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],  # Initial transformation
                         [0.0, 1.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

# Apply ICP
print("Applying ICP...")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)
)

# Transform the source point cloud using ICP result
source.transform(reg_p2p.transformation)

# Visualize the registration result
o3d.visualization.draw_geometries([source, target])

# Output the transformation matrix
print("Estimated transformation:")
print(reg_p2p.transformation)