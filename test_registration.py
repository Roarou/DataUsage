import open3d as o3d
import numpy as np
import cv2
import pyzed.sl as sl


def get_np_array(array):
    return np.asarray(array)


def from_np_to_pcd(points, colors):
    pcd_modified = o3d.geometry.PointCloud()
    pcd_modified.points = o3d.utility.Vector3dVector(points)
    pcd_modified.colors = o3d.utility.Vector3dVector(colors)
    return pcd_modified


def get_o3d_pcd(point_cloud):
    points = get_np_array(point_cloud.get_data())
    mask = ~np.isnan(points).any(axis=2)
    filtered_points = points[mask]
    xyz = filtered_points[:, :3].astype(np.float32)

    rgb = np.frombuffer(np.float32(filtered_points[:, 3]).tobytes(), np.uint8).reshape(-1, 4)[:, :3] / 255.0

    pcd = from_np_to_pcd(xyz, rgb)
    return pcd


def get_init_transformation(objpoints, imgpoints1, imgpoints2, gray1, gray2):
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
    return trans_init


def apply_ICP(source, target, trans_init, threshold=10):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000)
    )
    # Apply ICP
    print("Applying ICP...")
    # Transform the source point cloud using ICP result
    source.transform(reg_p2p.transformation)
    # Output the transformation matrix
    print("Estimated transformation:")
    print(reg_p2p.transformation)
    return source


init_params1 = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_linux_id=0)
init_params2 = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_linux_id=1)

cam1, cam2 = sl.Camera(), sl.Camera()
status1 = cam1.open(init_params1)
status2 = cam2.open(init_params2)

objpoints = []
imgpoints1 = []
imgpoints2 = []
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

runtime_params = sl.RuntimeParameters()
mat1, mat2 = sl.Mat(), sl.Mat()

if status1 != sl.ERROR_CODE.SUCCESS or status2 != sl.ERROR_CODE.SUCCESS:
    print("Error opening one or both cameras")
    exit(1)
try:
    while True:
        if cam1.grab(runtime_params) == sl.ERROR_CODE.SUCCESS and cam2.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            cam1.retrieve_image(mat1)
            cam2.retrieve_image(mat2)

            frame1 = mat1.get_data()
            frame2 = mat2.get_data()

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            ret1, corners1 = cv2.findChessboardCorners(gray1, (9, 6), None)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (9, 6), None)

            if ret1 and ret2:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)

                print("Found chessboard. Press 'c' to calibrate and start live stream.")

                # Wait for 'c' to be pressed
                key = cv2.waitKey(0)
                if key == ord('c'):
                    transformation = get_init_transformation(objpoints, imgpoints1, imgpoints2, gray1, gray2)
                    break

except RuntimeError as e:
    print(f"An error occurred: {e}")

print("Starting real-time livestream. \nPress q to quit.")
try:
    while True:
        if cam1.grab(runtime_params) == sl.ERROR_CODE.SUCCESS and cam2.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            cam1.retrieve_measure(mat1, sl.MEASURE.XYZRGBA)
            cam2.retrieve_measure(mat2, sl.MEASURE.XYZRGBA)

            # Convert ZED point cloud to Open3D format
            pcd1 = get_o3d_pcd(mat1)
            pcd2 = get_o3d_pcd(mat2)

            pcd2.transform(transformation)
            pcd2 = apply_ICP(pcd1, pcd2)
            pcd2 = pcd1 + pcd2
            o3d.visualization.draw_geometries([pcd2])
            # Exit on pressing 'q'
            if cv2.waitKey(1) == ord('q'):
                break


except RuntimeError as e:
    print(f"An error occurred: {e}")