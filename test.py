
import open3d as o3d
# Reading the point clouds:
pcd1 = o3d.io.read_point_cloud("test_0.pcd")
pcd2 = o3d.io.read_point_cloud("test_1.pcd")
print("Number of points in pcd1:", len(pcd1.points))
print("Number of points in pcd2:", len(pcd2.points))

# Preprocessing
pcd1 = pcd1.voxel_down_sample(voxel_size=0.05)
pcd2 = pcd2.voxel_down_sample(voxel_size=0.05)
print("Number of points in pcd1:", len(pcd1.points))
print("Number of points in pcd2:", len(pcd2.points))

pcd1, _ = pcd1.remove_radius_outlier(nb_points=20, radius=5)
pcd2, _ = pcd2.remove_radius_outlier(nb_points=20, radius=5)
print("Number of points in pcd1:", len(pcd1.points))
print("Number of points in pcd2:", len(pcd2.points))

pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
print("Number of points in pcd1:", len(pcd1.points))
print("Number of points in pcd2:", len(pcd2.points))

fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))
fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))

# Initial alignment
icp_coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    pcd1, pcd2,
    fpfh1, fpfh2,
    mutual_filter=True,
    max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.075)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=400)
)

pcd2.transform(icp_coarse.transformation)

# Fine-tuning
icp_fine = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2,
    max_correspondence_distance=0.05,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
)
pcd2.transform(icp_fine.transformation)

print(icp_fine.transformation)

# Visualization
o3d.visualization.draw_geometries([pcd1, pcd2])
