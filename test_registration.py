from learning3d.models import PointNetLK, PointNet
import open3d as o3d
import numpy as np
import torch  # Added torch import

# Create DCP model
dcp = PointNetLK(feature_model=PointNet(), delta=1e-02, xtol=1e-07, p0_zero_mean=True, p1_zero_mean=True, pooling='max')

# Load pre-trained DCP weights if needed
# dcp.load_state_dict(torch.load('dcp_pretrained_weights.pth'))

# Read the source and template point clouds
source = o3d.io.read_point_cloud('Pointcloud_pred0.pcd')
template = o3d.io.read_point_cloud('Pointcloud_pred1.pcd')

source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
template.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Apply RANSAC-based registration
threshold = 0.02  # Threshold for matching
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])  # Initial transformation

reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    source, template, threshold,
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

# Transform 'source' point cloud based on the estimated transformation
source.transform(reg_p2p.transformation)

# Visualize
o3d.visualization.draw_geometries([source, template])

print("Estimated transformation is:")
print(reg_p2p.transformation)

# Convert the point clouds to NumPy arrays and add a batch dimension
source_np = np.asarray(source.points)
source_np = np.expand_dims(source_np, axis=0)

template_np = np.asarray(template.points)
template_np = np.expand_dims(template_np, axis=0)

# Convert NumPy arrays to torch Tensors
source_tensor = torch.tensor(source_np, dtype=torch.float32)

template_tensor = torch.tensor(template_np, dtype=torch.float32)


print(template_tensor.size())
print(source_tensor.size())
# Perform point cloud registration using DCP
transformed_source, transformation_matrix = dcp(source_tensor, template_tensor)

# Extract the transformed source point cloud
transformed_source = transformed_source[0]

# Create an Open3D point cloud from the transformed source
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(transformed_source)
final_pcd.colors = source.colors

# Merge the transformed source and template point clouds
merged_pcds = final_pcd + template

# Visualize the merged point cloud
o3d.visualization.draw_geometries([merged_pcds])
