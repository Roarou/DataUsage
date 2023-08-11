import open3d as o3d
import numpy as np

# load point cloud
pcd = o3d.io.read_point_cloud('your_file.pcd')

def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid  # center
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance  # scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

if __name__ == "__main__":
    normalize_point_cloud(pcd)
