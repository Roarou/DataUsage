import open3d as o3d
import numpy as np
path = r"G:\SpineDepth\Specimen_1\Recordings\Recording0\pointcloud\Video_0\Pointcloud_0.pcd"

pcd = o3d.io.read_point_cloud(path)
print(np.asarray(pcd.colors))
o3d.visualization.draw_geometries([pcd])