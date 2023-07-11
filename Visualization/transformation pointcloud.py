import open3d as o3d
import copy
import numpy as np




def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def register(source, target,source_pcd, target_pcd):
    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target
    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source.transform(trans_init), target])
    o3d.visualization.draw_geometries([source_pcd.transform(trans_init), target_pcd])

    print(trans_init)
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03
    current_transformation = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    source.transform(reg_p2p.transformation)
    source_pcd.transform(reg_p2p.transformation)

    o3d.visualization.draw_geometries([source_pcd, target_pcd, source,target])

    threshold = 3
    reg_p2p_v2 = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, current_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    source_pcd.transform(reg_p2p_v2.transformation)
    source.transform(reg_p2p_v2.transformation)





    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    return source_pcd