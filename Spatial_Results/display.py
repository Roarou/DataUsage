import numpy as np
import open3d as o3d

pose_file = "E:/Ghazi/Recordings/Recording0/Poses_0.txt"  # Replace with the actual path to your pose file

# Load poses from file
poses = np.loadtxt(pose_file, delimiter=',').reshape(-1, 16)

# Extract transformation matrices
transformation_matrices = poses.reshape(-1, 4, 4)

# Create a visualizer and set up the scene
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)  # Specify the window size

# Add coordinate axes to visualize the reference frame
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axes)

# Add each pose as a coordinate frame to the scene
coordinate_frames = []
for i in range(len(transformation_matrices)):
    pose = transformation_matrices[i]
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coordinate_frame.transform(pose)
    vis.add_geometry(coordinate_frame)
    coordinate_frames.append(coordinate_frame)

# Set the camera viewpoint
vis.get_view_control().set_front([0, 0, -1])
vis.get_view_control().set_lookat([0, 0, 0])
vis.get_view_control().set_up([0, -1, 0])
vis.get_view_control().set_zoom(0.8)

# Run the visualization loop
while True:
    # Update the geometries in the scene
    for coordinate_frame in coordinate_frames:
        vis.update_geometry(coordinate_frame)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()