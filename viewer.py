# Take in a model path and the action history csv. Use open3d to visualize the model and the action poses

import open3d as o3d
import numpy as np

def rotation_matrix_from_axis_angle(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis

    rotation_matrix = np.array([
        [t*x*x + c,    t*x*y - z*s,  t*x*z + y*s],
        [t*x*y + z*s,  t*y*y + c,    t*y*z - x*s],
        [t*x*z - y*s,  t*y*z + x*s,  t*z*z + c]
    ])

    return rotation_matrix

def create_coordinate_frame(origin):
    # Create axes lines
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    axes.translate(origin)
    
    z_axis = np.array([0, 0, 1])
    pose_to_origin =  - origin
    pose_to_origin = pose_to_origin / np.linalg.norm(pose_to_origin)
    axis = np.cross(z_axis, pose_to_origin)
    angle = np.arccos(np.dot(z_axis, pose_to_origin))
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    # invert the rotation matrix
    axes.rotate(rotation_matrix, center=origin)

    return axes

def visualize_model(model_path, action_history_path, mesh_resolution=4968):
    mesh = o3d.io.read_triangle_mesh(model_path)
    # mesh.scale(0.01, center=mesh.get_center())
    mesh = mesh.translate(-mesh.get_center())
    print(f"Mesh centre: {mesh.get_center()}")
    print(f"Mesh Triangle Count: {len(mesh.triangles)}")

    mesh.paint_uniform_color([0.75, 0.75, 0])
    
    # read the action history
    print("Reading action history")
    action_history = np.genfromtxt(action_history_path, delimiter=',')
    # remove duplicates
    print(f"Action history Shape: {action_history.shape}")
    action_history = np.unique(action_history, axis=0)
    print("Unique action history")
    print(action_history.shape)
    actions = action_history
    # eliminate poses that are too close to each other
    action_history = [actions[0]]
    for i in range(1, len(actions)):
        if np.linalg.norm(actions[i] - actions[i-1]) > 1:
            action_history.append(actions[i])

    print(f"Filtered Action history Shape: {len(action_history)}")
    if len(action_history) >= 50:
        print("Too many poses to render! Truncating action history to 50...")
        action_history = action_history[:50]
    #action history is a list of poses. Generate a set of origins with the z axes pointing towards the model origin
    origins = []
    
    for pose in action_history:
        origin = create_coordinate_frame(pose)
        
        origins.append(origin)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)

    o3d.visualization.draw_geometries([mesh] + origins + [sphere])

if __name__ == "__main__":
    model_path = '/home/aman/Desktop/RL_CoveragePlanning/test_models/modified/test_1.obj'
    action_history_path = '/home/aman/Desktop/RL_CoveragePlanning/action/test_1_poses.csv'

    visualize_model(model_path, action_history_path)