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

def visualize_model(model_path, action_history_path):
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    # read the action history
    action_history = np.genfromtxt(action_history_path, delimiter=',')
    if len(action_history) >= 10:
        action_history = action_history[:10]
    #action history is a list of poses. Generate a set of origins with the z axes pointing towards the model origin
    origins = []
    
    for pose in action_history:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        origin.scale(0.03, center=pose)
        # rotate such that the z axis points towards the model origin
        z_axis = np.array([0, 0, 1])
        pose_to_origin =  mesh.get_center() - pose
        pose_to_origin = pose_to_origin / np.linalg.norm(pose_to_origin)
        axis = np.cross(z_axis, pose_to_origin)
        angle = np.arccos(np.dot(z_axis, pose_to_origin))
        rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
        # invert the rotation matrix
        origin.rotate(rotation_matrix, center=pose)

        # origin.paint_uniform_color([1, 0, 0])
        
        origins.append(origin)

    o3d.visualization.draw_geometries([mesh] + origins)

if __name__ == "__main__":
    model_path = '/home/aman/Desktop/RL_CoveragePlanning/viewpointPlaygroundEnv/meshes/stanford_bunny.obj'
    action_history_path = '/home/aman/Desktop/RL_CoveragePlanning/action/poses.csv'

    visualize_model(model_path, action_history_path)