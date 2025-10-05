import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
from merge_harvest import transform_pcd, crop_pcd


pcd = o3d.io.read_point_cloud("~/meshscan/fused_point_cloud_wrt_first_tcp.ply")
pcd = crop_pcd(pcd)
o3d.visualization.draw([pcd])


pcd = o3d.io.read_point_cloud("~/scan7/multiway_registration_full.ply")
o3d.visualization.draw([pcd])


xyz = np.load("./datasave/grasp_poses/pose_array.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.estimate_normals()
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location()
pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([pcd])
