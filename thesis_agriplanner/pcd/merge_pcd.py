import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis

path = "~/scan4/"


# # T optical to cameraLink
HcoTcl = rbt.conv_t_and_quat_to_h(
    [-0.0003928758669644594, 0.014598878100514412, 0.00030792783945798874],
    [-0.49577630257844, 0.501307043353262, -0.4997543001981413, 0.503132930267467],
)

cam_to_base = np.load(path + "cam_to_base_link.npy")

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh.scale(0.5, center=mesh.get_center())

# xyz = np.load(path+f'pcscan_camera_link_id_9.npy')[:,0:3]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# # pcd.transform(HcoTcl)
# o3d.visualization.draw_geometries([pcd, mesh])

pcdlist = []
for i in range(10):
    xyz = np.load(path + f"pcscan_camera_color_optical_id_{i}.npy")[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    HcamTbase = rbt.conv_t_and_quat_to_h(cam_to_base[i, 0:3], cam_to_base[i, 3:7])
    HccofTbase = HcamTbase @ HcoTcl
    pcd.transform(HccofTbase)
    pcdlist.append(pcd)

# pcdlist = pcdlist + [mesh]

pcdmerged = o3d.geometry.PointCloud()
for p in pcdlist:
    pcdmerged += p

o3d.visualization.draw_geometries([pcdmerged])
