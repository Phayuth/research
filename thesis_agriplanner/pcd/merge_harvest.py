import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from spatial_geometry.spatial_transformation import RigidBodyTransformation as rbt
import cv2
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import time


def tcp_to_h(tcp):
    R, _ = cv2.Rodrigues(np.array(tcp[3:6]))
    H = np.eye(4) @ rbt.ht(tcp[0], tcp[1], tcp[2])
    H[:3, :3] = R
    return H


def get_h_camleft_to_base(tcptool0Tobase):
    HcamleftToTool0 = rbt.ht(0, -0.01, 0) @ rbt.hrx(np.pi / 2) @ rbt.hrz(np.pi / 2)
    Htool0Tobase = tcp_to_h(tcptool0Tobase)
    return Htool0Tobase @ HcamleftToTool0


path = "~/scan7/"
tool0_to_base = np.load(path + "tool0_to_base.npy")


# # plot
# ax = pt.plot_transform(name="base", s=0.5)
# for tcpi in tool0_to_base:
#     Htool0Tobase = tcp_to_h(tcpi)
#     pt.plot_transform(ax, Htool0Tobase, s=0.1, name="TCP")
#     # HcamleftTobase = get_h_camleft_to_base(tcpi)
#     # pt.plot_transform(ax, HcamleftTobase, s=0.1, name="CAML")
# plt.show()


def numpy_to_o3dpointcloud(xyz, rgb=None, savetoplypath=None):
    assert (
        xyz.shape == rgb.shape
    ), f"Point XYZ and RGB shape is not equal {xyz.shape}, {rgb.shape}. Point Cloud will have no color."
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        rgb = rgb / 255
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if savetoplypath:
        o3d.io.write_point_cloud(savetoplypath, pcd)
    return pcd


def transform_pcd(pcdlist):
    for i, pcdi in enumerate(pcdlist):
        tool0tobase = tool0_to_base[i]
        HcamleftTobase = get_h_camleft_to_base(tool0tobase)
        pcdi.transform(HcamleftTobase)


def merged_o3dpointcloud(pcdlist):
    pcdmerged = o3d.geometry.PointCloud()
    for p in pcdlist:
        pcdmerged += p
    return pcdmerged


def crop_pcd(pcd):
    xmin = 0.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0
    zmin = -1.0
    zmax = 1.0
    min_bound = np.array(
        [xmin, ymin, zmin]
    )  # Minimum coordinates of the bounding box
    max_bound = np.array(
        [xmax, ymax, zmax]
    )  # Maximum coordinates of the bounding box
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)


# single pcd
i = 7
pcdnpy = np.load(path + f"pcd_leftframe_id_{i}.npy")
pcd = numpy_to_o3dpointcloud(pcdnpy[:, 0:3], pcdnpy[:, 3:6])
# pcd = crop_pcd(pcd)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
origin.scale(0.5, center=origin.get_center())
# o3d.visualization.draw_geometries([pcd, origin])
o3d.visualization.draw_geometries(
    [pcd],
    zoom=0.82000000000000028,
    front=[-0.77038850042737828, -0.5108504401011742, -0.38148838286072628],
    lookat=[3.0194881983280792, 1.1271710421207048, 0.72749091588348103],
    up=[-0.47058868018676925, 0.051896855322870195, 0.88082518724640912],
)

# multiple pcd no transform
pcdnpylist = [np.load(path + f"pcd_leftframe_id_{i}.npy") for i in range(8)]
pcdlist = [numpy_to_o3dpointcloud(py[:, 0:3], py[:, 3:6]) for py in pcdnpylist]
origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
origin.scale(0.5, center=origin.get_center())
o3d.visualization.draw_geometries(pcdlist + [origin])

# multiple pcd with transform
pcdnpylist = [np.load(path + f"pcd_leftframe_id_{i}.npy") for i in range(8)]
pcdlist = [numpy_to_o3dpointcloud(py[:, 0:3], py[:, 3:6]) for py in pcdnpylist]
transform_pcd(pcdlist)
pcdlist = [crop_pcd(pc) for pc in pcdlist]
origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
origin.scale(0.5, center=origin.get_center())
o3d.visualization.draw_geometries(pcdlist + [origin])


def pcd_preprocessing(voxel_size=0.0):
    for i, pcdi in enumerate(pcdlist):
        pcdi.voxel_down_sample(voxel_size=voxel_size)
        pcdi.estimate_normals()
        pcdi.normalize_normals()
        pcdi.orient_normals_towards_camera_location()
        pcdi.remove_radius_outlier(nb_points=16, radius=0.05)
        # o3d.io.write_point_cloud(path + f"pcd_leftframe_id_{i}.ply", pcdi)


from multiway_reg import full_registration

voxel_size = 0.02
timea = time.perf_counter_ns()
pcd_preprocessing(voxel_size)
timeb = time.perf_counter_ns()
preproctime = (timeb - timea) * 1e-6
print(f"> preproctime: {preproctime}")
o3d.visualization.draw(pcdlist)

print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(
        pcdlist,
        max_correspondence_distance_coarse,
        max_correspondence_distance_fine,
    )

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0,
)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

print("Transform points and display")
for point_id in range(len(pcdlist)):
    print(pose_graph.nodes[point_id].pose)
    pcdlist[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw(pcdlist)
timec = time.perf_counter_ns()
regtime = (timec - timeb) * 1e-6
print(f"> regtime: {regtime}")

# combine pcd and save
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcdlist)):
    #     pcdlist[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcdlist[point_id]
# pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

pcd_combined = crop_pcd(pcd_combined)
timed = time.perf_counter_ns()
postproctime = (timed - timec) * 1e-6
print(f"> postproctime: {postproctime}")


o3d.io.write_point_cloud("multiway_registration_full.ply", pcd_combined)
o3d.visualization.draw([pcd_combined])

# o3d.io.write_point_cloud("multiway_registration_downsampled.ply", pcd_combined_down)
# o3d.visualization.draw_geometries([pcd_combined_down])
